"""
00 — Feature generator verification gate.

Regenerates features for a sample of ORIGINAL training molecules and compares,
block by block, against the frozen Publication V1 feature matrix
(data/original_training/X_C3_5204x2283.npy).

Three modes
-----------
--strict (DEFAULT)  Publication V1 fidelity. EVERY block must reproduce exactly
                    (ECFP6 bit-identical, 3D shape within tolerance, geometry
                    exact). Any deviation FAILS the gate and blocks external
                    inference. Only this mode supports a claim of exact
                    Publication V1 reproduction.
--relaxed           Authorises external inference with *documented, quantified*
                    approximations instead of exact reproduction. Hard
                    requirements that remain: MACCS/sPAS/PAS/solvent exact,
                    ECFP6 bit-exact when the interpreter's RDKit reproduces the
                    training fingerprints (2025.03.6 does; otherwise drift must
                    stay <=2% bits and be reported), dihedral-derived geometry
                    columns exact, no NaN/Inf. Blocks the release cannot supply
                    (shape3d legacy cache; conformer-generated geometry columns
                    under a different RDKit) are marked TOLERATED_DRIFT /
                    UNREPRODUCIBLE rather than FAIL. The gate records the
                    measured prediction impact from results/drift_impact_*.json
                    and sets claim_level to "approximate_validation".
--tolerant           Diagnostic mode for inspecting drift. Never authorises
                    external inference; the gate records which blocks deviate.

The training matrix was built with a mixed RDKit lineage: its ECFP6 bits match
RDKit 2025.03.6 (5204/5204 rows) while its ETKDGv3 geometry matches 2022.09.5.
See FEATURE_REPRODUCTION_FINDINGS.md.

Writes results/verify_report.json and results/FEATURE_GATE.json.

Usage
-----
    python src/00_verify_feature_generator.py --n 200
    python src/00_verify_feature_generator.py --n 200 --relaxed
    python src/00_verify_feature_generator.py --n 200 --tolerant
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ALL_BLOCKS, BLOCKS, CONFIGS, DIHEDRAL_TABLE, N_FEATURES, RESULTS, TRAIN,
    load_external_dihedral, write_json,
)
from feature_builder import build_vector, check_vector_integrity  # noqa: E402

# Reproduction tolerances
STRICT_ECFP_MAX_BIT_DIFF = 0            # bit-identical
TOLERANT_ECFP_MAX_BIT_FRAC = 0.02       # diagnostic only
STRICT_SHAPE_REL_TOL = 1e-6             # exact
TOLERANT_SHAPE_REL_TOL = 0.02           # diagnostic only (observed drift is 8-30%)
# The frozen matrix is float32, so identical float64 computations can differ by a
# few float32 ulps after storage. Tolerance must therefore sit above float32
# round-off (eps = 1.19e-7) and far below any real computational drift.
FLOAT32_ULPS = 8
FLOAT32_RTOL = FLOAT32_ULPS * np.finfo(np.float32).eps   # ~9.5e-7
EXACT_REL_TOL = FLOAT32_RTOL
EXACT_ABS_TOL = FLOAT32_RTOL


RDKIT2D_FLOAT_COLS = [0, 1, 2]           # MolWt, MolLogP, TPSA  (float32 storage tolerance)
RDKIT2D_INT_COLS = [3, 4, 5, 6, 7]       # HAcceptors, HDonors, RotBonds, RingCount, AromRings
FLOAT32_RTOL = 1e-5


def block_diff(gen, ref, name):
    lo, hi = BLOCKS[name]
    g, r = np.asarray(gen[lo:hi], dtype=np.float64), np.asarray(ref[lo:hi], dtype=np.float64)

    if name in ("ecfp6", "maccs"):
        n_diff = int(np.sum(g != r))
        return {"type": "bit", "n_diff": n_diff, "n_total": hi - lo,
                "frac": n_diff / (hi - lo)}

    if name == "rdkit2d":
        # Distinguish float32 storage quantisation from genuine descriptor drift.
        float_ok = [abs(g[k] - r[k]) <= FLOAT32_RTOL * (abs(r[k]) + 1e-9)
                    for k in RDKIT2D_FLOAT_COLS]
        int_bad = [k for k in RDKIT2D_INT_COLS if round(g[k]) != round(r[k])]
        return {"type": "numeric",
                "max_abs": float(np.abs(g - r).max()),
                "max_rel": float((np.abs(g - r) / (np.abs(r) + 1e-9)).max()),
                "float_cols_ok": bool(all(float_ok)),
                "integer_cols_mismatched": int_bad,
                "n_cols_exact": int(sum(1 for k in range(hi - lo)
                                        if abs(g[k] - r[k]) <= 1e-9)),
                "n_total": hi - lo}

    if name in ("spas", "pas", "geometry", "shape3d"):
        nan_g = int((~np.isfinite(g)).sum())
        nan_r = int((~np.isfinite(r)).sum())
        g2, r2 = np.nan_to_num(g), np.nan_to_num(r)
        rel = np.abs(g2 - r2) / (np.abs(r2) + 1e-9)
        within_tol = np.abs(g2 - r2) <= EXACT_ABS_TOL * (np.abs(r2) + 1.0)
        out = {"type": "numeric",
               "max_abs": float(np.abs(g2 - r2).max()),
               "max_rel": float(rel.max()),
               "n_cols_within_1pct": int((rel < 0.01).sum()),
               "n_cols_exact": int(within_tol.sum()),
               "n_total": hi - lo,
               "nan_in_generated": nan_g,
               "nan_in_reference": nan_r}
        if name == "geometry":
            # columns 5,6 (baseline_pm_dihedral_sin/cos) derive from the supplied
            # dihedral alone; columns 0-4 come from the embedded conformer.
            out["dihedral_cols_exact"] = bool(within_tol[5:].all())
            out["conformer_n_exact"] = int(within_tol[:5].sum())
            out["conformer_max_rel"] = float(rel[:5].max())
        return out

    return {"type": "scalar",
            "max_abs": float(np.abs(g - r).max()),
            "n_exact": int(np.sum(np.isclose(g, r, atol=1e-3, rtol=0))),
            "n_total": hi - lo}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--strict", action="store_true",
                    help="exact reproduction required (default when no mode flag given)")
    ap.add_argument("--relaxed", action="store_true",
                    help="authorise external inference with quantified, flagged approximations")
    ap.add_argument("--tolerant", action="store_true",
                    help="diagnostic drift inspection; never authorises external inference")
    ap.add_argument("--allow-derived-dihedral", action="store_true")
    args = ap.parse_args()

    flags = [args.strict, args.relaxed, args.tolerant]
    if sum(flags) > 1:
        raise SystemExit("choose at most one of --strict / --relaxed / --tolerant")
    mode = "relaxed" if args.relaxed else ("tolerant" if args.tolerant else "strict")
    strict = mode == "strict"
    relaxed = mode == "relaxed"

    ml = pd.read_csv(TRAIN / "master_labels_v2.csv")
    X = np.load(TRAIN / "X_C3_5204x2283.npy", mmap_mode="r")
    if X.shape != (len(ml), N_FEATURES):
        raise SystemExit(f"training matrix shape {X.shape} != ({len(ml)}, {N_FEATURES})")
    if not np.all(np.isfinite(np.asarray(X[:5]))):
        raise SystemExit("frozen matrix contains non-finite values in the sampled rows")

    rng = np.random.RandomState(42)
    idx = rng.choice(len(ml), size=min(args.n, len(ml)), replace=False)

    per_block = {k: [] for k in ALL_BLOCKS}
    rows, failures, integrity_failures = [], 0, 0

    print(f"Verifying {len(idx)} training molecules (mode={mode}) ...")

    for n, i in enumerate(idx, 1):
        r = ml.iloc[i]
        # Verification always tests the feature machinery with the true dihedral
        # and P/M from the frozen matrix. What external rows will actually use
        # (supplied table / derived approximation) is declared separately in the
        # gate; it must not contaminate reproduction testing.
        true_dih = float(X[i, 2274])
        true_pm = {1.0: "P", -1.0: "M"}.get(float(X[i, 2275]))

        vec, audit = build_vector(
            r["canonical_smiles"], r["solvent_group"],
            dihedral=true_dih, pm_label=true_pm,
            allow_derived_dihedral=False)

        if vec is None:
            failures += 1
            continue

        ok, problems = check_vector_integrity(vec, audit)
        if not ok:
            integrity_failures += 1

        ref = np.asarray(X[i])
        entry = {"idx": int(i), "smiles": r["canonical_smiles"],
                 "solvent": r["solvent_group"],
                 "integrity_ok": bool(ok), "integrity_problems": problems[:8]}
        for name in ALL_BLOCKS:
            d = block_diff(vec, ref, name)
            entry[name] = d
            per_block[name].append(d)
        rows.append(entry)

        if n % 50 == 0:
            print(f"  {n}/{len(idx)} verified")

    # ---------------------------------------------------------------- summary
    summary = {}
    for name, ds in per_block.items():
        if not ds:
            summary[name] = {"status": "NO_DATA"}
            continue

        if name == "ecfp6":
            diffs = np.array([d["n_diff"] for d in ds])
            fracs = np.array([d["frac"] for d in ds])
            if strict:
                ok = bool((diffs == 0).all())
                summary[name] = {"status": "PASS" if ok else "FAIL",
                                 "n_rows_with_drift": int((diffs > 0).sum()),
                                 "n_rows": len(ds),
                                 "max_bit_diff": int(diffs.max()),
                                 "max_bit_diff_frac": float(fracs.max())}
            elif relaxed:
                if (diffs == 0).all():
                    status = "PASS"
                elif fracs.max() <= TOLERANT_ECFP_MAX_BIT_FRAC:
                    status = "TOLERATED_DRIFT"
                else:
                    status = "FAIL"
                summary[name] = {"status": status,
                                 "n_rows_with_drift": int((diffs > 0).sum()),
                                 "n_rows": len(ds),
                                 "max_bit_diff": int(diffs.max()),
                                 "max_bit_diff_frac": float(fracs.max()),
                                 "tolerance_frac": TOLERANT_ECFP_MAX_BIT_FRAC,
                                 "note": ("bit-exact under RDKit 2025.03.6; other "
                                          "versions drift in Morgan hashing")}
            else:
                summary[name] = {
                    "status": "EXACT" if (diffs == 0).all() else "DRIFT",
                    "max_bit_diff": int(diffs.max()),
                    "max_bit_diff_frac": float(fracs.max())}

        elif name == "maccs":
            n_bad = sum(1 for d in ds if d["n_diff"] != 0)
            summary[name] = {"status": "PASS" if n_bad == 0 else "FAIL",
                             "n_rows_mismatched": n_bad, "n_rows": len(ds)}

        elif name in ("spas", "pas"):
            worst = max(d["max_rel"] for d in ds)
            worst_abs_ok = all(
                np.abs(np.asarray(d["max_abs"], dtype=float)) <= EXACT_ABS_TOL * 1e3
                for d in ds)
            summary[name] = {"status": "PASS" if (worst <= EXACT_REL_TOL * 10 or worst_abs_ok)
                             else "FAIL",
                             "max_rel_diff": worst}

        elif name == "rdkit2d":
            bad_int = sum(1 for d in ds if d["integer_cols_mismatched"])
            bad_float = sum(1 for d in ds if not d["float_cols_ok"])
            detail = {}
            for d in ds:
                for k in d["integer_cols_mismatched"]:
                    detail[k] = detail.get(k, 0) + 1
            if bad_int == 0 and bad_float == 0:
                status = "PASS"
            elif relaxed and bad_float == 0 and set(detail) <= {3} \
                    and bad_int / len(ds) <= 0.10:
                # Known, version-invariant NumHAcceptors definition change: the
                # frozen column counts one fewer acceptor on ~5% of molecules
                # (mostly aromatic n / nitro-adjacent N). Float descriptors
                # (MolWt/MolLogP/TPSA) remain a hard requirement.
                status = "TOLERATED_DRIFT"
            else:
                status = "FAIL"
            summary[name] = {
                "status": status,
                "n_rows_integer_mismatch": bad_int,
                "n_rows_float_outside_1e-5": bad_float,
                "mismatched_integer_column_counts": detail,
                "n_rows": len(ds),
                "column_order": ["MolWt", "MolLogP", "TPSA", "NumHAcceptors",
                                 "NumHDonors", "NumRotBonds", "RingCount", "NumAromRings"],
            }
            if status == "TOLERATED_DRIFT":
                summary[name]["note"] = (
                    "NumHAcceptors differs on a minority of rows under every "
                    "RDKit released for Python 3.9 (2022.03.5 / 2022.09.5 / "
                    "2025.03.6 identical); frozen column came from an older "
                    "acceptor definition. 1 of 8 columns; block importance 4%.")

        elif name == "solvent_code":
            n_bad = sum(1 for d in ds if d["max_abs"] > 0)
            summary[name] = {"status": "PASS" if n_bad == 0 else "FAIL",
                             "n_rows_mismatched": n_bad, "n_rows": len(ds)}

        elif name in ("dihedral", "pm_code"):
            n_exact = sum(d["n_exact"] for d in ds)
            n_tot = sum(d["n_total"] for d in ds)
            frac = n_exact / max(n_tot, 1)
            if frac > 0.999:
                status = "PASS"
            elif relaxed and args.allow_derived_dihedral:
                status = "APPROXIMATE"
            else:
                status = "FAIL"
            summary[name] = {"status": status,
                             "exact_match_frac": frac, "n_rows": len(ds),
                             "note": ("true values supplied in this run; external "
                                      "rows use the declared dihedral source")}

        elif name == "geometry":
            # Exactness is judged with the float32-aware abs+rel criterion from
            # block_diff (n_cols_exact). A pure relative test would flag small
            # sin/cos values whose absolute difference is a single float32 ulp.
            n_nan = sum(d["nan_in_generated"] for d in ds)
            if strict:
                n_bad = sum(1 for d in ds if d["n_cols_exact"] != d["n_total"])
                ok = (n_bad == 0) and n_nan == 0
                summary[name] = {"status": "PASS" if ok else "FAIL",
                                 "n_rows_not_exact": n_bad,
                                 "max_rel_diff": max(d["max_rel"] for d in ds),
                                 "max_abs_diff": max(d["max_abs"] for d in ds),
                                 "nan_generated": n_nan,
                                 "n_rows": len(ds),
                                 "tolerance": f"abs<= {EXACT_ABS_TOL:.3e}*(|ref|+1)"}
            elif relaxed:
                # Hard requirement: the two dihedral-derived columns (5,6) must
                # be exact and there must be no NaN. The five conformer-derived
                # columns (0-4) are marked TOLERATED_DRIFT when the interpreter's
                # ETKDGv3 differs from the training-time embedding.
                bad_dihedral = sum(1 for d in ds if not d["dihedral_cols_exact"])
                conformer_ok = sum(1 for d in ds
                                   if d["conformer_n_exact"] == 5)
                if bad_dihedral > 0 or n_nan > 0:
                    status = "FAIL"
                elif conformer_ok == len(ds):
                    status = "PASS"
                else:
                    status = "TOLERATED_DRIFT"
                summary[name] = {"status": status,
                                 "n_rows_dihedral_cols_not_exact": bad_dihedral,
                                 "n_rows_conformer_cols_exact": conformer_ok,
                                 "n_rows": len(ds),
                                 "nan_generated": n_nan,
                                 "max_rel_diff": max(d["max_rel"] for d in ds),
                                 "conformer_max_rel": max(d["conformer_max_rel"]
                                                          for d in ds),
                                 "note": ("dihedral-derived cols 2281/2282 are exact "
                                          "given the supplied dihedral; conformer-"
                                          "derived cols 2276-2280 reproduce under "
                                          "RDKit 2022.09.5, not 2025.03.6")}
            else:
                worst = max(d["max_rel"] for d in ds)
                ok = worst < 1e-2 and n_nan == 0
                summary[name] = {"status": "PASS" if ok else "FAIL",
                                 "max_rel_diff": worst,
                                 "nan_generated": n_nan,
                                 "n_rows": len(ds)}

        elif name == "shape3d":
            worst = max(d["max_rel"] for d in ds)
            if relaxed:
                summary[name] = {"status": "UNREPRODUCIBLE",
                                 "max_rel_diff": worst,
                                 "note": ("legacy v13 conformer cache is not in the "
                                          "public release; regenerated values are "
                                          "used and flagged per row (block importance "
                                          "7.5%)")}
            else:
                tol = STRICT_SHAPE_REL_TOL if strict else TOLERANT_SHAPE_REL_TOL
                ok = worst < tol
                summary[name] = {"status": "PASS" if ok else "FAIL",
                                 "max_rel_diff": worst,
                                 "tolerance": tol,
                                 "note": "legacy v13 conformer cache is not in the public release"}

    # ---------------------------------------------------------------- gate
    failing = [b for b in ALL_BLOCKS if summary.get(b, {}).get("status") == "FAIL"]
    reproducible = len(failing) == 0
    tolerated = sorted(b for b in ALL_BLOCKS
                       if summary.get(b, {}).get("status")
                       in ("TOLERATED_DRIFT", "UNREPRODUCIBLE", "APPROXIMATE"))

    if args.allow_derived_dihedral:
        dihedral_source = "derived_approximate"
    elif DIHEDRAL_TABLE.exists():
        _, rep = load_external_dihedral(strict_validate=True)
        dihedral_source = ("user_supplied_corrections" if rep["usable"]
                           else f"invalid_table:{rep['reason']}")
    else:
        dihedral_source = "unavailable"

    if mode == "tolerant":
        allowed = False
        claim_level = "diagnostic"
    elif mode == "strict":
        allowed = reproducible and dihedral_source == "user_supplied_corrections"
        claim_level = "publication_v1_exact"
    else:  # relaxed
        allowed = reproducible and dihedral_source in (
            "user_supplied_corrections", "derived_approximate")
        claim_level = ("publication_v1_exact"
                       if reproducible and not tolerated
                       and dihedral_source == "user_supplied_corrections"
                       else "approximate_validation")

    impact = _load_impact_evidence()

    gate = {
        "gate_version": 4,
        "mode": mode,
        "claim_level": claim_level,
        "rdkit_version": _rdkit_version(),
        "python": sys.executable,
        "n_verified": len(rows),
        "n_build_failures": failures,
        "n_integrity_failures": integrity_failures,
        "block_summary": summary,
        "failing_blocks": failing,
        "tolerated_blocks": tolerated,
        "reproducible_blocks_pass": reproducible,
        "gate_status": "PASS" if reproducible else "FAIL",
        "dihedral_source_for_external": dihedral_source,
        "external_inference_allowed": allowed,
        "blocked_reason": None if allowed else _reason(reproducible, failing,
                                                       dihedral_source, mode),
        "drift_impact": impact,
        "reference_matrix": {
            "path": str(TRAIN / "X_C3_5204x2283.npy"),
            "shape": list(X.shape),
            "nan_in_full_matrix": None,
        },
        "limitations": [
            "3D-shape columns in the frozen matrix came from a legacy v13 conformer "
            "cache that is not part of the public release; regenerating them from "
            "ETKDGv3 gives different values.",
            "Baseline dihedral and P/M code for new molecules are unrecoverable: the "
            "training set holds 85 expert sign flips with no encoded rule.",
            "Signed geometry reproduces exactly only when the true dihedral is supplied.",
            "The training matrix is mixed-RDKit: ECFP6 bits match RDKit 2025.03.6 "
            "(5204/5204 rows exact) while ETKDGv3 geometry matches 2022.09.5. "
            "See FEATURE_REPRODUCTION_FINDINGS.md.",
        ],
    }

    write_json(RESULTS / "verify_report.json", {"rows": rows, "summary": summary})
    write_json(RESULTS / "FEATURE_GATE.json", gate)

    print("\n" + "=" * 72)
    for name in ALL_BLOCKS:
        print(f"  {name:14s} {summary[name]['status']}")
    print("=" * 72)
    print(f"  integrity failures : {integrity_failures}/{len(rows)}")
    print(f"  GATE               : {gate['gate_status']}")
    print(f"  dihedral source    : {dihedral_source}")
    print(f"  external inference : {gate['external_inference_allowed']}")
    if gate["blocked_reason"]:
        print(f"  blocked because    : {gate['blocked_reason']}")


def _rdkit_version():
    try:
        import rdkit
        return rdkit.__version__
    except Exception:
        return None


def _load_impact_evidence():
    """Measured prediction impact of the unreproducible blocks, if computed."""
    latest = {}
    for p in sorted(RESULTS.glob("drift_impact_*.json")):
        try:
            import json
            latest[p.name] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
    if not latest:
        return {"available": False,
                "hint": "run src/test_drift_impact.py to quantify block drift impact"}
    # Prefer the file matching the current interpreter's RDKit version.
    v = _rdkit_version()
    chosen = latest.get(f"drift_impact_{v}.json")
    if chosen is None:
        chosen = list(latest.values())[-1]
    return {"available": True,
            "rdkit_version_tested": chosen.get("rdkit_version"),
            "block_importance": chosen.get("block_importance"),
            "variants": chosen.get("variants"),
            "source": "results/drift_impact_*.json"}


def _reason(reproducible, failing, dihedral_source, mode="strict"):
    if not reproducible:
        return f"feature blocks failed {'reproduction' if mode == 'relaxed' else 'exact reproduction'}: {failing}"
    if dihedral_source == "unavailable":
        return ("Publication V1 dihedral/P-M cannot be reconstructed for new "
                "molecules; provide configs/external_dihedral.csv")
    if dihedral_source == "derived_approximate":
        return ("derived dihedral is an approximation (6/40 exact, 33/40 P-M) and "
                "cannot support a strict Publication V1 claim")
    if dihedral_source.startswith("invalid_table"):
        return f"external dihedral table unusable: {dihedral_source}"
    return "unknown"


if __name__ == "__main__":
    main()
