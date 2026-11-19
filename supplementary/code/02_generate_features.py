"""
02 — Generate 2283-dimensional features for curated external records.

Reads  results/external_curated.csv  (written by 01_clean_reaxys.py)
Writes results/X_external_2283.npy, results/external_feature_audit.csv

Dihedral / P-M source (Publication V1 fidelity)
-----------------------------------------------
configs/external_dihedral.csv must supply, per record:

    smiles,solvent,dihedral,pm_label

Rows are matched on (canonical_smiles, solvent). A record with no matching row
is INCOMPLETE — it is written with an explicit missing-block reason and is
excluded from evaluation. The mere presence of the file is never sufficient:
the table is parsed, validated (finite dihedral, P or M, one of the five
solvents, unique keys) and coverage of the curated set is reported. If any
curated record lacks a valid entry, strict mode refuses to write a matrix.

Modes
-----
--dihedral-mode table     (default) require complete coverage from the table
--dihedral-mode derived   derive per molecule via pm_calculator_correct and
                          stamp every row APPROXIMATE (never a strict claim)
--dihedral-mode none      leave dihedral/PM/geometry empty; rows incomplete

Resume safety
-------------
--resume only reuses rows that were marked complete in the previous run AND
whose stored feature vector is finite. The fingerprint of the inputs
(curated CSV, training labels, dihedral table, model manifest) must also match,
so features from a different cleaning version are never mixed in.

Usage
-----
    python src/02_generate_features.py --dihedral-mode table
    python src/02_generate_features.py --dihedral-mode derived --resume
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DIHEDRAL_TABLE, N_FEATURES, RESULTS, data_fingerprint, load_external_dihedral,
    read_json, write_json,
)
from feature_builder import build_vector, check_vector_integrity, fatal_blocks  # noqa: E402

CKPT_EVERY = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dihedral-mode", choices=["table", "derived", "none"],
                    default="table")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    gate = read_json(RESULTS / "FEATURE_GATE.json")
    if gate is None:
        raise SystemExit("FEATURE_GATE.json not found. Run src/00_verify_feature_generator.py.")
    if not gate.get("external_inference_allowed", False):
        raise SystemExit(
            "Feature gate does not authorise external inference.\n"
            f"  mode: {gate.get('mode')}  claim: {gate.get('claim_level')}\n"
            f"  failing blocks: {gate.get('failing_blocks')}\n"
            f"  reason: {gate.get('blocked_reason')}")
    claim_level = gate.get("claim_level", "unknown")
    if claim_level != "publication_v1_exact":
        print(f"NOTE: gate claim level is '{claim_level}'; results must be "
              f"reported with the approximations recorded in FEATURE_GATE.json "
              f"(tolerated blocks: {gate.get('tolerated_blocks')}).")

    curated = pd.read_csv(RESULTS / "external_curated.csv")
    if args.limit:
        curated = curated.head(args.limit)
    n = len(curated)

    # ---------------- resolve dihedral source ----------------
    dih_table, dih_report = {}, {"exists": False, "usable": False, "reason": "not_loaded"}

    if args.dihedral_mode == "table":
        dih_table, dih_report = load_external_dihedral(strict_validate=True)
        if not dih_report["usable"]:
            raise SystemExit(
                "dihedral-mode=table requires a valid configs/external_dihedral.csv\n"
                f"  exists : {dih_report['exists']}\n"
                f"  reason : {dih_report['reason']}\n"
                f"  rows   : {dih_report['n_rows']} (valid {dih_report['n_valid']}, "
                f"rejected {dih_report['n_rejected']})")

        missing_keys = []
        for _, r in curated.iterrows():
            key = (r["canonical_smiles"], r["solvent"])
            if key not in dih_table:
                missing_keys.append({"smiles": r["canonical_smiles"],
                                     "solvent": r["solvent"]})
        if missing_keys:
            write_json(RESULTS / "external_dihedral_missing.json",
                       {"n_missing": len(missing_keys),
                        "n_curated": n,
                        "examples": missing_keys[:50]})
            raise SystemExit(
                f"strict mode: {len(missing_keys)} of {n} curated records have no "
                "entry in configs/external_dihedral.csv.\n"
                "See results/external_dihedral_missing.json. Nothing was written.")

    # ---------------- resume fingerprint ----------------
    fp = data_fingerprint()
    X_path = RESULTS / "X_external_2283.npy"
    audit_path = RESULTS / "external_feature_audit.csv"
    prev = None

    if args.resume and X_path.exists() and audit_path.exists():
        prev_meta = read_json(RESULTS / "external_feature_meta.json", {})
        if prev_meta.get("fingerprint") != fp:
            print("resume refused: inputs changed since the previous run")
            print(f"  prev: {prev_meta.get('fingerprint')}")
            print(f"  now : {fp}")
            prev = None
        else:
            X_prev = np.load(X_path)
            prev = pd.read_csv(audit_path)
            if X_prev.shape != (n, N_FEATURES) or len(prev) != n:
                print("resume refused: shape/row mismatch with the previous run")
                prev = None
            else:
                n_reuse = int(prev["done"].fillna(0).sum()) if "done" in prev else 0
                print(f"resuming: {n_reuse} previously completed rows eligible for reuse")

    if prev is None:
        X = np.zeros((n, N_FEATURES), dtype=np.float32)
    else:
        X = np.load(X_path).copy()

    need_full = args.dihedral_mode != "none"
    need_blocks = fatal_blocks(require_full_vector=need_full)

    rows, t0 = [], time.time()
    n_ok = n_incomplete = n_reused = 0

    for pos, (_, r) in enumerate(curated.iterrows()):
        smi = r["canonical_smiles"]
        solv = r["solvent"]

        # ---- resume: reuse ONLY genuinely complete, finite rows ----
        if prev is not None:
            old = prev.iloc[pos]
            stored_finite = bool(np.all(np.isfinite(X[pos])))
            if (int(old.get("done", 0)) == 1
                    and bool(old.get("complete_vector", False))
                    and stored_finite):
                rows.append(old.to_dict())
                n_ok += 1
                n_reused += 1
                continue

        # ---- resolve inputs ----
        if args.dihedral_mode == "table":
            dih, pm = dih_table[(smi, solv)]
            vec, audit = build_vector(smi, solv, dihedral=dih, pm_label=pm)
        elif args.dihedral_mode == "derived":
            vec, audit = build_vector(smi, solv, allow_derived_dihedral=True)
        else:
            vec, audit = build_vector(smi, solv)

        missing = [b for b in need_blocks if not audit.get(b, False)]
        complete = False
        problems = []

        if vec is not None:
            ok, problems = check_vector_integrity(vec, audit)
            complete = ok and not missing
            X[pos] = vec

        if complete:
            n_ok += 1
        else:
            n_incomplete += 1

        rows.append({
            "position": pos,
            "record_index": r["record_index"],
            "smiles": smi,
            "solvent": solv,
            "OR_value": r["OR_value"],
            "OR_class": r["OR_class"],
            "complete_vector": bool(complete),
            "missing_blocks": ";".join(missing),
            "integrity_problems": ";".join(problems[:8]),
            "ecfp6": audit.get("ecfp6"), "maccs": audit.get("maccs"),
            "rdkit2d": audit.get("rdkit2d"), "spas": audit.get("spas"),
            "pas": audit.get("pas"), "shape3d": audit.get("shape3d"),
            "solvent_code": audit.get("solvent_code"),
            "dihedral": audit.get("dihedral"), "pm_code": audit.get("pm_code"),
            "geometry": audit.get("geometry"),
            "dihedral_source": audit.get("dihedral_source"),
            "pm_label": audit.get("pm_label"),
            "structural_zero_cols": ";".join(map(str, audit.get("structural_zero_cols", []))),
            "nan_inf_trapped": ";".join(map(str, audit.get("nan_inf_trapped", []))),
            "needs_blind_review": audit.get("needs_blind_review"),
            "errors": ";".join(audit.get("errors", [])),
            "done": 1,
        })

        if (pos + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {pos+1}/{n}  ok={n_ok} (reused {n_reused})  incomplete={n_incomplete}"
                  f"  {el/60:.1f} min")

        if (pos + 1) % CKPT_EVERY == 0:
            np.save(X_path, X)
            pd.DataFrame(rows).to_csv(audit_path, index=False)

    np.save(X_path, X)
    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(audit_path, index=False)

    summary = {
        "n_records": int(n),
        "n_complete_vector": int(n_ok),
        "n_reused_from_checkpoint": int(n_reused),
        "n_incomplete_vector": int(n_incomplete),
        "dihedral_mode": args.dihedral_mode,
        "dihedral_table": {k: dih_report.get(k) for k in
                          ("exists", "n_rows", "n_valid", "n_rejected", "usable", "reason")},
        "gate_status": gate.get("gate_status"),
        "gate_mode": gate.get("mode"),
        "elapsed_minutes": round((time.time() - t0) / 60, 2),
        # Claim level comes from the gate; a user-supplied dihedral table with an
        # exact gate is the only path to publication_v1_exact.
        "claim_level": ("publication_v1_exact"
                        if gate.get("claim_level") == "publication_v1_exact"
                        and args.dihedral_mode == "table"
                        else gate.get("claim_level", "unknown")),
        "tolerated_blocks": gate.get("tolerated_blocks", []),
        "drift_impact": gate.get("drift_impact"),
    }
    write_json(RESULTS / "external_feature_summary.json", summary)
    write_json(RESULTS / "external_feature_meta.json", {"fingerprint": fp})

    print("\n" + "=" * 74)
    for k, v in summary.items():
        print(f"  {k:30s} {v}")
    print("=" * 74)


if __name__ == "__main__":
    main()
