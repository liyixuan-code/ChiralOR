"""
10 — Single-conformer robustness (reviewer point #4).

The chirality-aware geometry block and the 3D-shape block depend on ONE
ETKDGv3+MMFF94s conformer per molecule (seed 42). This script quantifies how
sensitive predictions are to that choice.

For seeds {42, 123, 999}:
  * re-embed every unique training molecule and recompute
    - shape3d  (cols 2263-2272, 10 values)
    - geometry (cols 2276-2282, 7 values; cols 2281/2282 are sin/cos of the
      stored dihedral and therefore seed-invariant)
  * splice the recomputed blocks into the frozen matrix (all other 2266 columns
    unchanged) -> one matrix per seed
  * score OOF with the fold-own FROZEN models (prediction variance at fixed
    model) AND retrain 5-fold CatBoost identically (model + feature variance)

Reports
  * per-feature cross-seed SD (relative to the feature's own SD)
  * OOF AUC / BA / ACC per seed, frozen-model and retrained
  * |dp| and class-flip rate vs the seed-42 matrix
  * zero-rate of col 2280 (structurally undefined NS-path) and the effect of
    adding an explicit missingness indicator for it

Writes results/conformer_robustness.json
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RESULTS, TRAIN, add_original_modules, write_json  # noqa: E402

add_original_modules()
from feature_generation import block2_signed_geometry  # noqa: E402
from rdkit.Chem import Descriptors3D  # noqa: E402

SHAPE3D_NAMES = ["PMI1", "PMI2", "PMI3", "NPR1", "NPR2",
                 "RadiusOfGyration", "InertialShapeFactor",
                 "Eccentricity", "Asphericity", "SpherocityIndex"]
GEO_NAMES = ["signed_tetra_volume", "subst_to_ringplane_signed_dist",
             "ringnormal_dot_substvec", "signed_dihedral_subst_c_rn1_rn2",
             "signed_dihedral_NS_path", "baseline_pm_dihedral_sin",
             "baseline_pm_dihedral_cos"]
SEEDS = [42, 123, 999]


def embed_blocks(smiles, seed, dihedral, chiral_idx):
    mol_h, audit = block2_signed_geometry.embed_conformer(smiles, seed=seed)
    if mol_h is None or audit.get("embed_success", 0) != 1:
        return None, None, audit
    shape = [float(getattr(Descriptors3D, n)(mol_h)) for n in SHAPE3D_NAMES]
    geo, ok = block2_signed_geometry.compute(smiles, chiral_idx, dihedral, mol_h, audit)
    if not ok:
        return shape, None, audit
    geo = np.nan_to_num(np.asarray(geo, dtype=np.float64), nan=0.0,
                        posinf=0.0, neginf=0.0)
    return shape, geo, audit


def main():
    from catboost import CatBoostClassifier
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 roc_auc_score)

    ml = pd.read_csv(TRAIN / "master_labels_v2.csv")
    X0 = np.asarray(np.load(TRAIN / "X_C3_5204x2283.npy"))
    folds = pd.read_csv(TRAIN / "primary_grouped_fold_assignments_public.csv")
    fold_of = folds.fold.values
    y = ml.OR_label.values

    models = []
    for k in range(5):
        m = CatBoostClassifier()
        m.load_model(str(Path(__file__).resolve().parents[1] / "models"
                          / f"catboost_schemaA_fold{k}.cbm"))
        models.append(m)

    CB = dict(iterations=288, depth=8, learning_rate=0.1, l2_leaf_reg=1,
              random_seed=42, loss_function="Logloss", bootstrap_type="MVS",
              subsample=0.8, verbose=False, allow_writing_files=False)

    def oof_frozen(Xm):
        p = np.full(len(Xm), np.nan)
        for k in range(5):
            mask = fold_of == k
            p[mask] = models[k].predict_proba(Xm[mask])[:, 1]
        return p

    def oof_retrained(Xm):
        p = np.full(len(Xm), np.nan)
        for k in range(5):
            tr, te = fold_of != k, fold_of == k
            m = CatBoostClassifier(**CB)
            m.fit(Xm[tr], y[tr])
            p[te] = m.predict_proba(Xm[te])[:, 1]
        return p

    def report(p, tag):
        yhat = (p >= 0.5).astype(int)
        return {"tag": tag,
                "oof_auc": float(roc_auc_score(y, p)),
                "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
                "accuracy": float(accuracy_score(y, yhat))}

    # reference: frozen matrix itself
    p_ref = oof_frozen(X0)
    res = {"reference_frozen_matrix": report(p_ref, "frozen_X")}
    print(f"reference (frozen matrix, frozen models): AUC={res['reference_frozen_matrix']['oof_auc']:.4f}")

    # unique molecules and their reference dihedral/chiral index
    uniq = ml.drop_duplicates("canonical_smiles")[["canonical_smiles"]].copy()
    dihedral_by = {}
    center_by = {}
    for i, s in enumerate(ml.canonical_smiles):
        if s in dihedral_by:
            continue
        dihedral_by[s] = float(X0[i, 2274])
        m = Chem.MolFromSmiles(s)
        cs = Chem.FindMolChiralCenters(m, includeUnassigned=False) if m else []
        center_by[s] = cs[0][0] if cs else None
    print(f"unique molecules to embed: {len(uniq)}")

    per_seed = {}
    shape_store, geo_store = {}, {}
    for seed in SEEDS:
        t0 = time.time()
        shape_rows = np.full((len(ml), 10), np.nan)
        geo_rows = np.full((len(ml), 7), np.nan)
        fails = 0
        cache = {}
        for i, s in enumerate(ml.canonical_smiles):
            if s not in cache:
                if center_by.get(s) is None:
                    cache[s] = (None, None)
                else:
                    sh, ge, _ = embed_blocks(s, seed, dihedral_by[s], center_by[s])
                    cache[s] = (sh, ge)
            sh, ge = cache[s]
            if sh is not None:
                shape_rows[i] = sh
            if ge is not None:
                geo_rows[i] = ge
            else:
                fails += 1
        bad_shape = np.where(~np.isfinite(shape_rows).all(axis=1))[0]
        bad_geo = np.where(~np.isfinite(geo_rows).all(axis=1))[0]
        print(f"seed {seed}: {time.time()-t0:.1f} min  embed/geo failures={fails} "
              f"rows_with_bad_shape={len(bad_shape)} rows_with_bad_geo={len(bad_geo)}")

        Xs = X0.copy()
        ok_s = np.isfinite(shape_rows).all(axis=1)
        ok_g = np.isfinite(geo_rows).all(axis=1)
        Xs[ok_s, 2263:2273] = shape_rows[ok_s]
        Xs[ok_g, 2276:2283] = geo_rows[ok_g]
        shape_store[seed] = shape_rows
        geo_store[seed] = geo_rows

        p_frozen = oof_frozen(Xs)
        p_ret = oof_retrained(Xs)
        d = np.abs(p_frozen - p_ref)
        flips = int(np.sum((p_frozen >= 0.5) != (p_ref >= 0.5)))
        per_seed[seed] = {
            "frozen_model": report(p_frozen, f"seed{seed}_frozen_model"),
            "retrained": report(p_ret, f"seed{seed}_retrained"),
            "mean_abs_dp_vs_seed42": float(d.mean()),
            "p95_abs_dp_vs_seed42": float(np.percentile(d, 95)),
            "class_flips_vs_seed42": flips,
            "class_flip_rate": float(flips / len(d)),
            "embed_failures": int(fails),
            "elapsed_min": round((time.time() - t0) / 60, 1),
        }
        print(f"   frozen AUC={per_seed[seed]['frozen_model']['oof_auc']:.4f} "
              f"retrained AUC={per_seed[seed]['retrained']['oof_auc']:.4f} "
              f"mean|dp|={per_seed[seed]['mean_abs_dp_vs_seed42']:.4f} "
              f"flips={flips}")

    # cross-seed feature SD (conformer-dependent columns only)
    def cross_seed_stats(store, ncols, names, offset):
        arr = np.stack([store[s] for s in SEEDS])          # (seeds, rows, cols)
        finite = np.isfinite(arr).all(axis=0)
        sd = np.nanstd(arr, axis=0, ddof=1)
        ref_sd = np.nanstd(arr[0], axis=0, ddof=1)
        out = {}
        for c in range(ncols):
            vals = sd[finite[:, c], c] if finite[:, c].any() else np.array([np.nan])
            out[names[c]] = {
                "mean_cross_seed_sd": float(np.nanmean(vals)),
                "median_cross_seed_sd": float(np.nanmedian(vals)),
                "p95_cross_seed_sd": float(np.nanpercentile(vals, 95)),
                "feature_sd_across_molecules": float(ref_sd[c]),
                "relative_sd": float(np.nanmean(vals) / (ref_sd[c] + 1e-12)),
            }
        return out

    shape_stats = cross_seed_stats(shape_store, 10, SHAPE3D_NAMES, 2263)
    geo_stats = cross_seed_stats(geo_store, 7, GEO_NAMES, 2276)

    # missingness: col 2280 (NS path) structural zeros
    col2280_zero = float(np.mean(X0[:, 2280] == 0))
    X_ind = np.column_stack([X0, (X0[:, 2280] == 0).astype(np.float32)])
    p_ind = oof_retrained(X_ind)
    miss = {"zero_rate_col2280": col2280_zero,
            "retrained_with_indicator": report(p_ind, "with_missing_indicator"),
            "delta_auc_vs_reference_retrained": None}

    aucs_f = [per_seed[s]["frozen_model"]["oof_auc"] for s in SEEDS]
    aucs_r = [per_seed[s]["retrained"]["oof_auc"] for s in SEEDS]
    out = {
        "protocol": {
            "seeds": SEEDS,
            "conformer": "ETKDGv3 + MMFF94s, seed incremented per attempt (<=10)",
            "spliced_columns": "shape3d 2263-2272; geometry 2276-2280 "
                               "(2281/2282 are sin/cos of the stored dihedral, seed-invariant)",
            "evaluation": "published grouped folds; frozen fold-own models AND "
                          "identically retrained CatBoost (288/8/0.1)",
        },
        "reference_frozen_matrix": res["reference_frozen_matrix"],
        "per_seed": per_seed,
        "auc_spread_frozen_model": {"min": min(aucs_f), "max": max(aucs_f),
                                    "sd": float(np.std(aucs_f, ddof=1))},
        "auc_spread_retrained": {"min": min(aucs_r), "max": max(aucs_r),
                                 "sd": float(np.std(aucs_r, ddof=1))},
        "shape3d_cross_seed": shape_stats,
        "geometry_cross_seed": geo_stats,
        "missingness": miss,
    }
    write_json(RESULTS / "conformer_robustness.json", out)
    print("\nsaved: results/conformer_robustness.json")
    print(f"AUC spread (frozen model): {min(aucs_f):.4f}-{max(aucs_f):.4f}  "
          f"SD={np.std(aucs_f, ddof=1):.4f}")
    print(f"AUC spread (retrained)   : {min(aucs_r):.4f}-{max(aucs_r):.4f}  "
          f"SD={np.std(aucs_r, ddof=1):.4f}")


if __name__ == "__main__":
    main()
