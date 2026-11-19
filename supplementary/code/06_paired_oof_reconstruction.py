"""
06 — Paired OOF experiment: frozen matrix vs reconstructed features.

Protocol (per user requirement, no in-sample evaluation):
  * Every training sample is predicted ONLY by the fold model that held it out
    (primary grouped fold assignments from the public release).
  * Matrix A = frozen X_C3_5204x2283.npy (should reproduce published OOF AUC 0.9278).
  * Matrix B = same molecules rebuilt by the current generator, all 2283 dims,
    derived dihedral (= the realistic path used for external molecules).
  * Metrics: OOF AUC(A), OOF AUC(B), probability MAE |p_B - p_A|, class-flip
    rate at 0.5, plus per-block swap-in tests (replace one block of A with B)
    to attribute OOF degradation to specific blocks.

Writes results/paired_oof_reconstruction.json and
results/paired_oof_predictions.csv.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BLOCKS, RESULTS, TRAIN, write_json  # noqa: E402

N_FEATURES = 2283


def main():
    from catboost import CatBoostClassifier
    from feature_builder import build_vector
    from sklearn.metrics import roc_auc_score

    ml = pd.read_csv(TRAIN / "master_labels_v2.csv")
    X_A = np.asarray(np.load(TRAIN / "X_C3_5204x2283.npy"))
    folds = pd.read_csv(TRAIN / "primary_grouped_fold_assignments_public.csv")
    oof_A = pd.read_csv(TRAIN / "primary_grouped_oof_predictions.csv")

    assert len(ml) == len(X_A) == len(folds) == len(oof_A) == 5204
    assert (folds.observation_id.values == np.arange(5204)).all()
    assert (oof_A.observation_id.values == np.arange(5204)).all()
    fold_of = folds.fold.values
    y = ml["OR_label"].to_numpy(dtype=int)

    auc_A = roc_auc_score(y, oof_A.oof_probability_or_plus.values)
    print(f"reference: published OOF AUC(A) = {auc_A:.4f}  (paper 0.9278)")

    models = []
    for k in range(5):
        m = CatBoostClassifier()
        m.load_model(str(Path(__file__).resolve().parents[1] / "models"
                          / f"catboost_schemaA_fold{k}.cbm"))
        models.append(m)

    # ---- rebuild matrix B (all 2283 dims, derived dihedral path) ----
    cache = RESULTS / "X_reconstructed_2283.npy"
    audit_path = RESULTS / "reconstruction_audit.csv"
    if cache.exists() and audit_path.exists():
        X_B = np.load(cache)
        aud = pd.read_csv(audit_path)
        print(f"reusing cached reconstruction: {X_B.shape}")
    else:
        X_B = np.zeros((5204, N_FEATURES), dtype=np.float32)
        aud_rows = []
        t0 = time.time()
        for i in range(5204):
            r = ml.iloc[i]
            vec, a = build_vector(r["canonical_smiles"], r["solvent_group"],
                                  allow_derived_dihedral=True)
            complete = vec is not None
            if complete:
                X_B[i] = vec
            aud_rows.append({
                "idx": i, "complete": bool(complete),
                "pm_label": a.get("pm_label"),
                "dihedral_source": a.get("dihedral_source"),
                "errors": ";".join(a.get("errors", []))[:300],
            })
            if (i + 1) % 250 == 0:
                np.save(cache, X_B)
                pd.DataFrame(aud_rows).to_csv(audit_path, index=False)
                print(f"  {i+1}/5204  {(time.time()-t0)/60:.1f} min")
        np.save(cache, X_B)
        aud = pd.DataFrame(aud_rows)
        aud.to_csv(audit_path, index=False)
        print(f"reconstruction done: {(time.time()-t0)/60:.1f} min")

    X_B = X_B.astype(np.float32)

    # ---- OOF predictions: each sample scored only by its held-out fold model ----
    def oof_predict(Xmat):
        p = np.full(len(Xmat), np.nan)
        for k in range(5):
            mask = fold_of == k
            p[mask] = models[k].predict_proba(Xmat[mask])[:, 1]
        return p

    p_A = oof_A.oof_probability_or_plus.values
    p_A_recomputed = oof_predict(X_A)          # sanity: must match published OOF
    p_B = oof_predict(X_B)

    print(f"sanity: recomputed OOF(A) AUC = {roc_auc_score(y, p_A_recomputed):.4f} "
          f"(published {auc_A:.4f})")

    def metrics(pa, pb, tag):
        valid = np.isfinite(pa) & np.isfinite(pb)
        d = np.abs(pb - pa)[valid]
        flips = int(np.sum(((pa >= 0.5) != (pb >= 0.5))[valid]))
        out = {
            "tag": tag,
            "n_valid": int(valid.sum()),
            "oof_auc": float(roc_auc_score(y[valid], pb[valid])),
            "prob_mae": float(d.mean()),
            "prob_p95_abs": float(np.percentile(d, 95)),
            "class_flips": flips,
            "class_flip_rate": float(flips / max(valid.sum(), 1)),
        }
        print(f"  {tag:28s} OOF-AUC={out['oof_auc']:.4f}  MAE={out['prob_mae']:.4f}  "
              f"flips={flips}/{int(valid.sum())} ({out['class_flip_rate']:.1%})")
        return out

    print("\nPaired OOF comparison (fold-own-model, no in-sample):")
    res_B = metrics(p_A, p_B, "reconstructed_all_blocks")

    # ---- per-block swap-in attribution (OOF protocol) ----
    swap = {}
    for b, (lo, hi) in BLOCKS.items():
        M = X_A.copy()
        M[:, lo:hi] = X_B[:, lo:hi]
        pM = oof_predict(M)
        res = metrics(p_A, pM, f"swap_in_{b}")
        res["delta_oof_auc"] = res["oof_auc"] - auc_A
        swap[b] = res

    out = {
        "protocol": {
            "folds": "primary_grouped_fold_assignments_public.csv",
            "oof_rule": "each sample predicted only by its held-out fold model",
            "reference_published_oof_auc": auc_A,
            "reconstructed_path": "build_vector(allow_derived_dihedral=True), all 2283 dims",
        },
        "sanity_recomputed_oof_A_auc": float(roc_auc_score(y, p_A_recomputed)),
        "reconstructed": res_B,
        "block_swap_in": swap,
        "n_reconstruction_incomplete": int((~aud["complete"]).sum()),
        "dihedral_source_counts": aud["dihedral_source"].fillna("fail").value_counts().to_dict(),
        "note": ("OOF protocol only. No training-set in-sample prediction is used "
                 "anywhere in this file."),
    }
    write_json(RESULTS / "paired_oof_reconstruction.json", out)

    pd.DataFrame({
        "observation_id": np.arange(5204),
        "fold": fold_of,
        "y": y,
        "oof_prob_A_frozen": p_A,
        "oof_prob_B_reconstructed": p_B,
    }).to_csv(RESULTS / "paired_oof_predictions.csv", index=False)
    print("\nsaved: results/paired_oof_reconstruction.json")


if __name__ == "__main__":
    main()
