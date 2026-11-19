"""
03 — Frozen CatBoost inference on external features.

Loads the five Publication V1 (GitHub ChiralOR v1.1.0) CatBoost fold models,
verifies their SHA-256, checks the input feature dimension, and produces
per-model and ensemble probabilities.

ENSEMBLE RULE — NOT EVIDENCED BY ORIGINAL DEPLOYMENT CODE
---------------------------------------------------------
The release ships five frozen fold models but no inference/deployment script.
`configs/model_manifest.json` therefore declares
`unweighted_mean_of_fold_probabilities` as an ASSUMPTION consistent with the
released grouped-OOF construction. It is not proven to be the authors' official
deployment rule. Per-fold probabilities are always written out, so any ranking
can be re-derived under a different rule without re-running the models.

This script never reads external labels into the model path, never retrains,
never fine-tunes and never calibrates.

Usage
-----
    python src/03_predict_catboost.py
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    CONFIGS, MODELS, N_FEATURES, RESULTS, read_json, write_json,
)


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    from catboost import CatBoostClassifier

    manifest = read_json(CONFIGS / "model_manifest.json")
    if manifest is None:
        raise SystemExit("configs/model_manifest.json missing")

    if manifest.get("n_features") != N_FEATURES:
        raise SystemExit(
            f"manifest expects {manifest.get('n_features')} features, "
            f"package constant is {N_FEATURES}")

    if manifest.get("ensemble_rule") != "unweighted_mean_of_fold_probabilities":
        raise SystemExit(
            "unexpected ensemble_rule in manifest; only the declared, documented "
            "rule is supported by this script")

    gate = read_json(RESULTS / "FEATURE_GATE.json")
    if gate is None or not gate.get("external_inference_allowed", False):
        raise SystemExit(
            "Feature gate does not authorise external inference.\n"
            f"  status: {None if gate is None else gate.get('gate_status')}\n"
            f"  reason: {None if gate is None else gate.get('blocked_reason')}\n"
            "See results/FEATURE_GATE.json and README section 5.")

    feature_summary = read_json(RESULTS / "external_feature_summary.json", {})
    claim_level = feature_summary.get("claim_level", "unknown")

    X_path = RESULTS / "X_external_2283.npy"
    if not X_path.exists():
        raise SystemExit(f"{X_path} not found — run src/02_generate_features.py first")
    X = np.load(X_path)

    if X.shape[1] != N_FEATURES:
        raise SystemExit(f"feature matrix has {X.shape[1]} columns, expected {N_FEATURES}")

    audit = pd.read_csv(RESULTS / "external_feature_audit.csv")
    if len(audit) != X.shape[0]:
        raise SystemExit(f"audit rows ({len(audit)}) != matrix rows ({X.shape[0]})")

    complete_mask = (audit["complete_vector"].astype(bool).values
                     & np.all(np.isfinite(X), axis=1))
    print(f"records: {X.shape[0]}   complete: {int(complete_mask.sum())}   "
          f"claim_level: {claim_level}")

    probs = np.full((X.shape[0], len(manifest["models"])), np.nan)
    model_report = []

    for k, m in enumerate(manifest["models"]):
        path = MODELS / m["file"]
        if not path.exists():
            raise SystemExit(f"missing model file: {path}")

        digest = sha256_of(path)
        if digest != m["sha256"]:
            raise SystemExit(
                f"SHA-256 mismatch for {m['file']}\n  found:    {digest}\n"
                f"  expected: {m['sha256']}")
        model_report.append({"file": m["file"], "sha256": digest, "match": True})

        clf = CatBoostClassifier()
        clf.load_model(str(path))
        n_model_feats = clf.feature_count_ if hasattr(clf, "feature_count_") else N_FEATURES
        try:
            n_model_feats = clf.get_feature_importance().shape[0]
        except Exception:
            pass
        if n_model_feats not in (0, N_FEATURES):
            raise SystemExit(
                f"{m['file']} expects {n_model_feats} features, package has {N_FEATURES}")

        if complete_mask.any():
            p = clf.predict_proba(X[complete_mask])[:, 1]
            if not np.all(np.isfinite(p)):
                raise SystemExit(f"non-finite probability from {m['file']}")
            probs[complete_mask, k] = p
        print(f"  fold{k}: sha256 OK, predicted {int(complete_mask.sum())} rows")

    with np.errstate(invalid="ignore"):
        ensemble = np.nanmean(probs, axis=1)

    out = pd.DataFrame({
        "position": np.arange(X.shape[0]),
        "record_index": audit["record_index"],
        "smiles": audit["smiles"],
        "solvent": audit["solvent"],
        "OR_value": audit["OR_value"],
        "OR_class": audit["OR_class"],
        "complete_vector": complete_mask,
        "missing_blocks": audit["missing_blocks"],
        "integrity_problems": audit["integrity_problems"],
        "dihedral_source": audit.get("dihedral_source"),
        "pm_label": audit.get("pm_label"),
    })
    for k in range(len(manifest["models"])):
        out[f"prob_fold{k}"] = probs[:, k]
    out["prob_ensemble"] = ensemble
    out["pred_class"] = np.where(np.isnan(ensemble), np.nan,
                                 (ensemble >= manifest.get("threshold", 0.5)).astype(float))
    out.to_csv(RESULTS / "external_predictions.csv", index=False)

    summary = {
        "n_records": int(X.shape[0]),
        "n_complete": int(complete_mask.sum()),
        "n_with_ensemble_prob": int(np.isfinite(ensemble).sum()),
        "ensemble_rule": manifest.get("ensemble_rule"),
        "ensemble_rule_evidence": manifest.get("ensemble_rule_evidence", "NOT_FOUND"),
        "threshold": manifest.get("threshold", 0.5),
        "claim_level": claim_level,
        "models": model_report,
        "note": ("predict_proba only; no retraining, no fine-tuning, no calibration. "
                 "Per-fold probabilities retained so the ensemble rule can be replaced "
                 "without re-running the models."),
    }
    write_json(RESULTS / "external_prediction_summary.json", summary)

    print("\n" + "=" * 74)
    print(f"  predictions: {RESULTS / 'external_predictions.csv'}")
    print(f"  with ensemble probability: {summary['n_with_ensemble_prob']}")
    print(f"  ensemble rule evidence   : {summary['ensemble_rule_evidence']}")
    print("=" * 74)


if __name__ == "__main__":
    main()
