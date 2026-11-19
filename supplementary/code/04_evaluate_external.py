"""
04 — External evaluation metrics with molecule-cluster bootstrap CI.

Evaluates only rows that (a) have a complete 2283-dim vector and (b) carry an
experimental OR label. Incomplete rows are never dropped silently: they are
written to a separate file with their reason, and their count is reported.

Usage
-----
    python src/04_evaluate_external.py --n-boot 2000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RESULTS, read_json, write_json  # noqa: E402


def ece_10bin(y, p):
    """10-bin equal-frequency expected calibration error (paper definition)."""
    df = pd.DataFrame({"y": y, "p": p}).sort_values("p").reset_index(drop=True)
    df["bin"] = (np.arange(len(df)) * 10) // len(df)
    gaps = df.groupby("bin").apply(
        lambda g: abs(g["y"].mean() - g["p"].mean()), include_groups=False)
    return float(gaps.mean())


def calibration_slope_intercept(y, p, eps=1e-9):
    """Logistic regression of y on logit(p); slope/intercept are diagnostics."""
    from sklearn.linear_model import LogisticRegression
    p = np.clip(p, eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(logit, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def all_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    slope, intercept = calibration_slope_intercept(y, p)
    out = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "average_precision": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "f1": float(f1_score(y, yhat)),
        "mcc": float(matthews_corrcoef(y, yhat)),
        "brier_score": float(brier_score_loss(y, p)),
        "ece_10bin": ece_10bin(y, p),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "precision_ppv": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        "npv": float(tn / (tn + fn)) if (tn + fn) else float("nan"),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    return out


def cluster_bootstrap_ci(y, p, clusters, n_boot=5000, seed=42):
    """Group-cluster bootstrap (units = canonical_smiles x solvent, as in the
    paper) with CIs for the metrics the paper reports."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(clusters)
    index_by_cluster = {c: np.where(clusters == c)[0] for c in uniq}

    aucs, aps, briers, eces, slopes, intercepts, bas = [], [], [], [], [], [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_by_cluster[c] for c in pick])
        yy, pp = y[idx], p[idx]
        if len(set(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, pp))
        aps.append(average_precision_score(yy, pp))
        briers.append(brier_score_loss(yy, pp))
        eces.append(ece_10bin(yy, pp))
        if len(np.unique(yy)) == 2:
            s, b = calibration_slope_intercept(yy, pp)
            slopes.append(s)
            intercepts.append(b)
        bas.append(balanced_accuracy_score(yy, (pp >= 0.5).astype(int)))

    def ci(v):
        if not v:
            return [float("nan"), float("nan")]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    return {
        "roc_auc_ci95": ci(aucs),
        "average_precision_ci95": ci(aps),
        "brier_score_ci95": ci(briers),
        "ece_10bin_ci95": ci(eces),
        "calibration_slope_ci95": ci(slopes),
        "calibration_intercept_ci95": ci(intercepts),
        "balanced_accuracy_ci95": ci(bas),
        "n_bootstrap_effective": len(aucs),
        "n_bootstrap_requested": n_boot,
        "cluster_unit": "canonical_smiles x solvent (paper protocol)",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    pred_path = RESULTS / "external_predictions.csv"
    if not pred_path.exists():
        raise SystemExit(f"{pred_path} not found — run src/03_predict_catboost.py first")
    df = pd.read_csv(pred_path)

    gate = read_json(RESULTS / "FEATURE_GATE.json", {}) or {}
    feature_summary = read_json(RESULTS / "external_feature_summary.json", {}) or {}
    claim_level = feature_summary.get("claim_level", "unknown")

    usable = (df["complete_vector"].astype(bool)
              & df["prob_ensemble"].notna()
              & df["OR_class"].notna())
    evaluated = df[usable].copy()
    excluded = df[~usable].copy()

    ex_cols = ["position", "record_index", "smiles", "solvent",
               "OR_value", "OR_class", "missing_blocks"]
    if "integrity_problems" in excluded.columns:
        ex_cols.append("integrity_problems")
    excluded[ex_cols].to_csv(RESULTS / "external_excluded_records.csv", index=False)

    if len(evaluated) == 0:
        write_json(RESULTS / "external_metrics.json", {
            "error": "no records with complete 2283-dim vectors",
            "n_records": int(len(df)),
            "n_complete": 0,
        })
        print("No complete rows — metrics not computed.")
        return

    y = evaluated["OR_class"].astype(int).values
    p = evaluated["prob_ensemble"].values
    # Paper protocol: resampling unit = canonical_smiles x solvent_group.
    clusters = (evaluated["smiles"].astype(str) + "|"
                + evaluated["solvent"].astype(str)).values

    m = all_metrics(y, p, thr=args.threshold)
    m.update(cluster_bootstrap_ci(y, p, clusters, n_boot=args.n_boot))
    m.update({
        "claim_level": claim_level,
        "gate_status": gate.get("gate_status"),
        "gate_mode": gate.get("mode"),
        "dihedral_source_for_external": gate.get("dihedral_source_for_external"),
        "evaluation_protocol": {
            "model": "published Publication V1 five-fold CatBoost ensemble (frozen, no retraining)",
            "ensemble_rule": "unweighted mean of fold probabilities (manifest assumption)",
            "threshold": args.threshold,
            "bootstrap": f"group-cluster (canonical_smiles x solvent), {args.n_boot} replicates",
            "paper_reference": "JCIM manuscript: pooled OOF AUC 0.9278; ECE 10-bin; "
                               "Brier; calibration slope/intercept; 0.5 threshold",
        },
        "reference_paper_metrics": {
            "oof_auc": 0.9278,
            "balanced_accuracy": 0.850,
            "sensitivity": 0.799,
            "specificity": 0.900,
            "accuracy": 0.860,
            "brier": 0.1027,
            "ece_10bin": 0.0215,
            "calibration_slope": 0.893,
            "calibration_intercept": 0.067,
        },
        "coverage": {
            "n_cleaned_records": int(len(df)),
            "n_complete_vector": int(usable.sum()),
            "n_evaluated": int(len(evaluated)),
            "n_excluded": int(len(excluded)),
            "completion_rate": float(usable.mean()),
            "n_unique_molecules": int(evaluated["smiles"].nunique()),
        },
        "dihedral_source_counts": (evaluated["dihedral_source"]
                                   .fillna("unavailable").value_counts().to_dict()),
        "excluded_reasons": (excluded["missing_blocks"]
                             .fillna("unknown").value_counts().to_dict()),
        "assumptions": [
            "Evaluation includes only rows with complete 2283-dim vectors.",
            "Cluster bootstrap resamples canonical SMILES (all solvent records move together).",
            "If dihedral_source is 'derived_approximate', those rows use an "
            "APPROXIMATION of the Publication V1 dihedral, not the original value "
            "and the result is an approximate estimate, not a strict reproduction.",
            "The five-model ensemble rule is a declared assumption; per-fold "
            "probabilities are retained in external_predictions.csv.",
        ],
    })
    write_json(RESULTS / "external_metrics.json", m)

    print("=" * 74)
    print("  EXTERNAL EVALUATION (paper metric set)")
    print("=" * 74)
    print(f"  evaluated rows        : {m['coverage']['n_evaluated']}")
    print(f"  unique molecules      : {m['coverage']['n_unique_molecules']}")
    print(f"  completion rate       : {m['coverage']['completion_rate']:.3f}")
    print(f"  ROC-AUC               : {m['roc_auc']:.4f} "
          f"[{m['roc_auc_ci95'][0]:.4f}, {m['roc_auc_ci95'][1]:.4f}]")
    print(f"  Average Precision     : {m['average_precision']:.4f} "
          f"[{m['average_precision_ci95'][0]:.4f}, {m['average_precision_ci95'][1]:.4f}]")
    print(f"  Balanced accuracy     : {m['balanced_accuracy']:.4f} "
          f"[{m['balanced_accuracy_ci95'][0]:.4f}, {m['balanced_accuracy_ci95'][1]:.4f}]")
    print(f"  Sensitivity / Spec.   : {m['sensitivity']:.3f} / {m['specificity']:.3f}")
    print(f"  PPV / NPV             : {m['precision_ppv']:.3f} / {m['npv']:.3f}")
    print(f"  Accuracy / MCC        : {m['accuracy']:.3f} / {m['mcc']:.3f}")
    print(f"  Brier                 : {m['brier_score']:.4f} "
          f"[{m['brier_score_ci95'][0]:.4f}, {m['brier_score_ci95'][1]:.4f}]")
    print(f"  ECE (10-bin)          : {m['ece_10bin']:.4f} "
          f"[{m['ece_10bin_ci95'][0]:.4f}, {m['ece_10bin_ci95'][1]:.4f}]")
    print(f"  Calibration slope     : {m['calibration_slope']:.3f} "
          f"[{m['calibration_slope_ci95'][0]:.3f}, {m['calibration_slope_ci95'][1]:.3f}]")
    print(f"  Calibration intercept : {m['calibration_intercept']:.3f} "
          f"[{m['calibration_intercept_ci95'][0]:.3f}, {m['calibration_intercept_ci95'][1]:.3f}]")
    print("=" * 74)
    print(f"  paper reference (grouped OOF, internal): AUC 0.9278 | Brier 0.1027 | ECE 0.0215")
    print(f"  metrics : {RESULTS / 'external_metrics.json'}")
    print(f"  excluded: {RESULTS / 'external_excluded_records.csv'}")


if __name__ == "__main__":
    main()
