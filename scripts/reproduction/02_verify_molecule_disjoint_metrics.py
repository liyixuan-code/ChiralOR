#!/usr/bin/env python3
"""
Verify Molecule-Disjoint Metrics

Type: RECONSTRUCTED_REPRODUCTION_SCRIPT
Purpose: Verify molecule-disjoint sensitivity analysis
Input: results/molecule_disjoint/molecule_disjoint_oof_predictions.csv
Output: All molecule-disjoint metrics and grouping constraint verification
"""

import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, recall_score, precision_score,
    f1_score, matthews_corrcoef, brier_score_loss
)

EXPECTED = {
    'ROC-AUC': 0.9264,
    'AP': 0.9033,
    'Accuracy': 0.8561,
    'Balanced Accuracy': 0.8442,
    'Sensitivity': 0.7867,
    'Specificity': 0.9016,
    'Precision': 0.8401,
    'NPV': 0.8655,
    'F1': 0.8125,
    'MCC': 0.6969,
    'Brier': 0.1035
}

def main():
    # Load predictions
    df = pd.read_csv('results/molecule_disjoint/molecule_disjoint_oof_predictions.csv')

    y_true = df['observed_label']
    y_prob = df['oof_probability_or_plus']
    y_pred = df['predicted_label_t0_5']

    # Compute metrics
    results = {
        'ROC-AUC': roc_auc_score(y_true, y_prob),
        'AP': average_precision_score(y_true, y_prob),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Sensitivity': recall_score(y_true, y_pred),
        'Specificity': recall_score(y_true, y_pred, pos_label=0),
        'Precision': precision_score(y_true, y_pred),
        'NPV': precision_score(y_true, y_pred, pos_label=0),
        'F1': f1_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Brier': brier_score_loss(y_true, y_prob)
    }

    print("MOLECULE-DISJOINT SENSITIVITY VERIFICATION")
    print("=" * 60)
    print(f"Dataset: {len(df)} observations")
    print()

    # Verify grouping constraint
    audit = pd.read_csv('results/molecule_disjoint/molecule_disjoint_group_audit.csv')
    violations = audit[audit['n_unique_folds'] > 1]
    print(f"Unique molecules: {len(audit)}")
    print(f"Grouping constraint violations: {len(violations)}")
    if len(violations) > 0:
        print("WARNING: Some molecules span multiple folds!")
    print()

    print(f"{'Metric':<20} {'Paper':<12} {'Reproduced':<15} {'Status'}")
    print("-" * 60)

    all_match = True
    for metric, expected in EXPECTED.items():
        reproduced = results[metric]
        diff = abs(reproduced - expected)
        status = "MATCH" if diff < 0.0001 else "MISMATCH"
        if diff >= 0.0001:
            all_match = False
        print(f"{metric:<20} {expected:<12.4f} {reproduced:<15.10f} {status}")

    print()
    if all_match and len(violations) == 0:
        print("STATUS: ALL METRICS AND CONSTRAINTS PASS")
        return 0
    else:
        print("STATUS: ISSUE DETECTED")
        return 1

if __name__ == '__main__':
    exit(main())
