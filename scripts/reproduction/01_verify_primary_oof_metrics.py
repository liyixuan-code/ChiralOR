#!/usr/bin/env python3
"""Verify Primary Grouped-OOF Metrics
Type: RECONSTRUCTED_REPRODUCTION_SCRIPT"""
import pandas as pd
from sklearn.metrics import *

EXPECTED = {'ROC-AUC': 0.9278, 'AP': 0.9005, 'Accuracy': 0.8599, 'Balanced Accuracy': 0.8495,
            'Sensitivity': 0.7993, 'Specificity': 0.8997, 'Precision': 0.8396, 'NPV': 0.8722,
            'F1': 0.8190, 'MCC': 0.7054, 'Brier': 0.1027}

df = pd.read_csv('results/primary_oof/primary_grouped_oof_predictions.csv')
y_true, y_prob = df['observed_label'], df['oof_probability_or_plus']
y_pred = (y_prob >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

results = {
    'ROC-AUC': roc_auc_score(y_true, y_prob), 'AP': average_precision_score(y_true, y_prob),
    'Accuracy': accuracy_score(y_true, y_pred), 'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
    'Sensitivity': recall_score(y_true, y_pred), 'Specificity': recall_score(y_true, y_pred, pos_label=0),
    'Precision': precision_score(y_true, y_pred), 'NPV': precision_score(y_true, y_pred, pos_label=0),
    'F1': f1_score(y_true, y_pred), 'MCC': matthews_corrcoef(y_true, y_pred),
    'Brier': brier_score_loss(y_true, y_prob)}

print("PRIMARY OOF VERIFICATION\n" + "="*60)
print(f"Dataset: {len(df)} observations\nFolds: {df['fold'].nunique()}\n")
print(f"{'Metric':<20} {'Paper':<12} {'Reproduced':<15} {'Status'}")
print("-"*60)

all_match = True
for metric, expected in EXPECTED.items():
    reproduced = results[metric]
    status = "MATCH" if abs(reproduced - expected) < 0.0001 else "MISMATCH"
    if abs(reproduced - expected) >= 0.0001: all_match = False
    print(f"{metric:<20} {expected:<12.4f} {reproduced:<15.10f} {status}")

print(f"\nConfusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"\nSTATUS: {'ALL METRICS PASS' if all_match else 'MISMATCH DETECTED'}")
exit(0 if all_match else 1)
