#!/usr/bin/env python3
"""
Task 1: Canonical-SMILES-Only Grouped OOF Sensitivity Analysis
Uses publication X_C3_5204x2283.npy with molecule-level grouping
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef,
    brier_score_loss, confusion_matrix
)
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Paths
base_dir = Path("C:/Users/lenovo/.claude/projects/PM/JCIM_MANUSCRIPT/output_revision/d3full")
X_path = base_dir / "X_C3_5204x2283.npy"
labels_path = base_dir / "oof_predictions_master_labels_v2.parquet"
output_dir = Path("C:/Users/lenovo/.claude/projects/PM/JCIM_MANUSCRIPT")

print("="*80)
print("TASK 1: CANONICAL-SMILES-ONLY GROUPED OOF")
print("="*80)
print("\nObjective: Test molecule-level generalization")
print("Change: group = canonical_smiles (not canonical_smiles × solvent)")
print()

# Load data
print("Loading publication data...")
X = np.load(X_path)
df_labels = pd.read_parquet(labels_path)

print(f"X shape: {X.shape}")
print(f"Labels shape: {df_labels.shape}")

# Get labels and SMILES
y = df_labels['OR_label'].values
smiles = df_labels['canonical_smiles'].values
solvents = df_labels['solvent_group'].values if 'solvent_group' in df_labels.columns else None

print(f"\nLabel distribution:")
print(f"  OR- (0): {np.sum(y==0)}")
print(f"  OR+ (1): {np.sum(y==1)}")

# Create groups: ONLY canonical SMILES
groups_smiles = pd.Categorical(smiles).codes
n_unique_smiles = len(np.unique(groups_smiles))

print(f"\nGrouping:")
print(f"  Unique canonical SMILES: {n_unique_smiles}")
print(f"  Total observations: {len(y)}")
print(f"  Average obs per molecule: {len(y)/n_unique_smiles:.2f}")

# CatBoost configuration (LOCKED - from publication)
catboost_params = {
    'iterations': 288,
    'depth': 8,
    'learning_rate': 0.1,
    'l2_leaf_reg': 1,
    'random_seed': 42,
    'loss_function': 'Logloss',
    'verbose': False,
    'allow_writing_files': False
}

print("\nCatBoost Configuration (LOCKED):")
for k, v in catboost_params.items():
    print(f"  {k}: {v}")

# Setup CV
skf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nCross-Validation:")
print("  Method: StratifiedGroupKFold")
print("  n_splits: 5")
print("  shuffle: True")
print("  random_seed: 42")
print("  Group unit: canonical_smiles ONLY")

# Storage
oof_predictions = np.zeros(len(y))
fold_assignments = np.zeros(len(y), dtype=int)
fold_metrics = []

print("\n" + "="*80)
print("TRAINING FOLDS")
print("="*80)

try:
    splits = list(skf.split(X, y, groups=groups_smiles))
except Exception as e:
    print(f"\n[ERROR] ERROR in StratifiedGroupKFold split: {e}")
    print("\nAttempting diagnosis...")
    print(f"Unique groups: {n_unique_smiles}")
    print(f"Min group size: {pd.Series(groups_smiles).value_counts().min()}")
    raise

for fold, (train_idx, val_idx) in enumerate(splits, 1):
    print(f"\nFold {fold}/5")
    print("-" * 40)

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    groups_train = groups_smiles[train_idx]
    groups_val = groups_smiles[val_idx]

    # Verify no molecule leakage
    train_molecules = set(smiles[train_idx])
    val_molecules = set(smiles[val_idx])
    leakage = train_molecules & val_molecules

    if len(leakage) > 0:
        print(f"[ERROR] MOLECULE LEAKAGE DETECTED: {len(leakage)} molecules")
        print(f"Example leaked molecules: {list(leakage)[:5]}")
        raise ValueError("Canonical SMILES leakage detected!")

    n_train_molecules = len(np.unique(groups_train))
    n_val_molecules = len(np.unique(groups_val))

    print(f"  Train: {len(train_idx)} obs, {n_train_molecules} unique molecules")
    print(f"  Val:   {len(val_idx)} obs, {n_val_molecules} unique molecules")
    print(f"  Train OR+: {np.sum(y_train==1)} ({np.sum(y_train==1)/len(y_train)*100:.1f}%)")
    print(f"  Val OR+:   {np.sum(y_val==1)} ({np.sum(y_val==1)/len(y_val)*100:.1f}%)")
    print(f"  [OK] No molecule leakage")

    # Train
    model = CatBoostClassifier(**catboost_params)
    model.fit(X_train, y_train)

    # Predict
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    oof_predictions[val_idx] = y_pred_proba
    fold_assignments[val_idx] = fold

    # Metrics
    fold_auc = roc_auc_score(y_val, y_pred_proba)
    fold_acc = accuracy_score(y_val, (y_pred_proba >= 0.5).astype(int))
    fold_bal_acc = balanced_accuracy_score(y_val, (y_pred_proba >= 0.5).astype(int))

    print(f"  Fold AUC: {fold_auc:.4f}")
    print(f"  Fold Accuracy: {fold_acc:.4f}")
    print(f"  Fold Balanced Accuracy: {fold_bal_acc:.4f}")

    fold_metrics.append({
        'fold': fold,
        'train_n': len(train_idx),
        'val_n': len(val_idx),
        'train_molecules': n_train_molecules,
        'val_molecules': n_val_molecules,
        'train_OR_plus': int(np.sum(y_train==1)),
        'train_OR_minus': int(np.sum(y_train==0)),
        'val_OR_plus': int(np.sum(y_val==1)),
        'val_OR_minus': int(np.sum(y_val==0)),
        'auc': fold_auc,
        'accuracy': fold_acc,
        'balanced_accuracy': fold_bal_acc,
        'molecule_leakage': 0
    })

# Pooled OOF metrics
print("\n" + "="*80)
print("POOLED OOF METRICS")
print("="*80)

y_pred_class = (oof_predictions >= 0.5).astype(int)

# All metrics
roc_auc = roc_auc_score(y, oof_predictions)
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y, oof_predictions)
accuracy = accuracy_score(y, y_pred_class)
balanced_acc = balanced_accuracy_score(y, y_pred_class)
sensitivity = recall_score(y, y_pred_class)
specificity = recall_score(y, y_pred_class, pos_label=0)
precision = precision_score(y, y_pred_class)
npv = precision_score(y, y_pred_class, pos_label=0)
f1 = f1_score(y, y_pred_class)
mcc = matthews_corrcoef(y, y_pred_class)
brier = brier_score_loss(y, oof_predictions)

tn, fp, fn, tp = confusion_matrix(y, y_pred_class).ravel()

print(f"\nROC-AUC:              {roc_auc:.6f}")
print(f"Average Precision:    {average_precision:.6f}")
print(f"Accuracy:             {accuracy:.6f}")
print(f"Balanced Accuracy:    {balanced_acc:.6f}")
print(f"Sensitivity (Recall): {sensitivity:.6f}")
print(f"Specificity:          {specificity:.6f}")
print(f"Precision:            {precision:.6f}")
print(f"NPV:                  {npv:.6f}")
print(f"F1 Score:             {f1:.6f}")
print(f"MCC:                  {mcc:.6f}")
print(f"Brier Score:          {brier:.6f}")

print(f"\nConfusion Matrix:")
print(f"  TN: {tn}, FP: {fp}")
print(f"  FN: {fn}, TP: {tp}")

# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# OOF predictions
df_oof = pd.DataFrame({
    'sample_id': np.arange(len(y)),
    'canonical_smiles': smiles,
    'label': y,
    'oof_prediction': oof_predictions,
    'oof_class': y_pred_class,
    'fold': fold_assignments
})

if solvents is not None:
    df_oof['solvent'] = solvents

df_oof.to_csv(output_dir / "CANONICAL_SMILES_GROUPED_OOF_PREDICTIONS.csv", index=False)
print("[OK] Saved: CANONICAL_SMILES_GROUPED_OOF_PREDICTIONS.csv")

# Metrics
metrics = {
    'validation_scheme': 'canonical_SMILES_only',
    'grouping_unit': 'canonical_smiles',
    'n_folds': 5,
    'total_observations': len(y),
    'unique_molecules': n_unique_smiles,
    'OR_minus': int(np.sum(y==0)),
    'OR_plus': int(np.sum(y==1)),
    'pooled_oof': {
        'roc_auc': roc_auc,
        'average_precision': average_precision,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'npv': npv,
        'f1': f1,
        'mcc': mcc,
        'brier': brier,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    },
    'fold_metrics': fold_metrics
}

with open(output_dir / "CANONICAL_SMILES_GROUPED_METRICS.json", 'w') as f:
    json.dump(metrics, f, indent=2)
print("[OK] Saved: CANONICAL_SMILES_GROUPED_METRICS.json")

# Fold audit
df_fold_audit = pd.DataFrame(fold_metrics)
df_fold_audit.to_csv(output_dir / "CANONICAL_SMILES_GROUPED_FOLD_AUDIT.csv", index=False)
print("[OK] Saved: CANONICAL_SMILES_GROUPED_FOLD_AUDIT.csv")

print("\n" + "="*80)
print("TASK 1 COMPLETE")
print("="*80)
print(f"\n[OK] Molecule-level grouped OOF AUC: {roc_auc:.4f}")
print(f"[OK] Zero molecule leakage verified across all 5 folds")
print(f"[OK] All {n_unique_smiles} unique molecules properly grouped")
