#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RECONSTRUCTED_REPRODUCTION_SCRIPT

Clean Publication Schema A Grouped-CV CatBoost Training

This script provides a clean publication reference implementation of the
recovered Publication Schema A grouped-CV CatBoost training and model-persistence workflow.

The exact first historical root-model persistence entry point was not uniquely identified.
The training logic is independently present in multiple recovered original scripts, and
fig13_step2_surfaces.py explicitly regenerates five ModelB fold models that are
byte-identical to both the historical root models and the publicly released models.

This reference implementation uses the locked final configuration:
- StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
- Group: canonical structure × solvent
- CatBoost: iterations=288, depth=8, learning_rate=0.1, l2_leaf_reg=1, random_seed=42, Logloss
- Input: 5204 × 2283 Publication Schema A features
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix
import os

def train_publication_schemaA_grouped_cv(X_path, labels_path, output_dir):
    """
    Train Publication Schema A CatBoost with grouped-CV and save fold models.

    Parameters
    ----------
    X_path : str
        Path to frozen 5204×2283 Schema A matrix
    labels_path : str
        Path to master labels with groups
    output_dir : str
        Directory to save fold models

    Returns
    -------
    oof_predictions : ndarray
        Out-of-fold predictions for all 5204 observations
    """
    # Load private inputs
    X = np.load(X_path)
    labels = pd.read_parquet(labels_path)

    y = labels['OR_label'].values  # 1=OR+, 0=OR-
    groups = (labels['canonical_smiles'] + "||" + labels['solvent_group'].astype(str)).values

    # Publication configuration
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    catboost_params = {
        'iterations': 288,
        'depth': 8,
        'learning_rate': 0.1,
        'l2_leaf_reg': 1,
        'random_seed': 42,
        'loss_function': 'Logloss',
        'verbose': False
    }

    oof_predictions = np.zeros(len(y))

    os.makedirs(output_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = CatBoostClassifier(**catboost_params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

        # Save fold model
        model_path = os.path.join(output_dir, f"catboost_schemaA_fold{fold}.cbm")
        model.save_model(model_path)

        # OOF predictions
        oof_predictions[val_idx] = model.predict_proba(X_val)[:, 1]

        print(f"Fold {fold}: AUC={roc_auc_score(y_val, oof_predictions[val_idx]):.4f}")

    # Pooled metrics
    auc = roc_auc_score(y, oof_predictions)
    ap = average_precision_score(y, oof_predictions)

    print(f"\nPooled OOF AUC: {auc:.4f}")
    print(f"Pooled OOF AP: {ap:.4f}")

    return oof_predictions

if __name__ == "__main__":
    print("Publication Schema A Training Reference - REQUIRES PRIVATE INPUTS")
    print("This script requires private 5204×2283 matrix and master labels")
