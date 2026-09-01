#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3-4: Rebuild Feature Cache (V2)
======================================
Assembles the new feature matrix with:
  - ECFP6 (2048) — unchanged, useChirality=True
  - MACCS (167) — unchanged
  - RDKit 2D (8) — unchanged
  - sPAS (20) — NEW: signed PAS from compute_signed_pas.py
  - PAS original (20) — keep for backward compatibility
  - 3D shape (10) — unchanged
  - Solvent (1) — unchanged
  - Dihedral (1) — NEW: raw dihedral angle
  - PM code (1) — NEW: P=+1, M=-1

Total: 2048 + 167 + 8 + 20 + 20 + 10 + 1 + 1 + 1 = 2276 dimensions

Input:  scripts/processed_data_v2.csv (from Steps 1-2)
        v13_revised/features_cache.npz (original ECFP+MACCS+RDKit2D+3D)
        outputs/v24_audit/ (keep_mask, y_kept, w_kept)

Output: outputs/v26_data/X_kept.npy (2276-dim)
        outputs/v26_data/y_kept.npy
        outputs/v26_data/w_kept.npy
        outputs/v26_data/keep_mask_v24.npy
        outputs/v26_data/feature_cols.npy

Usage: python scripts/rebuild_features_v2.py

Requires: RDKit, numpy, pandas
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CSV_PATH = BASE / 'scripts' / 'processed_data_v2.csv'
OLD_CACHE = BASE / 'v13_revised' / 'features_cache.npz'
V24_DIR = BASE / 'outputs' / 'v24_audit'
OUT_DIR = BASE / 'outputs' / 'v26_data'


def main():
    print("=" * 60)
    print("  Step 3-4: Rebuild Feature Cache (V2 → 2276 dims)")
    print("=" * 60)

    # --- Load processed data ---
    df = pd.read_csv(CSV_PATH)
    print(f"  Loaded CSV: {len(df)} samples")

    # Check that sPAS columns exist
    spas_cols = [f'sPAS_{i}' for i in range(20)]
    pas_cols = [f'PAS_{i}' for i in range(20)]
    missing_spas = [c for c in spas_cols if c not in df.columns]
    if missing_spas:
        raise RuntimeError(f"Missing sPAS columns: {missing_spas}. Run compute_signed_pas.py first!")

    # --- Load original feature cache (for ECFP, MACCS, RDKit2D, 3D) ---
    cache = np.load(OLD_CACHE, allow_pickle=True)
    X_old = cache['X']
    feature_cols_old = cache['feature_cols'].tolist()
    print(f"  Old cache: shape={X_old.shape}, features={len(feature_cols_old)}")

    assert X_old.shape[0] == len(df), \
        f"Shape mismatch: cache has {X_old.shape[0]} but CSV has {len(df)} rows"

    # --- Extract old feature blocks by column name ---
    ecfp_idx = [feature_cols_old.index(f'ECFP_{i}') for i in range(2048)]
    maccs_idx = [feature_cols_old.index(f'MACCS_{i}') for i in range(167)]
    rdkit_names = ['MolWt', 'MolLogP', 'TPSA', 'NumHAcceptors',
                   'NumHDonors', 'NumRotBonds', 'RingCount', 'NumAromRings']
    rdkit_idx = [feature_cols_old.index(c) for c in rdkit_names]
    d3_names = ['3D_PMI1', '3D_PMI2', '3D_PMI3', '3D_NPR1', '3D_NPR2',
                '3D_RoG', '3D_ISF', '3D_Ecc', '3D_Asph', '3D_Sphero']
    d3_idx = [feature_cols_old.index(c) for c in d3_names]
    solvent_idx = [feature_cols_old.index('solvent_code')]

    X_ecfp = X_old[:, ecfp_idx]       # (N, 2048)
    X_maccs = X_old[:, maccs_idx]      # (N, 167)
    X_rdkit = X_old[:, rdkit_idx]      # (N, 8)
    X_3d = X_old[:, d3_idx]            # (N, 10)
    X_solvent = X_old[:, solvent_idx]  # (N, 1)

    print(f"  ECFP: {X_ecfp.shape}, MACCS: {X_maccs.shape}, "
          f"RDKit: {X_rdkit.shape}, 3D: {X_3d.shape}, Solvent: {X_solvent.shape}")

    # --- Extract new features from DataFrame ---
    X_spas = df[spas_cols].values.astype(np.float32)   # (N, 20) — NEW
    X_pas = df[pas_cols].values.astype(np.float32)      # (N, 20) — recomputed original

    # Dihedral angle (NEW feature)
    X_dihedral = df['dihedral'].values.astype(np.float32).reshape(-1, 1)  # (N, 1)
    # Handle NaN dihedrals
    X_dihedral = np.nan_to_num(X_dihedral, nan=0.0)

    # PM code: P=+1, M=-1 (NEW feature)
    X_pm = np.where(df['PM'].values == 'P', 1.0, -1.0).astype(np.float32).reshape(-1, 1)  # (N, 1)

    print(f"  sPAS: {X_spas.shape}, PAS: {X_pas.shape}, "
          f"Dihedral: {X_dihedral.shape}, PM: {X_pm.shape}")

    # --- Assemble new feature matrix ---
    # Order: ECFP(2048) + MACCS(167) + RDKit(8) + sPAS(20) + PAS(20) + 3D(10) + Solvent(1) + Dihedral(1) + PM(1) = 2276
    X_new = np.hstack([
        X_ecfp,       # 2048
        X_maccs,      # 167
        X_rdkit,      # 8
        X_spas,       # 20  (NEW)
        X_pas,        # 20  (recomputed)
        X_3d,         # 10
        X_solvent,    # 1
        X_dihedral,   # 1   (NEW)
        X_pm,         # 1   (NEW)
    ]).astype(np.float32)

    feature_cols_new = (
        [f'ECFP_{i}' for i in range(2048)] +
        [f'MACCS_{i}' for i in range(167)] +
        rdkit_names +
        [f'sPAS_{i}' for i in range(20)] +
        [f'PAS_{i}' for i in range(20)] +
        d3_names +
        ['solvent_code', 'dihedral', 'pm_code']
    )

    print(f"\n  New feature matrix: {X_new.shape} ({len(feature_cols_new)} features)")
    assert X_new.shape[1] == len(feature_cols_new) == 2276, \
        f"Expected 2276, got {X_new.shape[1]}"

    # --- Check for NaN ---
    nan_count = np.isnan(X_new).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values found, replacing with 0")
        X_new = np.nan_to_num(X_new, nan=0.0)

    # --- Apply V24 audit mask ---
    if V24_DIR.exists():
        keep_mask = np.load(V24_DIR / 'keep_mask_v24.npy')
        y_v24 = np.load(V24_DIR / 'y_kept.npy').astype(int)
        w_v24 = np.load(V24_DIR / 'w_kept.npy')
        print(f"\n  V24 audit: keep_mask sum={keep_mask.sum()}, y shape={y_v24.shape}")

        # Apply mask to new features
        X_kept = X_new[keep_mask]
        y_kept = y_v24  # Already filtered in V24
        w_kept = w_v24

        # Verify shapes match
        assert X_kept.shape[0] == len(y_kept), \
            f"Shape mismatch: X_kept={X_kept.shape[0]}, y_kept={len(y_kept)}"
        print(f"  After V24 mask: X={X_kept.shape}, y={y_kept.shape}, w={w_kept.shape}")
    else:
        print(f"\n  WARNING: V24 audit dir not found at {V24_DIR}")
        print(f"  Using full dataset without audit filtering")
        y_or = (df['OR'] > 0).astype(int).values
        X_kept = X_new
        y_kept = y_or
        w_kept = np.ones(len(y_or), dtype=np.float32)
        keep_mask = np.ones(len(df), dtype=bool)

    # --- Save ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUT_DIR / 'X_kept.npy', X_kept)
    np.save(OUT_DIR / 'y_kept.npy', y_kept)
    np.save(OUT_DIR / 'w_kept.npy', w_kept)
    np.save(OUT_DIR / 'keep_mask_v24.npy', keep_mask)
    np.save(OUT_DIR / 'feature_cols.npy', np.array(feature_cols_new))

    # Also save full (unfiltered) feature cache for other analyses
    np.savez_compressed(
        OUT_DIR / 'features_cache_v2.npz',
        X=X_new,
        feature_cols=np.array(feature_cols_new),
    )

    print(f"\n  Saved to {OUT_DIR}:")
    print(f"    X_kept.npy:       {X_kept.shape}")
    print(f"    y_kept.npy:       {y_kept.shape}")
    print(f"    w_kept.npy:       {w_kept.shape}")
    print(f"    keep_mask_v24.npy: {keep_mask.shape} (sum={keep_mask.sum()})")
    print(f"    feature_cols.npy:  {len(feature_cols_new)} features")
    print(f"    features_cache_v2.npz: {X_new.shape}")

    # --- Spot check the 3 target compounds ---
    print(f"\n  --- Target compound feature check ---")
    for target_idx in [5373, 5393, 1314]:
        if target_idx < len(df):
            print(f"  #{target_idx}:")
            print(f"    sPAS[0:5] = {X_spas[target_idx, :5]}")
            print(f"    PAS[0:5]  = {X_pas[target_idx, :5]}")
            print(f"    dihedral  = {X_dihedral[target_idx, 0]:.4f}")
            print(f"    pm_code   = {X_pm[target_idx, 0]:.0f}")

    print(f"\n{'=' * 60}")
    print(f"  Feature rebuild complete!")
    print(f"  Next: python scripts/run_v26_superlearner.py")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
