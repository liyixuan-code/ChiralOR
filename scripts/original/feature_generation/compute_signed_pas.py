#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 2: Compute Signed-PAS (sPAS) Descriptors
===============================================
Problem: Current PAS uses permutation-invariant statistics (std, mean, range, max_diff).
         Enantiomers have identical PAS → 35% of SHAP importance is wasted.

Fix: Use CIP priority to ORDER the 4 ligands, then compute SIGNED pairwise differences.
     For enantiomers, CIP order is reversed → signed differences flip → sPAS distinguishes them.

Input:  scripts/processed_data_v2.csv  (from Step 1)
Output: scripts/processed_data_v2.csv  (updated with sPAS_0..sPAS_19 columns)

Usage: python scripts/compute_signed_pas.py

Requires: RDKit
"""
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

BASE = Path(__file__).resolve().parent.parent
CSV_PATH = BASE / 'scripts' / 'processed_data_v2.csv'

# Pauling electronegativity
ELECTRONEGATIVITY = {
    'H': 2.2, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
    'S': 2.58, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66, 'P': 2.19,
    'B': 2.04, 'Si': 1.90, 'Se': 2.55,
}

PROP_NAMES = ['charge', 'vdw_radius', 'electronegativity', 'mass', 'polarizability']


def get_ligand_properties(mol, ligand_idx):
    """Get 5 physicochemical properties for a ligand atom."""
    if ligand_idx == -1:  # Implicit hydrogen
        return np.array([0.0, 1.2, 2.2, 1.008, 0.667])

    atom = mol.GetAtomWithIdx(ligand_idx)

    # Gasteiger charge
    try:
        charge = float(atom.GetProp('_GasteigerCharge'))
        if np.isnan(charge) or np.isinf(charge):
            charge = 0.0
    except:
        charge = 0.0

    # Van der Waals radius
    try:
        vdw_radius = Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())
    except:
        vdw_radius = 1.7

    # Electronegativity
    electronegativity = ELECTRONEGATIVITY.get(atom.GetSymbol(), 2.5)

    # Mass
    mass = atom.GetMass()

    # Polarizability (approximate)
    polarizability = vdw_radius ** 3

    return np.array([charge, vdw_radius, electronegativity, mass, polarizability])


def get_cip_ranked_neighbors(mol, chiral_idx):
    """
    Get the 4 neighbors of a chiral center, ordered by CIP priority.
    Returns list of atom indices in CIP priority order (highest first).
    For enantiomers (R vs S), the order will be reversed.
    """
    atom = mol.GetAtomWithIdx(chiral_idx)
    neighbors = [n.GetIdx() for n in atom.GetNeighbors()]

    # If only 3 explicit neighbors, add implicit H as -1
    if len(neighbors) == 3:
        neighbors.append(-1)
    elif len(neighbors) != 4:
        return None

    # Use RDKit's CIP ranking
    # The _CIPRank property is assigned by AssignStereochemistry
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    ranked = []
    for n_idx in neighbors:
        if n_idx == -1:
            # Hydrogen has lowest CIP priority
            ranked.append((0, n_idx))
        else:
            n_atom = mol.GetAtomWithIdx(n_idx)
            # Try to get CIP rank (higher = higher priority)
            try:
                cip_rank = int(n_atom.GetUnsignedProp('_CIPRank'))
            except:
                # Fallback: use atomic number
                cip_rank = n_atom.GetAtomicNum()
            ranked.append((cip_rank, n_idx))

    # Sort by CIP rank DESCENDING (highest priority first)
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [idx for _, idx in ranked]


def compute_signed_pas(mol, chiral_idx):
    """
    Compute 20-dimensional Signed-PAS descriptor.

    Ligands are ordered by CIP priority: L1 > L2 > L3 > L4
    For each of 5 properties:
      sPAS[0-4]   = L1 - L2 (signed difference, highest vs 2nd)
      sPAS[5-9]   = L1 - L3 (signed difference, highest vs 3rd)
      sPAS[10-14]  = L2 - L3 (signed difference, 2nd vs 3rd)
      sPAS[15-19]  = std across all 4 (backward compatible with original PAS)

    For enantiomers: CIP ordering flips (R↔S swaps the sense of the tetrahedron),
    causing signed differences to change sign → sPAS distinguishes enantiomers.
    """
    # Compute Gasteiger charges (needed for ligand properties)
    AllChem.ComputeGasteigerCharges(mol)

    ordered_neighbors = get_cip_ranked_neighbors(mol, chiral_idx)
    if ordered_neighbors is None:
        return np.zeros(20)

    # Get properties for each ligand in CIP order
    props = np.array([get_ligand_properties(mol, n) for n in ordered_neighbors])
    # props shape: (4, 5) — 4 ligands, 5 properties

    L1, L2, L3, L4 = props[0], props[1], props[2], props[3]

    # Signed differences (order matters for enantiomer distinction)
    diff_12 = L1 - L2    # 5 values
    diff_13 = L1 - L3    # 5 values
    diff_23 = L2 - L3    # 5 values

    # Standard deviation (permutation-invariant, backward compatible)
    std_vals = props.std(axis=0)  # 5 values

    # Concatenate: 20 dimensions total
    spas = np.concatenate([diff_12, diff_13, diff_23, std_vals])

    return spas


def compute_original_pas(mol, chiral_idx):
    """Compute original 20-dim PAS (permutation-invariant) for comparison."""
    AllChem.ComputeGasteigerCharges(mol)

    atom = mol.GetAtomWithIdx(chiral_idx)
    neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
    if len(neighbors) == 3:
        neighbors.append(-1)

    props = np.array([get_ligand_properties(mol, n) for n in neighbors])

    std_features = props.std(axis=0)
    mean_features = props.mean(axis=0)
    range_features = props.max(axis=0) - props.min(axis=0)

    # Max pairwise diff
    pair_diff = []
    for col in range(5):
        vals = props[:, col]
        max_d = max(abs(vals[i] - vals[j]) for i in range(4) for j in range(i + 1, 4))
        pair_diff.append(max_d)
    pair_diff = np.array(pair_diff)

    return np.concatenate([std_features, mean_features, range_features, pair_diff])


def main():
    print("=" * 60)
    print("  Step 2: Compute Signed-PAS Descriptors")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH)
    print(f"  Loaded: {len(df)} samples")

    spas_all = []
    pas_orig_all = []
    n_failed = 0

    for idx, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['smi'])
            if mol is None:
                raise ValueError("Invalid SMILES")

            chiral_idx = int(row['chiral_idx'])
            spas = compute_signed_pas(mol, chiral_idx)
            pas_orig = compute_original_pas(mol, chiral_idx)
        except Exception as e:
            spas = np.zeros(20)
            pas_orig = np.zeros(20)
            n_failed += 1

        spas_all.append(spas)
        pas_orig_all.append(pas_orig)

        if (idx + 1) % 1000 == 0:
            print(f"    Progress: {idx + 1}/{len(df)}")

    spas_arr = np.array(spas_all)
    pas_orig_arr = np.array(pas_orig_all)

    print(f"  Computed: {len(df)} samples ({n_failed} failed)")

    # --- Verify enantiomer distinction ---
    print(f"\n  --- Enantiomer distinction test ---")
    test_pairs = [(5372, 5373), (5392, 5393), (1314, 1315)]
    for idx_a, idx_b in test_pairs:
        if idx_a < len(df) and idx_b < len(df):
            sa, sb = spas_arr[idx_a], spas_arr[idx_b]
            pa, pb = pas_orig_arr[idx_a], pas_orig_arr[idx_b]
            spas_diff = np.abs(sa - sb).sum()
            pas_diff = np.abs(pa - pb).sum()
            print(f"    #{idx_a} vs #{idx_b}:")
            print(f"      sPAS L1-norm diff: {spas_diff:.6f} {'DIFFERENT' if spas_diff > 1e-6 else 'SAME (BAD!)'}")
            print(f"      PAS  L1-norm diff: {pas_diff:.6f} {'DIFFERENT' if pas_diff > 1e-6 else 'SAME (expected)'}")

    # --- Save sPAS and original PAS columns ---
    for i in range(20):
        df[f'sPAS_{i}'] = spas_arr[:, i]
    for i in range(20):
        df[f'PAS_{i}'] = pas_orig_arr[:, i]  # Overwrite with freshly computed

    df.to_csv(CSV_PATH, index=False)
    print(f"\n  Updated: {CSV_PATH}")
    print(f"  New columns: sPAS_0..sPAS_19 (signed), PAS_0..PAS_19 (original, recomputed)")

    # --- Summary statistics ---
    print(f"\n  sPAS summary:")
    for i in [0, 5, 10, 15]:
        col = f'sPAS_{i}'
        print(f"    {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}, "
              f"min={df[col].min():.4f}, max={df[col].max():.4f}")


if __name__ == '__main__':
    main()
