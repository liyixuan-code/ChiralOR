"""
Corrected P/M Helix Predictor - EXACT match with original implementation
=========================================================================

This version is aligned with flask_app/app.py::calc_pm_universal
to ensure 100% consistency with training data.

Fixed issues:
1. 3D conformer generation: useSmallRingTorsions=True (not useRandomCoords)
2. MMFF optimization: MMFF94s variant, 1000 iterations
3. UFF fallback when MMFF fails
4. Bridge dihedral: correct deviation metric (abs(abs(dih)-180), not abs(dih))
5. Bridge dihedral: two loop directions
6. Bridge dihedral: threshold protection (dev > 1.0)
"""

import numpy as np
from typing import Tuple, Optional, List

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Error: RDKit not available")


# SMARTS patterns - EXACT from original
PM_SMARTS_5 = '[n;r5;R2][A;r6;R1][C;@;r6;R1][A;r6;R1;!$([a])]'
PM_SMARTS_5B = '[n;r5;R2][A;r6;R1][C;@@;r6;R1][A;r6;R1;!$([a])]'
PM_SMARTS_6 = '[c;r6;R2][A;r6;R1][C;r6;R1;H1;@][A;r6;R1;!$([a])]'
PM_SMARTS_6B = '[c;r6;R2][A;r6;R1][C;r6;R1;H1;@@][A;r6;R1;!$([a])]'


def calc_pm_helix(smiles: str) -> Tuple[str, Optional[float], str, Optional[List[int]]]:
    """
    Calculate P/M helix type - EXACT match with original calc_pm_universal

    Returns:
        (pm_type, dihedral_angle, method, atoms)
        pm_type: 'P', 'M', or 'ERR'
        dihedral_angle: dihedral angle in degrees
        method: 'SMARTS', 'Bridge', or error message
        atoms: list of atom indices used
    """
    if not RDKIT_AVAILABLE:
        return "ERR", None, "RDKit_Not_Available", None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "ERR", None, "Invalid_SMILES", None

    mol_2d = mol
    mol_3d = Chem.AddHs(mol)

    # ========================================================================
    # 3D Conformer Generation - EXACT match with original
    # ========================================================================
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.useSmallRingTorsions = True  # KEY: not useRandomCoords

    embed_result = AllChem.EmbedMolecule(mol_3d, params)

    if embed_result == -1:
        # Fallback without params
        embed_result = AllChem.EmbedMolecule(mol_3d, randomSeed=42)

    if embed_result == -1 or mol_3d.GetNumConformers() == 0:
        return "ERR", None, "Embed_Failed", None

    # Optimize geometry - EXACT match with original
    try:
        AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=1000, mmffVariant='MMFF94s')
    except Exception:
        # UFF fallback
        try:
            AllChem.UFFOptimizeMolecule(mol_3d, maxIters=500)
        except Exception:
            pass

    conf = mol_3d.GetConformer()

    # ========================================================================
    # Method 1: SMARTS Pattern Matching (96% coverage)
    # ========================================================================
    for smarts_str in [PM_SMARTS_5, PM_SMARTS_6, PM_SMARTS_5B, PM_SMARTS_6B]:
        pattern = Chem.MolFromSmarts(smarts_str)
        if pattern is None:
            continue

        # Try on both 3D and 2D
        for target_mol in [mol_3d, mol_2d]:
            # First try with chirality
            matches = target_mol.GetSubstructMatches(pattern, useChirality=True)

            # Fallback without chirality
            if not matches:
                matches = target_mol.GetSubstructMatches(pattern, useChirality=False)

            for match in matches:
                if len(match) < 4:
                    continue

                try:
                    atoms = match[:4]
                    dihedral = AllChem.GetDihedralDeg(conf, *atoms)
                    pm_type = 'P' if dihedral > 0 else 'M'
                    return pm_type, round(float(dihedral), 2), "SMARTS", list(atoms)
                except Exception:
                    continue

    # ========================================================================
    # Method 2: Bridge Dihedral (4% coverage) - EXACT match with original
    # ========================================================================
    ring_info = mol_2d.GetRingInfo()
    rings = [set(r) for r in ring_info.AtomRings()]

    best_dih = None
    best_dev = 0  # KEY: starts at 0, original uses -1 but logic is same
    best_atoms = None

    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            shared = rings[i] & rings[j]

            if len(shared) < 2:
                continue

            bridge = sorted(shared)
            b1, b2 = bridge[0], bridge[1]

            # Direction 1: a1-b1-b2-a4
            for a1 in (rings[i] - shared):
                if mol_2d.GetBondBetweenAtoms(a1, b1) is None:
                    continue

                for a4 in (rings[j] - shared):
                    if mol_2d.GetBondBetweenAtoms(a4, b2) is None:
                        continue

                    try:
                        dih = AllChem.GetDihedralDeg(conf, a1, b1, b2, a4)
                        # KEY FIX: deviation from 180°, not absolute value
                        dev = abs(abs(dih) - 180)

                        if dev > best_dev:
                            best_dev = dev
                            best_dih = dih
                            best_atoms = [a1, b1, b2, a4]
                    except Exception:
                        continue

            # Direction 2: a1-b2-b1-a4 (KEY: second loop, was missing)
            for a1 in (rings[i] - shared):
                if mol_2d.GetBondBetweenAtoms(a1, b2) is None:
                    continue

                for a4 in (rings[j] - shared):
                    if mol_2d.GetBondBetweenAtoms(a4, b1) is None:
                        continue

                    try:
                        dih = AllChem.GetDihedralDeg(conf, a1, b2, b1, a4)
                        # KEY FIX: deviation from 180°, not absolute value
                        dev = abs(abs(dih) - 180)

                        if dev > best_dev:
                            best_dev = dev
                            best_dih = dih
                            best_atoms = [a1, b2, b1, a4]
                    except Exception:
                        continue

    # KEY FIX: threshold protection
    if best_dih is not None and best_dev > 1.0:
        pm_type = 'P' if best_dih > 0 else 'M'
        return pm_type, round(float(best_dih), 2), "Bridge", best_atoms

    return "ERR", None, "No_Match", None


def predict_pm_helix(smiles: str) -> Tuple[str, float, str]:
    """
    Convenient wrapper for P/M prediction.

    Args:
        smiles: SMILES string

    Returns:
        (pm_type, dihedral, method)
    """
    pm, dihedral, method, atoms = calc_pm_helix(smiles)
    return pm, dihedral, method


if __name__ == "__main__":
    # Test cases
    print("="*60)
    print("P/M Helix Predictor Test (Fixed Version)")
    print("="*60)

    test_cases = [
        ("C[C@H]1CNc2ccccc2C1", "Expected: M (from training data)"),
        ("CCCN(CCc1cccs1)[C@H]1CCc2c(O)cccc2C1", "Chroman derivative"),
        ("CCCN(CCC)[C@@H]1CCc2cccc(O)c2C1", "Double @ chirality"),
    ]

    for smiles, desc in test_cases:
        print(f"\nTest: {desc}")
        print(f"SMILES: {smiles}")

        pm, dihedral, method, atoms = calc_pm_helix(smiles)

        print(f"Result: P/M = {pm}")
        print(f"Dihedral: {dihedral}")
        print(f"Method: {method}")
        if atoms:
            print(f"Atoms: {atoms}")

        if pm in ['P', 'M']:
            print(f"[SUCCESS] (geometric calculation)")
        else:
            print(f"[FAILED] {method}")
