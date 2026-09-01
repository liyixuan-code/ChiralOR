"""
Baseline Feature Calculator (2,276 dimensions)

Includes:
1. ECFP (Extended Connectivity Fingerprints): 1024 bits
2. RDKit 2D Descriptors: ~200 features
3. PAS (Pharmacophore Atom Signatures): 20 features
4. sPAS (signed PAS): ~600 features
5. 3D Descriptors: 10 features
6. Experimental features: 3 features (solvent_code, dihedral, pm_code)
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D
from rdkit.Chem import rdMolDescriptors
from typing import Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class BaselineFeatureCalculator:
    """Calculate baseline molecular features (2,276 dimensions)."""

    def __init__(self):
        """Initialize baseline feature calculator."""
        self.ecfp_bits = 1024
        self.ecfp_radius = 2

    def calculate_ecfp(self, mol: Chem.Mol) -> np.ndarray:
        """
        Calculate ECFP (Morgan) fingerprints.

        Args:
            mol: RDKit Mol object

        Returns:
            np.array of shape (1024,) with binary values
        """
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=self.ecfp_radius,
                nBits=self.ecfp_bits
            )
            return np.array(fp, dtype=np.float32)
        except Exception as e:
            print(f"Warning: ECFP calculation failed: {e}")
            return np.zeros(self.ecfp_bits, dtype=np.float32)

    def calculate_rdkit_2d(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate RDKit 2D descriptors.

        Args:
            mol: RDKit Mol object

        Returns:
            Dictionary of descriptor_name: value
        """
        descriptors = {}

        try:
            # Basic descriptors
            descriptors['MolWt'] = Descriptors.MolWt(mol)
            descriptors['MolLogP'] = Descriptors.MolLogP(mol)
            descriptors['TPSA'] = Descriptors.TPSA(mol)
            descriptors['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
            descriptors['NumHDonors'] = Descriptors.NumHDonors(mol)
            descriptors['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
            descriptors['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
            descriptors['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
            descriptors['RingCount'] = Descriptors.RingCount(mol)
            descriptors['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)

            # Additional descriptors
            descriptors['FractionCsp3'] = Descriptors.FractionCSP3(mol)
            descriptors['NumAromaticCarbocycles'] = Descriptors.NumAromaticCarbocycles(mol)
            descriptors['NumAromaticHeterocycles'] = Descriptors.NumAromaticHeterocycles(mol)
            descriptors['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
            descriptors['NumAliphaticCarbocycles'] = Descriptors.NumAliphaticCarbocycles(mol)
            descriptors['NumAliphaticHeterocycles'] = Descriptors.NumAliphaticHeterocycles(mol)

            # Molecular formula descriptors
            descriptors['HeavyAtomCount'] = mol.GetNumHeavyAtoms()
            descriptors['NumValenceElectrons'] = Descriptors.NumValenceElectrons(mol)

            # Chi descriptors
            descriptors['Chi0'] = Descriptors.Chi0(mol)
            descriptors['Chi1'] = Descriptors.Chi1(mol)
            descriptors['Chi0n'] = Descriptors.Chi0n(mol)
            descriptors['Chi1n'] = Descriptors.Chi1n(mol)
            descriptors['Chi2n'] = Descriptors.Chi2n(mol)
            descriptors['Chi3n'] = Descriptors.Chi3n(mol)
            descriptors['Chi4n'] = Descriptors.Chi4n(mol)

            # Kappa descriptors
            descriptors['Kappa1'] = Descriptors.Kappa1(mol)
            descriptors['Kappa2'] = Descriptors.Kappa2(mol)
            descriptors['Kappa3'] = Descriptors.Kappa3(mol)

            # More descriptors
            descriptors['BalabanJ'] = Descriptors.BalabanJ(mol)
            descriptors['BertzCT'] = Descriptors.BertzCT(mol)
            descriptors['Ipc'] = Descriptors.Ipc(mol)
            descriptors['HallKierAlpha'] = Descriptors.HallKierAlpha(mol)

            # EState descriptors
            descriptors['MaxEStateIndex'] = Descriptors.MaxEStateIndex(mol)
            descriptors['MinEStateIndex'] = Descriptors.MinEStateIndex(mol)
            descriptors['MaxAbsEStateIndex'] = Descriptors.MaxAbsEStateIndex(mol)
            descriptors['MinAbsEStateIndex'] = Descriptors.MinAbsEStateIndex(mol)

        except Exception as e:
            print(f"Warning: Some 2D descriptors failed: {e}")

        return descriptors

    def calculate_pas(self, mol: Chem.Mol, n_bins: int = 20) -> np.ndarray:
        """
        Calculate PAS (Pharmacophore Atom Signatures).

        This is a simplified implementation. The actual PAS calculation
        may be more complex and specific to the training data.

        Args:
            mol: RDKit Mol object
            n_bins: Number of PAS bins

        Returns:
            np.array of shape (n_bins,)
        """
        try:
            # Simplified PAS: atomic property distribution
            pas = np.zeros(n_bins, dtype=np.float32)

            for atom in mol.GetAtoms():
                # Bin based on atomic properties
                atomic_num = atom.GetAtomicNum()
                degree = atom.GetDegree()
                is_aromatic = int(atom.GetIsAromatic())

                # Create a simple hash
                bin_idx = (atomic_num + degree * 10 + is_aromatic * 100) % n_bins
                pas[bin_idx] += 1.0

            # Normalize
            if pas.sum() > 0:
                pas = pas / pas.sum()

            return pas

        except Exception as e:
            print(f"Warning: PAS calculation failed: {e}")
            return np.zeros(n_bins, dtype=np.float32)

    def calculate_spas(self, mol: Chem.Mol, n_features: int = 600) -> np.ndarray:
        """
        Calculate sPAS (signed Pharmacophore Atom Signatures).

        This is a placeholder implementation. The actual sPAS calculation
        requires the specific algorithm used in the training pipeline.

        Args:
            mol: RDKit Mol object
            n_features: Number of sPAS features

        Returns:
            np.array of shape (n_features,)
        """
        # TODO: Implement actual sPAS calculation
        # For now, return placeholder zeros
        # This needs to be replaced with the actual sPAS algorithm
        return np.zeros(n_features, dtype=np.float32)

    def calculate_3d_descriptors(self, mol3d: Chem.Mol) -> Dict[str, float]:
        """
        Calculate 3D molecular descriptors.

        Args:
            mol3d: RDKit Mol object with 3D coordinates

        Returns:
            Dictionary of descriptor_name: value
        """
        descriptors = {}

        try:
            if mol3d.GetNumConformers() == 0:
                # No 3D coordinates
                return {
                    '3D_PMI1': 0.0,
                    '3D_PMI2': 0.0,
                    '3D_PMI3': 0.0,
                    '3D_NPR1': 0.0,
                    '3D_NPR2': 0.0,
                    '3D_RoG': 0.0,
                    '3D_ISF': 0.0,
                    '3D_Ecc': 0.0,
                    '3D_Asph': 0.0,
                    '3D_Sphero': 0.0,
                }

            # PMI (Principal Moments of Inertia)
            pmi1, pmi2, pmi3 = Descriptors3D.PMI1(mol3d), Descriptors3D.PMI2(mol3d), Descriptors3D.PMI3(mol3d)
            descriptors['3D_PMI1'] = pmi1
            descriptors['3D_PMI2'] = pmi2
            descriptors['3D_PMI3'] = pmi3

            # NPR (Normalized Principal Ratios)
            descriptors['3D_NPR1'] = Descriptors3D.NPR1(mol3d)
            descriptors['3D_NPR2'] = Descriptors3D.NPR2(mol3d)

            # Radius of Gyration
            descriptors['3D_RoG'] = Descriptors3D.RadiusOfGyration(mol3d)

            # Inertial Shape Factor
            descriptors['3D_ISF'] = Descriptors3D.InertialShapeFactor(mol3d)

            # Eccentricity
            descriptors['3D_Ecc'] = Descriptors3D.Eccentricity(mol3d)

            # Asphericity
            descriptors['3D_Asph'] = Descriptors3D.Asphericity(mol3d)

            # Spherocity
            descriptors['3D_Sphero'] = Descriptors3D.SpherocityIndex(mol3d)

        except Exception as e:
            print(f"Warning: 3D descriptors calculation failed: {e}")
            # Return zeros if calculation fails
            descriptors = {
                '3D_PMI1': 0.0,
                '3D_PMI2': 0.0,
                '3D_PMI3': 0.0,
                '3D_NPR1': 0.0,
                '3D_NPR2': 0.0,
                '3D_RoG': 0.0,
                '3D_ISF': 0.0,
                '3D_Ecc': 0.0,
                '3D_Asph': 0.0,
                '3D_Sphero': 0.0,
            }

        return descriptors

    def calculate(self,
                  mol: Chem.Mol,
                  mol3d: Optional[Chem.Mol] = None,
                  solvent_code: int = 1,
                  dihedral: float = 0.0,
                  pm_code: int = 1) -> np.ndarray:
        """
        Calculate all baseline features.

        Args:
            mol: RDKit Mol object (2D)
            mol3d: RDKit Mol object with 3D coordinates (optional)
            solvent_code: Solvent code (0-7)
            dihedral: Dihedral angle
            pm_code: P/M code (0 or 1)

        Returns:
            np.array of shape (2276,) - baseline features
        """
        features = []

        # 1. ECFP (1024 bits)
        ecfp = self.calculate_ecfp(mol)
        features.append(ecfp)

        # 2. RDKit 2D descriptors
        desc_2d = self.calculate_rdkit_2d(mol)
        features.append(np.array(list(desc_2d.values()), dtype=np.float32))

        # 3. PAS (20 features)
        pas = self.calculate_pas(mol, n_bins=20)
        features.append(pas)

        # 4. sPAS (600 features) - TODO: implement actual sPAS
        spas = self.calculate_spas(mol, n_features=600)
        features.append(spas)

        # 5. 3D descriptors (10 features)
        if mol3d is not None:
            desc_3d = self.calculate_3d_descriptors(mol3d)
            features.append(np.array(list(desc_3d.values()), dtype=np.float32))
        else:
            # Use zeros if no 3D structure
            features.append(np.zeros(10, dtype=np.float32))

        # 6. Experimental features (3 features)
        exp_features = np.array([solvent_code, dihedral, pm_code], dtype=np.float32)
        features.append(exp_features)

        # Concatenate all features
        all_features = np.concatenate(features)

        # Pad or trim to exact 2276 dimensions
        if len(all_features) < 2276:
            all_features = np.pad(all_features, (0, 2276 - len(all_features)))
        elif len(all_features) > 2276:
            all_features = all_features[:2276]

        return all_features.astype(np.float32)
