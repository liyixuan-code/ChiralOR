# ChiralOR: Machine Learning for Optical Rotation Sign Prediction

[![DOI](https://img.shields.io/badge/DOI-pending-blue.svg)](https://doi.org/pending)

## Overview

ChiralOR predicts the sign of optical rotation (OR+ vs OR−) in chiral fused-ring molecules from molecular structure and solvent conditions using machine learning.

**Publication**: [Citation to be added after publication]

**Dataset**: 5,204 observations (3,141 OR−, 2,063 OR+)

**Model**: CatBoost classifier with 2,283-dimensional feature representation

**Performance** (Structure-solvent grouped 5-fold OOF):
- ROC-AUC: 0.928
- Average Precision: 0.901
- Scaffold-disjoint mean AUC: 0.785 ± 0.051

## Important: Publication Schema A vs Deployment Schema B

⚠️ **All performance estimates in the publication apply exclusively to Publication Schema A.**

The web deployment at chiralor.cn uses a distinct feature schema (Schema B) optimized for prospective predictions. **Do not assign publication performance metrics to Schema B.**

See [deployment/README_SCHEMA_B.md](deployment/README_SCHEMA_B.md) for details.

## Publication Schema A

**2,283 features**:
- 2,048 ECFP6 bits
- 167 MACCS keys
- 8 core RDKit 2D descriptors
- 20 signed PAS (sPAS)
- 20 unsigned PAS
- 10 3D shape descriptors
- 1 categorical solvent code
- 1 baseline P/M dihedral
- 1 P/M binary code
- **7 direction-sensitive geometry variables**

## Installation

```bash
# Clone repository
git clone https://github.com/[REPO_URL]
cd ChiralOR

# Create environment
conda env create -f environment.yml
conda activate chiralor

# Or using pip
pip install -r requirements.txt
```

**Requirements**:
- Python 3.11.9
- NumPy 2.4.6
- pandas 3.0.5
- scikit-learn 1.9.0
- CatBoost 1.2.10
- RDKit 2026.03.5

See [reproducibility/SOFTWARE_ENVIRONMENT.md](reproducibility/SOFTWARE_ENVIRONMENT.md) for complete environment.

## Repository Structure

```
ChiralOR/
├── data/                           # [LICENSE REVIEW REQUIRED]
│   ├── features_schema_A/         # 2,283-dimensional feature matrix
│   └── splits/                    # Fold assignments
├── models/
│   └── publication_schema_A/      # 5 CatBoost fold models
├── results/
│   ├── primary_oof/               # Grouped OOF predictions
│   ├── molecule_disjoint/         # Molecule-level validation
│   └── scaffold_disjoint/         # Scaffold generalization
├── scripts/                        # Reproduction scripts
├── reproducibility/                # Provenance documentation
└── deployment/                     # Schema B documentation
```

## Reproducibility

**Tier 1 - Fully Reproducible** (from frozen predictions/artifacts):
- ✅ Primary grouped-OOF (11 metrics)
- ✅ Molecule-disjoint sensitivity
- ✅ Scaffold-disjoint evaluation (10 splits)

**Tier 2 - Documented from Final SI**:
- ⚠️ Ablation (full model verified, no-signed/geometry-only documented)
- ⚠️ Benchmark (CatBoost verified, 6 others documented)
- ⚠️ SHAP/Interaction (methodology documented)

See [reproducibility/FINAL_REPRODUCIBILITY_AUDIT.md](reproducibility/FINAL_REPRODUCIBILITY_AUDIT.md) and [reproducibility/KNOWN_LIMITATIONS.md](reproducibility/KNOWN_LIMITATIONS.md).

## Primary Validation

Reproduce primary grouped-OOF metrics from frozen predictions:

```bash
python scripts/verify_primary_oof.py \
  --predictions results/primary_oof/oof_predictions_master_labels_v2.parquet
```

**Expected output**: ROC-AUC 0.9278, AP 0.9005

## Molecule-Disjoint Validation

```bash
python scripts/verify_molecule_disjoint.py \
  --predictions results/molecule_disjoint/canonical_smiles_grouped_oof.csv
```

**Expected output**: ROC-AUC 0.9264

## Scaffold-Disjoint Evaluation

```bash
python scripts/verify_scaffold_disjoint.py \
  --results results/scaffold_disjoint/SCAFFOLD_SPLIT_RESULTS_FINAL.csv
```

**Expected output**: Mean AUC 0.7850 ± 0.0508 (10 splits)

## Seven Signed Geometry Features

The 7 direction-sensitive geometry variables (indices 2276-2282 in Schema A):

1. signed_tetra_volume
2. subst_to_ringplane_signed_dist
3. ringnormal_dot_substvec
4. signed_dihedral_subst_c_rn1_rn2
5. signed_dihedral_NS_path
6. baseline_pm_dihedral_sin
7. baseline_pm_dihedral_cos

**Incremental contribution**: ΔAUC +0.011 (bootstrap 95% CI +0.007 to +0.015)

## Data Availability

⚠️ **[LICENSE REVIEW REQUIRED]**

Chemical structures and labels are derived from literature data accessed via Reaxys (Elsevier). Due to licensing restrictions:

- ❌ Raw Reaxys exports: Not redistributable
- ⚠️ Processed data: Pending institution legal review

This repository provides:
- ✅ Derived 2,283-dimensional feature matrix (if approved)
- ✅ Fold assignments
- ✅ Model predictions
- ✅ Scaffold split assignments
- ✅ Aggregated statistics

For data access inquiries, contact [AUTHOR EMAIL].

## Citation

```bibtex
@article{chiralor2026,
  title={[TITLE]},
  author={[AUTHORS]},
  journal={Journal of Chemical Information and Modeling},
  year={2026},
  doi={[DOI]}
}
```

## License

**Software**: [TO BE DETERMINED BY AUTHOR]

**Data**: Subject to Reaxys licensing terms. See REAXYS_DATA_RELEASE_POLICY.md.

## Acknowledgments

Chemical structures and optical rotation data were derived from literature accessed via Reaxys (Elsevier).

## Contact

[AUTHOR CONTACT]

---

**Repository Version**: 1.0.0-candidate  
**Last Updated**: 2026-08-31  
**Status**: Pending author review and license decisions
