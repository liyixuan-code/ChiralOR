# ChiralOR v1.1.0

**ChiralOR: An Interpretable Chirality-Aware Machine Learning Model for Optical-Rotation Sign Prediction**

ChiralOR is the publication code and structure-free reproducibility package for optical-rotation sign prediction in the curated 5,204-observation study dataset.

- Repository: https://github.com/liyixuan-code/ChiralOR
- Existing Zenodo v1.0.0 archive: 10.5281/zenodo.22210778
- Zenodo concept DOI (all versions): 10.5281/zenodo.22210777

Version **v1.1.0** expands the release with recovered publication-workflow code, figure workflows, public-safe source tables, provenance maps, and the five frozen Publication Schema A CatBoost fold models. A new version-specific Zenodo DOI can be added after the v1.1.0 Zenodo deposit is minted.

## Quick verification

From the repository root:

```bash
python scripts/reproduction/01_verify_primary_oof_metrics.py
python scripts/reproduction/02_verify_molecule_disjoint_metrics.py
python scripts/reproduction/03_verify_scaffold_disjoint_results.py
```

These scripts verify the reported primary grouped-OOF, molecule-disjoint, and repeated scaffold-disjoint results from public-safe frozen output artifacts.

## Publication Schema A

The ordered schema contains **2,283 features**:

- ECFP6: 2,048
- MACCS: 167
- RDKit 2D descriptors: 8
- signed PAS (sPAS): 20
- PAS: 20
- 3D shape descriptors: 10
- solvent code: 1
- baseline P/M dihedral: 1
- P/M code: 1
- direction-sensitive geometry: 7

The 2,276-feature baseline assembly is represented by `scripts/original/feature_generation/rebuild_features_v2.py`; the seven direction-sensitive variables are implemented in `scripts/original/feature_generation/block2_signed_geometry.py` with `ring_neighbor_shared.py`. The exact frozen feature order is in `data/features_schema_A/feature_schema.csv`.

The retained historical `baseline_features.py` is stored under `scripts/historical/nonfinal/` because it is not the final Publication Schema A implementation.

## Repository contents

- `models/publication_schema_A/` — five frozen publication fold models.
- `scripts/original/` — recovered original publication scripts and support modules.
- `scripts/reproduction/` — compact public verification/reference scripts. The grouped-CV training reference requires private inputs and is not needed to verify the frozen publication results.
- `scripts/historical/nonfinal/` — retained non-final development code, explicitly separated from the publication workflow.
- `data/features_schema_A/` — exact ordered 2,283-feature schema.
- `data/splits/` — structure-free public fold/split assignments.
- `results/` — public-safe predictions, metrics, validation summaries, SHAP summaries, interaction summaries, and structural-domain tables.
- `figure_source_data/` — machine-readable, public-safe source tables for main and Supporting Information figures.
- `reproducibility/` — publication pipeline, figure lineage, script provenance, model-binary identity, limitations, and audit summaries.
- `deployment/README_SCHEMA_B.md` — distinction between Publication Schema A and deployment-specific Schema B.

## Structural domains R01–R41

The recovered stereochemistry-invariant classification workflow is represented by `scripts/original/structural_domains/generate_stereo_invariant_rules.py` and the shared ring-neighbor logic. The final public-safe 5,204-observation class table is `results/structural_domains/structural_rule_table_final_5204.csv` (also provided in `figure_source_data/supporting_information/structural_rule_table.csv`). It contains 41 mutually exclusive classes and reproduces the final SI class counts and empirical directional consistencies.

OR sign is **not** used to define the structural classes; OR counts and consistency are descriptive summaries computed after class assignment.

## Data and licensing restrictions

This is intentionally a **structure-free** public release. It does **not** redistribute:

- raw Reaxys exports or Reaxys identifiers;
- canonical-SMILES observation tables or other molecular-structure tables;
- private identifier-to-structure mappings;
- Bemis–Murcko structure strings;
- the private 5,204 × 2,283 feature matrix; or
- other third-party licensed source-record fields.

No separate open-source software license is granted by this repository. The files are released for publication transparency and reproducibility; reuse and redistribution remain subject to copyright and third-party restrictions.

Because licensed/private inputs are excluded, the public package verifies the frozen publication outputs and exposes the recovered computational logic, but it is not a complete public raw-record-to-feature-matrix rebuild.

## Historical provenance notes

The curation workflow was multi-stage and the final 5,204-record artifact and all publication counts were verified. The exact wrapper used for the initial source-file import and the final serialization step was not uniquely identified. Likewise, the exact first historical persistence entry point for the root `ModelB_fold*.cbm` files was not uniquely identified; however, the historical, Figure-13-regenerated, and public fold models were verified as byte-identical for all five folds.

See `reproducibility/PUBLICATION_PIPELINE_MAP.csv`, `reproducibility/SCRIPT_PROVENANCE_MAP.csv`, and `reproducibility/KNOWN_LIMITATIONS.md` for details.

## Citation

For the repository, cite Yixuan Li and the ChiralOR repository. For a frozen archival snapshot, use the version-specific Zenodo DOI corresponding to the release used. The existing v1.0.0 snapshot is DOI **10.5281/zenodo.22210778**; the concept DOI is **10.5281/zenodo.22210777**.
