# Supplementary code and results — ChiralOR optical-rotation sign prediction

This directory accompanies the manuscript on chirality-aware signed-geometry
descriptors for optical-rotation sign classification. It contains the scripts
and result files for every analysis reported in the revised manuscript, including
the feature-reproduction audit, the paired out-of-fold comparison, the
same-dataset representation baselines, and the conformer and missing-value
sensitivity analyses.

## Contents

```
code/    analysis scripts (Python 3.9+; see dependencies below)
data/    result files (JSON/CSV) produced by those scripts
docs/    additional notes
```

### code/

| File | Purpose |
|---|---|
| `00_verify_feature_generator.py` | Regenerates features for a training sample and compares them block by block with the frozen matrix; writes `FEATURE_GATE.json`. Supports `--strict`, `--relaxed`, `--tolerant`. |
| `02_generate_features.py` | Builds the 2,283-dimensional vector for external molecules. |
| `03_predict_catboost.py` | Loads the five frozen fold models (SHA-256 verified) and writes per-fold and ensemble probabilities. |
| `04_evaluate_external.py` | External evaluation with the manuscript's metric set and group-cluster bootstrap. |
| `05_stratified_analysis.py` | Domain-distance, per-solvent and per-fold stratification. |
| `06_paired_oof_reconstruction.py` | Rebuilds the full training matrix and scores it with the same held-out fold models (the paired out-of-fold experiment). |
| `07_external_ad_validation.py` | Applicability-domain–restricted external metrics. |
| `08_same_dataset_baselines.py` | PAS, PAS+sPAS, stereocentre-centred (Mamede-style), baseline and full representations on the published folds, with random forest and CatBoost. |
| `10_conformer_robustness.py` | Three embedding seeds; frozen-model and retrained sensitivity. |
| `11_multiconformer_robustness.py` | Six-conformer ensembles on a stratified 500-molecule sample; conformer mean and lowest-energy aggregation. |
| `probe_ecfp_version.py` | ECFP6 reproduction across RDKit versions (the version audit). |
| `probe_hacceptor_definition.py` | Tests candidate H-bond-acceptor definitions against the frozen column. |
| `test_drift_impact.py` | Feature-block importance and perturbation bounds. |
| `common.py`, `feature_builder.py` | Shared constants, paths and vector construction. |

### data/

Both JSON and CSV files carry the numbers quoted in the manuscript and SI.
Each JSON records its own `protocol`, `rdkit_version` and interpreter path in
the top-level fields; **reproducing results requires the stated RDKit version**,
since ECFP6 bits depend on it (see `docs/`).

| File | Content |
|---|---|
| `FEATURE_GATE.json` | Per-block reproduction status with the generating environment. |
| `ecfp6_probe_2022.09.5.json`, `ecfp6_probe_2023.03.1.json`, `ecfp6_probe_2025.03.6.json` | ECFP6 exactness for every training record under each RDKit build. |
| `paired_oof_reconstruction.json` | Frozen versus reconstructed matrix under the fold-own-model protocol. |
| `geometry_block_decomposition.json` | Five conformer-derived descriptors, two dihedral terms, P/M code, and subset ladder. |
| `missingness_experiment.json` | Zero-fill, median imputation, indicator variables, column deletion. |
| `conformer_robustness.json` | Three-seed stability, frozen and retrained. |
| `multiconformer_robustness.json` | Six-conformer within-molecule variances and AUC. |
| `same_dataset_baselines.json`, `.csv`, `same_dataset_mamede_catboost_oof.csv` | Published-fold representation comparison. |
| `same_dataset_ohecc.json` | Cartesian-coordinate representation (Zhou et al.) reconstruction and its atom-ordering sensitivity. |
| `label_magnitude_stratification.csv`, `label_error_rate_by_magnitude.csv` | Performance and error rate by rotation magnitude. |
| `external_metrics.json`, `external_ad_validation.json`, `external_stratified_metrics.json`, `external_predictions_705.csv` | Final external validation on the CORRECTED_V2_FINAL 705-record primary set (691 complete vectors, 441 molecules). |
| `derived_dihedral_accuracy.json` | Calculator benchmark against curated dihedral and P/M values (26/60 and 40/60 on 60 training molecules). |
| `drift_impact_2022.09.5.json` | Block importance and perturbation effect sizes. |

## Environment

- Python 3.9 and 3.13 were both used; each script states what it needs.
- RDKit 2025.03.6 for fingerprints and 2D descriptors; RDKit 2022.09.5 for
  ETKDGv3 conformer embedding. Results differ between the two for the
  conformer-derived and fingerprint blocks; the split is documented in the
  manuscript Methods.
- CatBoost 1.2, scikit-learn 1.3–1.6, SHAP 0.42.1, NumPy 1.26, pandas 2.2.

## Data not included

Raw Reaxys exports are not redistributed because the source database is
licensed. The processed dataset, the frozen 2,283-dimensional matrix, the fold
assignments and the five fold models accompany the manuscript at the main
repository.
