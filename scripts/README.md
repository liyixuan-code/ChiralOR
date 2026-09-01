# Scripts

The release separates recovered publication code, compact public verification code, and retained non-final development code.

## Public verification scripts

Run from the repository root:

```bash
python scripts/reproduction/01_verify_primary_oof_metrics.py
python scripts/reproduction/02_verify_molecule_disjoint_metrics.py
python scripts/reproduction/03_verify_scaffold_disjoint_results.py
```

These operate only on structure-free frozen public artifacts.

`04_train_publication_schemaA_grouped_cv.py` is a reconstructed reference implementation of the recovered grouped-CV logic. It requires the private 5,204 × 2,283 matrix and private grouping metadata and is not the first historical root-model persistence script.

## Recovered original publication code

`original/` contains recovered code for curation/QC, P/M annotation, structural-domain rules, PAS/sPAS, the 2,276-feature baseline assembly, the seven direction-sensitive geometry variables, grouped-CV/benchmark/ablation logic, calibration, SHAP, interaction analysis, molecule/scaffold-disjoint validation, conditional surfaces, and figure workflows. Historical local paths are intentionally retained in original scripts where relevant; these scripts may therefore require path adaptation and private inputs for execution.

## Historical non-final code

`historical/nonfinal/` contains retained development code that is **not** part of the final Publication Schema A implementation. In particular, `baseline_features.py` uses a non-final fingerprint/placeholder feature definition and must not be used as the publication feature generator.

See `../reproducibility/SCRIPT_PROVENANCE_MAP.csv` and `../reproducibility/PUBLICATION_PIPELINE_MAP.csv`.
