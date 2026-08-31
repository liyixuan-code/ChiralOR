SCRIPT PROVENANCE MAP - ChiralOR Publication
============================================

Based on systematic code search of JCIM_MANUSCRIPT/output_revision/FINAL_RELEASE/05_CODE/

PRIMARY MODEL TRAINING (P0)
===========================

Task: Primary grouped-OOF CatBoost training
Paper Section: Methods, Table 1
Original Script: JCIM_MANUSCRIPT/output_revision/d3full/SERVER_D3FULL_standalone.py
Classification: ORIGINAL_ANALYSIS_SCRIPT (inferred from iter=288 search)
Inputs: X_C3_5204x2283.npy, master_labels_v2.parquet, fold assignments
Outputs: ModelB_fold0-4.cbm, oof_predictions_master_labels_v2.parquet
Config: iter=288, depth=8, lr=0.1, l2=1, seed=42
Publication Values: ROC-AUC 0.9278, AP 0.9005
Verified: ✅ YES (Phase 3)
GitHub: original/primary_catboost_training.py

BENCHMARK MODELS (P0)
====================

Task: 7-model benchmark comparison
Paper Section: Figure 4, Table 2
Original Script: JCIM_MANUSCRIPT/output_revision/FINAL_RELEASE/05_CODE/model_training/fig4_v2_compute.py
Classification: ORIGINAL_ANALYSIS_SCRIPT
Inputs: X_C3_5204x2283.npy, master_labels_v2.parquet, fold assignments
Outputs: 7 model predictions, benchmark metrics
Models: LR, HGB, LightGBM, XGBoost, ET, RF, CatBoost
Config: Fixed (no tuning), same folds, iter=288 for CatBoost
Publication Values: CatBoost 0.9278, RF 0.9260, ET 0.9248, XGBoost 0.9231, LightGBM 0.9209, HGB 0.9172, LR 0.8366
Verified: CatBoost YES, others DOCUMENTED_SOURCE_ONLY
GitHub: original/benchmark_seven_models.py

ABLATION ANALYSIS (P0)
======================

Task: Geometry feature ablation
Paper Section: Figure 4, Results
Original Script: NOT_RECOVERED
Classification: NOT_RECOVERED
Expected Outputs: Full 0.9278, No-geometry 0.9170, Geometry-only 0.8625
Publication Values: ΔAUC +0.0108, CI [0.0065, 0.0149]
Source: Fig4_ablation_metrics_revised.csv, Fig4_ablation_statistics.csv
Status: DOCUMENTED_SOURCE_ONLY
GitHub: NOT_RECOVERED

PAIRED BOOTSTRAP (P0)
=====================

Task: Paired bootstrap CI for ablation
Paper Section: Results
Original Script: NOT_RECOVERED
Classification: NOT_RECOVERED
Expected: 5000 replicates, grouped resampling
Publication Values: CI [0.0065, 0.0149]
Status: DOCUMENTED_SOURCE_ONLY
GitHub: NOT_RECOVERED

MOLECULE-DISJOINT (P0)
======================

Task: Molecule-level grouped validation
Paper Section: Results, SI
Original Script: JCIM_MANUSCRIPT/scripts/task1_canonical_smiles_grouped_oof.py
Classification: ORIGINAL_ANALYSIS_SCRIPT
Inputs: X_C3, master_labels_v2, canonical_smiles grouping
Outputs: CANONICAL_SMILES_GROUPED_OOF_PREDICTIONS.csv
Config: CatBoost iter=288, canonical-SMILES-only grouped folds
Publication Values: ROC-AUC 0.9264, AP 0.9033
Verified: ✅ YES (sanitized version)
GitHub: original/molecule_disjoint_validation.py

SCAFFOLD-DISJOINT (P0)
======================

Task: Scaffold generalization evaluation
Paper Section: Figure 3, Table 3
Original Scripts: 
  - scaffold_split_results/FINAL_RERUN/scaffold_validation_stage2_rerun_server.py
  - scaffold_split_results/FINAL_RERUN/run_scaffold_validation_patch_v3.py
Classification: ORIGINAL_ANALYSIS_SCRIPT
Inputs: X_C3, master_labels_v2, Murcko scaffolds
Outputs: SCAFFOLD_SPLIT_RESULTS_FINAL.csv, SCAFFOLD_LEAKAGE_AUDIT_FINAL.csv
Config: 10 seeds (42, 142, ..., 942), scaffold-disjoint splits
Publication Values: Mean 0.7850 ± 0.0508
Verified: ✅ YES
GitHub: original/scaffold_validation_pipeline.py

CALIBRATION (P0)
================

Task: Model calibration analysis
Paper Section: Results, SI
Original Script: NOT_RECOVERED
Classification: NOT_RECOVERED
Expected: 10-bin equal-frequency ECE, calibration slope/intercept
Publication Values: Brier 0.1027, ECE 0.0215, slope 0.893, intercept 0.067
Status: Brier reproduced, ECE/slope/intercept DOCUMENTED_SOURCE_ONLY
GitHub: reproduction/calibration_analysis.py (RECONSTRUCTED)

OOF TREESHAP (P0)
=================

Task: Global OOF TreeSHAP analysis
Paper Section: Figure 5, Figure 6
Original Script: JCIM_MANUSCRIPT/output_revision/FINAL_RELEASE/05_CODE/shap/_gate9_oof_shap.py
Classification: ORIGINAL_ANALYSIS_SCRIPT
Inputs: ModelB_fold0-4.cbm, X_C3, fold assignments
Outputs: oof_shap_global_importance.csv, SHAP arrays
Config: TreeExplainer, OOF (each obs by held-out model), n=5204
Publication Values: Signed tetra volume mean|SHAP| ≈ 0.406
Status: Script found, arrays NOT_RECOVERED
GitHub: original/oof_treeshap_global.py

SHAP INTERACTION (P0)
=====================

Task: SHAP interaction stability
Paper Section: Figure 7, SI
Original Scripts:
  - JCIM_MANUSCRIPT/output_revision/FINAL_RELEASE/05_CODE/interaction/_gate9_interaction.py
  - JCIM_MANUSCRIPT/output_revision/FINAL_RELEASE/05_CODE/interaction/SERVER_compute_oof_interaction_v2.py
Classification: ORIGINAL_ANALYSIS_SCRIPT
Expected: Historical n=128, 5 sensitivity subsets (seeds 101-505)
Publication Values: Mean Spearman 0.852, MACCS-129 × dihedral rank 1
Status: Scripts found, artifacts PARTIAL
GitHub: original/shap_interaction_analysis.py

CONDITIONAL SURFACES (P1)
=========================

Task: Conditional OR+ response surfaces
Paper Section: Figure 8
Original Script: JCIM_MANUSCRIPT/output_revision/FINAL_RELEASE/05_CODE/model_training/fig13_step2_surfaces.py
Classification: ORIGINAL_ANALYSIS_SCRIPT
Inputs: 6 feature pairs, 60×60 grid, 5 fold models
Config: iter=288
Status: Script found
GitHub: original/conditional_response_surfaces.py

FIGURE GENERATION (P1)
======================

Task: Final figure rendering
Paper Section: All figures
Original Scripts: JCIM_MANUSCRIPT/output_revision/FINAL_RELEASE/05_CODE/figure_generation/*.py
Classification: ORIGINAL_ANALYSIS_SCRIPT (rendering only)
Examples:
  - fig4_v2_render.py
  - fig13_step3_render.py
Status: Multiple figure scripts found
GitHub: figures/

EXCLUDED SCRIPTS
================

HISTORICAL_CONFLICT (iter=1200):
- All scripts in JCIM_MANUSCRIPT with iterations=1200
- Produce wrong ablation AUC ≈ 0.9526
- Wrong scaffold results
- Classification: HISTORICAL_CONFLICT
- Action: EXCLUDE from GitHub

SCHEMA_B_DEPLOYMENT:
- CHIRALOR_WEB_MODEL_2283P_REMOTE_PACKAGE/*
- CHIRALOR_PROSPECTIVE_MODEL/*
- train_final_model.py
- Classification: DEPLOYMENT_SCHEMA_B
- Action: EXCLUDE from publication pipeline

DEVELOPMENT_INTERMEDIATES:
- JCIM_MANUSCRIPT/output_revision/ARCHIVE/development_intermediates/*
- Classification: DEVELOPMENT_ONLY
- Action: EXCLUDE

RECONSTRUCTED SCRIPTS (Structure-Free)
=======================================

Created for structure-free public release:

1. scripts/01_verify_primary_oof_metrics.py
   Type: RECONSTRUCTED_REPRODUCTION_SCRIPT
   Purpose: Verify 11 primary metrics from sanitized predictions
   Status: ✅ Tested

2. scripts/02_verify_molecule_disjoint_metrics.py
   Type: RECONSTRUCTED_REPRODUCTION_SCRIPT
   Purpose: Verify molecule-disjoint from sanitized predictions
   Status: ✅ Tested

3. scripts/03_verify_scaffold_disjoint_results.py
   Type: RECONSTRUCTED_REPRODUCTION_SCRIPT
   Purpose: Verify scaffold results and leakage
   Status: ✅ Tested

MISSING/NOT_RECOVERED
=====================

Critical items where original scripts NOT recovered:

1. Ablation training scripts (no-geometry, geometry-only)
   - Status: NOT_RECOVERED
   - Evidence: Fig4_ablation_metrics_revised.csv
   
2. Paired bootstrap script
   - Status: NOT_RECOVERED
   - Evidence: Fig4_ablation_statistics.csv

3. Calibration analysis script
   - Status: NOT_RECOVERED
   - Evidence: ECE/slope/intercept values in SI

4. Schema A feature generation complete pipeline
   - Status: PARTIAL (web deployment scripts exist but Schema B)
   - Need: Original Schema A feature generation

5. Primary fold generation script
   - Status: NOT_RECOVERED
   - Inferred: StratifiedGroupKFold, n=5, seed=42

6. SuperLearner pipeline
   - Status: NOT_RECOVERED
   - Evidence: ROC-AUC 0.9330 in SI

7. Structural rule R01-R41 generation
   - Status: NOT_RECOVERED

8. Force plot generation (10 OOF exemplars)
   - Status: NOT_RECOVERED

---

Summary: 8 ORIGINAL scripts found, 3 RECONSTRUCTED created, 8 NOT_RECOVERED
