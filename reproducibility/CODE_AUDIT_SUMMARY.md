CODE AUDIT FINAL SUMMARY
========================
Generated: 2026-08-31

A. ORIGINAL PUBLICATION SCRIPTS FOUND
======================================

✅ **8 ORIGINAL_ANALYSIS_SCRIPT identified**:

1. **Primary CatBoost Training**
   - Path: JCIM_MANUSCRIPT/output_revision/d3full/SERVER_D3FULL_standalone.py
   - Purpose: Train 5-fold grouped-OOF CatBoost models
   - Config: iter=288, depth=8, lr=0.1, l2=1, seed=42
   - Outputs: ModelB_fold0-4.cbm, oof_predictions_master_labels_v2.parquet
   - Result: ROC-AUC 0.9278
   - Status: VERIFIED

2. **Benchmark Models (7 models)**
   - Path: JCIM_MANUSCRIPT/output_revision/FINAL_RELEASE/05_CODE/model_training/fig4_v2_compute.py
   - Purpose: Train 7 standard models (LR, HGB, LightGBM, XGBoost, ET, RF, CatBoost)
   - Result: All benchmark AUCs (0.8366-0.9278)
   - Status: CatBoost VERIFIED, others DOCUMENTED

3. **Molecule-Disjoint Validation**
   - Path: JCIM_MANUSCRIPT/scripts/task1_canonical_smiles_grouped_oof.py
   - Purpose: Canonical-SMILES-level grouped OOF
   - Result: ROC-AUC 0.9264
   - Status: VERIFIED

4. **Scaffold-Disjoint Validation**
   - Paths: scaffold_validation_stage2_rerun_server.py, run_scaffold_validation_patch_v3.py
   - Purpose: 10-seed scaffold generalization
   - Result: Mean 0.7850 ± 0.0508
   - Status: VERIFIED

5. **OOF TreeSHAP**
   - Path: JCIM_MANUSCRIPT/output_revision/FINAL_RELEASE/05_CODE/shap/_gate9_oof_shap.py
   - Purpose: Global OOF TreeSHAP (n=5204)
   - Result: Signed tetra volume mean|SHAP| ≈ 0.406
   - Status: Script found, arrays PARTIAL

6. **SHAP Interaction**
   - Path: JCIM_MANUSCRIPT/output_revision/FINAL_RELEASE/05_CODE/interaction/SERVER_compute_oof_interaction_v2.py
   - Purpose: Interaction stability analysis
   - Result: Mean Spearman 0.852
   - Status: Script found, artifacts PARTIAL

7. **Conditional Surfaces**
   - Path: JCIM_MANUSCRIPT/output_revision/FINAL_RELEASE/05_CODE/model_training/fig13_step2_surfaces.py
   - Purpose: 6 feature-pair conditional response surfaces
   - Config: 60×60 grid, 5-fold ensemble
   - Status: Script found

8. **Figure Generation Scripts**
   - Paths: JCIM_MANUSCRIPT/output_revision/FINAL_RELEASE/05_CODE/figure_generation/*.py
   - Purpose: Final manuscript/SI figure rendering
   - Multiple scripts: fig4_v2_render.py, fig13*.py, etc.
   - Status: Scripts found

B. RECONSTRUCTED REPRODUCTION SCRIPTS
======================================

✅ **3 RECONSTRUCTED_REPRODUCTION_SCRIPT created** (structure-free):

1. scripts/01_verify_primary_oof_metrics.py
   - Purpose: Verify 11 primary metrics from sanitized predictions
   - Status: ✅ Tested, all metrics match

2. scripts/02_verify_molecule_disjoint_metrics.py
   - Purpose: Verify molecule-disjoint from sanitized predictions
   - Status: ✅ Tested, all metrics match

3. scripts/03_verify_scaffold_disjoint_results.py
   - Purpose: Verify scaffold results and zero leakage
   - Status: ✅ Tested, verified

C. IMPORTANT SCRIPTS NOT_RECOVERED
===================================

❌ **8 critical items NOT_RECOVERED**:

1. **Ablation Training Scripts**
   - No-geometry model (AUC 0.9170)
   - Geometry-only model (AUC 0.8625)
   - Evidence: Fig4_ablation_metrics_revised.csv
   - Status: DOCUMENTED_SOURCE_ONLY

2. **Paired Bootstrap Script**
   - 5000 replicates, grouped resampling
   - Evidence: Fig4_ablation_statistics.csv (CI [0.0065, 0.0149])
   - Status: DOCUMENTED_SOURCE_ONLY

3. **Calibration Analysis Script**
   - ECE, calibration slope/intercept
   - Evidence: Values in final SI
   - Status: Brier reproduced, ECE/slope/intercept DOCUMENTED

4. **Schema A Feature Generation Pipeline**
   - 2283-feature complete pipeline
   - Status: PARTIAL (web deployment has Schema B version)

5. **Primary Fold Generation Script**
   - StratifiedGroupKFold, n=5, seed=42
   - Status: NOT_RECOVERED (embedded in training scripts)

6. **SuperLearner Pipeline**
   - 6 base learners, meta learner
   - Evidence: ROC-AUC 0.9330
   - Status: NOT_RECOVERED

7. **Structural Rules R01-R41**
   - Mutually exclusive structural domain assignment
   - Status: NOT_RECOVERED

8. **Force Plots (10 exemplars)**
   - OOF force plot generation
   - Status: NOT_RECOVERED

D. HISTORICAL / WRONG SCRIPTS EXCLUDED
=======================================

❌ **HISTORICAL_CONFLICT (iter=1200)**:

1. JCIM_MANUSCRIPT/_b3b2_compute.py
   - Config: iter=1200 (wrong)
   - Produces: Full AUC ≈ 0.9526 (wrong)
   - Classification: HISTORICAL_CONFLICT
   - Action: EXCLUDE

❌ **SCHEMA_B_DEPLOYMENT**:

2. CHIRALOR_WEB_MODEL_2283P_REMOTE_PACKAGE/*
   - All deployment scripts
   - Classification: DEPLOYMENT_SCHEMA_B
   - Action: EXCLUDE from publication pipeline

❌ **DEVELOPMENT_INTERMEDIATES**:

3. JCIM_MANUSCRIPT/output_revision/ARCHIVE/development_intermediates/*
   - All archived development scripts
   - Classification: DEVELOPMENT_ONLY
   - Action: EXCLUDE

E-N. DETAILED STATUS BY COMPONENT
==================================

E. PRIMARY MODEL TRAINING: ✅ FOUND
F. SCHEMA A FEATURE GENERATION: ⚠️ PARTIAL (Schema B exists)
G. BENCHMARK CODE: ✅ FOUND
H. ABLATION / BOOTSTRAP: ❌ NOT_RECOVERED
I. MOLECULE-DISJOINT: ✅ FOUND
J. SCAFFOLD: ✅ FOUND
K. CALIBRATION: ❌ NOT_RECOVERED
L. SHAP: ✅ FOUND (arrays PARTIAL)
M. INTERACTION: ✅ FOUND (artifacts PARTIAL)
N. FIGURE CODE: ✅ FOUND

O. GITHUB scripts/ DIRECTORY TREE
==================================

CHIRALOR_PUBLICATION_CODE_CANDIDATE/scripts/
├── README.md
├── original/
│   ├── primary_catboost_training.py (SERVER_D3FULL_standalone.py)
│   ├── benchmark_seven_models.py (fig4_v2_compute.py)
│   ├── molecule_disjoint_validation.py (task1_canonical_smiles_grouped_oof.py)
│   ├── scaffold_validation_pipeline.py (scaffold_validation_stage2_rerun_server.py)
│   ├── oof_treeshap_global.py (_gate9_oof_shap.py)
│   ├── shap_interaction_analysis.py (SERVER_compute_oof_interaction_v2.py)
│   └── conditional_response_surfaces.py (fig13_step2_surfaces.py)
├── reproduction/
│   ├── 01_verify_primary_oof_metrics.py (RECONSTRUCTED)
│   ├── 02_verify_molecule_disjoint_metrics.py (RECONSTRUCTED)
│   └── 03_verify_scaffold_disjoint_results.py (RECONSTRUCTED)
└── figures/
    ├── fig4_v2_render.py
    ├── fig13_step3_render.py
    └── [other figure scripts]

P. SCRIPT_PROVENANCE_MAP.csv PATH
==================================

code_audit/SCRIPT_PROVENANCE_MAP.md (markdown format for readability)

Q. CODE CANDIDATE PATH
=======================

CHIRALOR_PUBLICATION_CODE_CANDIDATE/scripts/

R. MISSING CODE ITEMS
======================

Critical NOT_RECOVERED:
1. Ablation training (no-geometry, geometry-only)
2. Paired bootstrap (5000 replicates)
3. Calibration analysis (ECE/slope/intercept)
4. Schema A feature generation (complete pipeline)
5. Primary fold generation (standalone)
6. SuperLearner (complete pipeline)
7. Structural rules R01-R41 generation
8. Force plots (10 OOF exemplars)

Impact: Core validation results fully reproducible from frozen artifacts. Extended analyses documented with methodology.

RECOMMENDATION
==============

**Proceed with GitHub scripts/ assembly**:
- Copy 7 ORIGINAL scripts to original/
- Keep 3 RECONSTRUCTED scripts in reproduction/
- Copy figure scripts to figures/
- Document NOT_RECOVERED items in scripts/README.md
- All scripts labeled with provenance type

**NOT_RECOVERED acceptable** given:
- Frozen artifacts enable verification
- Methodologies documented
- Paper values from trusted final SI
- Transparency over false completeness

---

**Code Audit Status**: ✅ COMPLETE
**Candidate Directory**: READY FOR ASSEMBLY
**Original Scripts Found**: 8
**Reconstructed Scripts**: 3
**NOT_RECOVERED**: 8 (acceptable, documented)
