PUBLIC FILE FIELD-LEVEL LICENSE AUDIT
=====================================

⚠️ **CRITICAL FINDINGS**

FILES CONTAIN CANONICAL SMILES
===============================

The following PUBLIC files in the safe candidate contain canonical_smiles:

1. results/primary_oof/oof_predictions_master_labels_v2.parquet
   - Column: canonical_smiles (object)
   - Contains: Full chemical structures (e.g., "CCCN(CCc1cccs1)[C@H]1CCc2c(O)cccc2C1")
   - License Status: ⚠️ LICENSE_REVIEW_REQUIRED

2. results/molecule_disjoint/CANONICAL_SMILES_GROUPED_OOF_PREDICTIONS.csv
   - Column: canonical_smiles (object)
   - Contains: Full chemical structures
   - License Status: ⚠️ LICENSE_REVIEW_REQUIRED

3. results/scaffold_disjoint/SCAFFOLD_SPLIT_ASSIGNMENTS_FINAL.csv (1.6MB)
   - Column: canonical_smiles (object)
   - Contains: Full chemical structures for all 5,204 observations
   - License Status: ⚠️ LICENSE_REVIEW_REQUIRED

FILES CONTAIN SOLVENT METADATA
===============================

4. results/primary_oof/oof_predictions_master_labels_v2.parquet
   - Column: solvent_group (e.g., "methanol")
   - May be Reaxys-derived experimental condition
   - License Status: ⚠️ LICENSE_REVIEW_REQUIRED

5. results/molecule_disjoint/CANONICAL_SMILES_GROUPED_OOF_PREDICTIONS.csv
   - Column: solvent (experimental condition)
   - License Status: ⚠️ LICENSE_REVIEW_REQUIRED

6. results/scaffold_disjoint/SCAFFOLD_SPLIT_ASSIGNMENTS_FINAL.csv
   - Column: solvent_group
   - License Status: ⚠️ LICENSE_REVIEW_REQUIRED

MURCKO SCAFFOLDS
================

7. results/scaffold_disjoint/SCAFFOLD_SPLIT_ASSIGNMENTS_FINAL.csv
   - Column: murcko_scaffold (derived from SMILES)
   - License Status: SAFE_DERIVED_OUTPUT (if SMILES themselves are safe)

SAFE DERIVED OUTPUTS
====================

✅ results/scaffold_disjoint/SCAFFOLD_SPLIT_RESULTS_FINAL.csv
   - Contains: Aggregated metrics only (no structures)
   - License Status: SAFE_DERIVED_OUTPUT

✅ results/scaffold_disjoint/SCAFFOLD_LEAKAGE_AUDIT_FINAL.csv
   - Contains: Validation checks only (no structures)
   - License Status: SAFE_DERIVED_OUTPUT

✅ results/scaffold_disjoint/SCAFFOLD_PREDICTIONS_FINAL.csv
   - Contains: Model predictions only (sample_id, y_true, p_ORplus)
   - NO canonical_smiles column
   - License Status: SAFE_DERIVED_OUTPUT

FIELD-BY-FIELD CLASSIFICATION
==============================

File: oof_predictions_master_labels_v2.parquet
- sample_id: SAFE_DERIVED_OUTPUT
- compound_id: SAFE_DERIVED_OUTPUT (internal ID)
- OR_label: LICENSE_REVIEW_REQUIRED (experimental label)
- canonical_smiles: ⚠️ LICENSE_REVIEW_REQUIRED (CRITICAL)
- solvent_group: ⚠️ LICENSE_REVIEW_REQUIRED
- PM_label: LICENSE_REVIEW_REQUIRED (experimental assignment)
- y_true: LICENSE_REVIEW_REQUIRED (experimental label)
- oof_prob: SAFE_DERIVED_OUTPUT (model output)
- fold: SAFE_DERIVED_OUTPUT

File: CANONICAL_SMILES_GROUPED_OOF_PREDICTIONS.csv
- sample_id: SAFE_DERIVED_OUTPUT
- canonical_smiles: ⚠️ LICENSE_REVIEW_REQUIRED (CRITICAL)
- label: LICENSE_REVIEW_REQUIRED (experimental)
- oof_prediction: SAFE_DERIVED_OUTPUT
- oof_class: SAFE_DERIVED_OUTPUT
- fold: SAFE_DERIVED_OUTPUT
- solvent: ⚠️ LICENSE_REVIEW_REQUIRED

File: SCAFFOLD_SPLIT_ASSIGNMENTS_FINAL.csv (1.6MB - LARGEST FILE)
- sample_id: SAFE_DERIVED_OUTPUT
- row_index: SAFE_DERIVED_OUTPUT
- canonical_smiles: ⚠️ LICENSE_REVIEW_REQUIRED (CRITICAL - ALL 5204 ROWS)
- solvent_group: ⚠️ LICENSE_REVIEW_REQUIRED
- murcko_scaffold: SAFE_DERIVED_OUTPUT (if base data safe)
- split_id: SAFE_DERIVED_OUTPUT
- seed: SAFE_DERIVED_OUTPUT
- partition: SAFE_DERIVED_OUTPUT

CRITICAL ASSESSMENT
===================

❌ **THREE PUBLIC FILES CONTAIN CANONICAL SMILES**

These are complete chemical structures potentially derived from Reaxys.

If Reaxys licensing restricts redistribution of chemical structures, these files MUST be removed or have canonical_smiles columns dropped before public release.

RECOMMENDATION
==============

**Option A - Restrictive Interpretation**:
Remove canonical_smiles columns from all public files.
Repository still reproduces metrics from structure-free predictions.

**Option B - Permissive Interpretation**:
If institutional legal review confirms that:
1. Structures were independently generated from public literature
2. OR: Derived features are redistributable under Reaxys terms
Then keep as-is.

**Option C - Minimum Reproducible**:
Remove all three files with SMILES.
Keep only SCAFFOLD_SPLIT_RESULTS_FINAL.csv and SCAFFOLD_PREDICTIONS_FINAL.csv.
Primary OOF and molecule-disjoint reproduction would be DOCUMENTED_ONLY.

CURRENT RECOMMENDATION
======================

⚠️ **SAFE_ONLY_AFTER_REMOVING_FLAGGED_FILES**

Until institutional legal review confirms redistribution permission:
- Remove or redact canonical_smiles columns
- Consider removing solvent/label columns if Reaxys-derived

This makes the repository "upload-safe" at cost of reduced reproducibility.

ALTERNATIVE
===========

If author confirms redistribution permission has been granted:
- Change status to SAFE_FOR_PRIVATE_GITHUB
- Document permission in data/README.md
