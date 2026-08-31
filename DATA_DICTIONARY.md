DATA DICTIONARY - ChiralOR Public Release
=========================================

This dictionary documents all fields in the structure-free public release.

PRIMARY GROUPED-OOF PREDICTIONS
================================

File: results/primary_oof/primary_grouped_oof_predictions.csv

observation_id (int)
  Description: Anonymous stable identifier for each observation
  Units: N/A
  Source: Original dataset
  Transformation: None (opaque ID)
  License Status: SAFE_DERIVED_OUTPUT

fold (int)
  Description: Cross-validation fold assignment (0-4)
  Units: N/A
  Source: Structure-solvent grouped stratified split
  Transformation: Grouped by canonical_smiles × solvent (structure withheld)
  License Status: SAFE_DERIVED_OUTPUT

observed_label (int)
  Description: Binary experimental optical rotation class (0=OR−, 1=OR+)
  Units: Binary
  Source: Curated from literature via Reaxys
  Transformation: None
  License Status: LICENSE_REVIEW_REQUIRED (experimental label, included for reproducibility)

oof_probability_or_plus (float)
  Description: Out-of-fold predicted probability of OR+ class
  Units: Probability [0,1]
  Source: CatBoost model output (held-out fold)
  Transformation: Model prediction
  License Status: SAFE_DERIVED_OUTPUT

predicted_label_t0_5 (int)
  Description: Predicted class using threshold t=0.5
  Units: Binary
  Source: Derived from oof_probability_or_plus
  Transformation: (oof_probability_or_plus >= 0.5)
  License Status: SAFE_DERIVED_OUTPUT

MOLECULE-DISJOINT PREDICTIONS
==============================

File: results/molecule_disjoint/molecule_disjoint_oof_predictions.csv

observation_id (int)
  Description: Same as primary predictions
  License Status: SAFE_DERIVED_OUTPUT

molecule_group_id (string)
  Description: Anonymous identifier preserving molecular-identity equivalence
  Format: MOL_XXXXXX
  Units: N/A
  Source: Derived from canonical_smiles (structure withheld)
  Transformation: Opaque stable mapping (structure NOT recoverable from ID)
  License Status: SAFE_DERIVED_OUTPUT
  Note: "Mapping to molecular structure withheld due to source-data redistribution restrictions. ID preserves equivalence: same molecule_group_id = same chemical structure."

fold (int)
  Description: Molecule-level grouped fold (entire molecule in one fold)
  License Status: SAFE_DERIVED_OUTPUT

observed_label (int)
  Description: Same as primary
  License Status: LICENSE_REVIEW_REQUIRED

oof_probability_or_plus (float)
  Description: Same as primary
  License Status: SAFE_DERIVED_OUTPUT

predicted_label_t0_5 (int)
  Description: Same as primary
  License Status: SAFE_DERIVED_OUTPUT

MOLECULE-DISJOINT GROUP AUDIT
==============================

File: results/molecule_disjoint/molecule_disjoint_group_audit.csv

molecule_group_id (string)
  Description: Same as above
  License Status: SAFE_DERIVED_OUTPUT

n_observations (int)
  Description: Number of observations for this molecule
  Units: Count
  License Status: SAFE_DERIVED_OUTPUT

fold (int/string)
  Description: Fold assignment (or 'MULTIPLE' if constraint violated)
  License Status: SAFE_DERIVED_OUTPUT

n_unique_folds (int)
  Description: Number of unique folds containing this molecule (expected: 1)
  Units: Count
  License Status: SAFE_DERIVED_OUTPUT

SCAFFOLD SPLIT ASSIGNMENTS
===========================

File: results/scaffold_disjoint/scaffold_split_assignments_public.csv

split (int)
  Description: Split number (1-10)
  Units: N/A
  License Status: SAFE_DERIVED_OUTPUT

seed (int)
  Description: Random seed (42, 142, 242, ..., 942)
  Units: N/A
  License Status: SAFE_DERIVED_OUTPUT

observation_id (int)
  Description: Same as primary
  License Status: SAFE_DERIVED_OUTPUT

molecule_group_id (string)
  Description: Same as molecule-disjoint
  License Status: SAFE_DERIVED_OUTPUT

scaffold_group_id (string)
  Description: Anonymous identifier preserving Bemis-Murcko scaffold equivalence
  Format: SCF_XXXXXX
  Units: N/A
  Source: Derived from murcko_scaffold (structure withheld)
  Transformation: Opaque stable mapping (scaffold structure NOT recoverable)
  License Status: SAFE_DERIVED_OUTPUT
  Note: "Scaffold structure representation not redistributed. ID preserves equivalence."

structure_solvent_group_id (string)
  Description: Anonymous identifier preserving structure-solvent grouping
  Format: SSG_XXXXXX
  Units: N/A
  Source: Derived from canonical_smiles × solvent_group
  Transformation: Opaque stable mapping
  License Status: SAFE_DERIVED_OUTPUT
  Note: "Publication grouping constraint preserved without exposing structure or solvent."

partition (string)
  Description: Train/test partition
  Values: 'train' or 'test'
  License Status: SAFE_DERIVED_OUTPUT

SCAFFOLD SPLIT RESULTS
=======================

File: results/scaffold_disjoint/SCAFFOLD_SPLIT_RESULTS_FINAL.csv

[Columns documented in original - contains aggregated metrics only, no structures]
License Status: SAFE_DERIVED_OUTPUT

SCAFFOLD LEAKAGE AUDIT (PUBLIC)
================================

File: results/scaffold_disjoint/scaffold_public_group_leakage_audit.csv

split (int)
  Description: Split number
  License Status: SAFE_DERIVED_OUTPUT

seed (int)
  Description: Random seed
  License Status: SAFE_DERIVED_OUTPUT

train_n (int)
  Description: Training set size
  Units: Count
  License Status: SAFE_DERIVED_OUTPUT

test_n (int)
  Description: Test set size
  Units: Count
  License Status: SAFE_DERIVED_OUTPUT

scaffold_group_overlap (int)
  Description: Number of scaffold_group_id values in both train and test
  Units: Count
  Expected: 0
  License Status: SAFE_DERIVED_OUTPUT

molecule_group_overlap (int)
  Description: Number of molecule_group_id values in both train and test
  Units: Count
  Expected: 0
  License Status: SAFE_DERIVED_OUTPUT

structure_solvent_group_overlap (int)
  Description: Number of structure_solvent_group_id values in both train and test
  Units: Count
  Expected: 0
  License Status: SAFE_DERIVED_OUTPUT

status (string)
  Description: PASS if all overlaps = 0, else FAIL
  Values: 'PASS' or 'FAIL'
  License Status: SAFE_DERIVED_OUTPUT

SCAFFOLD PREDICTIONS
====================

File: results/scaffold_disjoint/SCAFFOLD_PREDICTIONS_FINAL.csv

[Columns contain predictions only, no structures - documented separately]
License Status: SAFE_DERIVED_OUTPUT

WITHHELD INFORMATION
====================

The following are NOT redistributed in the public release:

- canonical_smiles: Complete chemical structure representations
- murcko_scaffold: Bemis-Murcko scaffold SMILES
- solvent_group: Experimental solvent metadata (may be Reaxys-derived)
- compound_id: Internal compound identifiers
- OR_label: Original experimental label encoding
- PM_label: P/M stereochemical configuration
- row_index: Internal row indices

OPAQUE ID PROPERTIES
====================

All opaque IDs (MOL_*, SCF_*, SSG_*):
- Are stable and reproducible within this release
- Preserve equivalence relationships (same ID = same entity)
- Do NOT encode molecular structure
- Cannot be reverse-engineered to recover structure
- Are NOT deterministic hashes of SMILES (not matchable against databases)
- Private mappings to structures are withheld

---
Dictionary Version: 1.0
Release: Structure-Free Public Release
Date: 2026-08-31
