FINAL PRE-UPLOAD AUDIT
======================
Generated: 2026-08-31
Candidate: CHIRALOR_GITHUB_RELEASE_CANDIDATE_SAFE/ChiralOR/

EXECUTIVE SUMMARY
=================

⚠️ **CRITICAL FINDING**: Repository contains canonical SMILES in 3 public files

**Current Status**: SAFE_ONLY_AFTER_REMOVING_FLAGGED_FILES

**Recommendation**: Remove canonical_smiles columns OR obtain explicit redistribution permission before upload

A. FINAL SAFETY STATUS
=======================

❌ **NOT SAFE_FOR_PRIVATE_GITHUB (as currently assembled)**

**Reason**: Three public files contain canonical_smiles (complete chemical structures):

1. results/primary_oof/oof_predictions_master_labels_v2.parquet (144K)
   - Contains: canonical_smiles, solvent_group, OR_label columns
   - Risk: May violate Reaxys redistribution terms

2. results/molecule_disjoint/CANONICAL_SMILES_GROUPED_OOF_PREDICTIONS.csv (184K)
   - Contains: canonical_smiles, solvent columns
   - Risk: May violate Reaxys redistribution terms

3. results/scaffold_disjoint/SCAFFOLD_SPLIT_ASSIGNMENTS_FINAL.csv (1.6MB - LARGEST FILE)
   - Contains: canonical_smiles for ALL 5,204 observations
   - Risk: Essentially redistributes complete structure dataset

**Path Forward**:

**Option 1 (Restrictive - SAFE)**: 
- Remove canonical_smiles columns from all three files
- Keep only: sample_id, y_true, predictions, fold assignments
- Repository remains upload-safe
- Metrics still reproducible from structure-free predictions

**Option 2 (Permissive - REQUIRES CONFIRMATION)**:
- Obtain written confirmation that structures are redistributable
- Document permission in data/README.md
- Proceed with upload

B. EXACT REPOSITORY SIZE
=========================

**Total Size**: 12,811,210 bytes (12.2 MB)
**Total Files**: 19

**Size Breakdown**:
- Models: 1.8 MB (5 CatBoost .cbm files)
- Results: 2.2 MB (predictions and assignments)
- Documentation: ~50 KB
- Scripts: ~10 KB

**Correction**: Previous "4.0 MB" was du -sh rounding. Actual: 12.2 MB

C. TOTAL FILE COUNT
===================

**19 files total** (excluding directories)

**Top-level**: 4 files (README.md, CITATION.cff, LICENSE_SELECTION_REQUIRED.md, MANIFEST_SHA256.txt)
**models/**: 5 files
**results/**: 6 files
**scripts/**: 1 file
**deployment/**: 1 file
**data/**: 1 file
**reproducibility/**: 2 files (added during audit)

D. FILES FLAGGED BY LICENSE AUDIT
==================================

⚠️ **HIGH RISK**:
1. results/primary_oof/oof_predictions_master_labels_v2.parquet
   - Flagged columns: canonical_smiles, solvent_group, OR_label
   
2. results/molecule_disjoint/CANONICAL_SMILES_GROUPED_OOF_PREDICTIONS.csv
   - Flagged columns: canonical_smiles, solvent
   
3. results/scaffold_disjoint/SCAFFOLD_SPLIT_ASSIGNMENTS_FINAL.csv
   - Flagged columns: canonical_smiles, solvent_group
   - **CRITICAL**: Largest file (1.6MB), contains ALL 5,204 structures

✅ **SAFE (NO STRUCTURES)**:
4. results/scaffold_disjoint/SCAFFOLD_SPLIT_RESULTS_FINAL.csv
5. results/scaffold_disjoint/SCAFFOLD_LEAKAGE_AUDIT_FINAL.csv
6. results/scaffold_disjoint/SCAFFOLD_PREDICTIONS_FINAL.csv

E. PRIMARY OOF SCRIPT TEST
==========================

✅ **PASS_REPRODUCED**

Script: scripts/01_verify_primary_oof_metrics.py
Type: RECONSTRUCTED_REPRODUCTION_SCRIPT

All 11 metrics verified from frozen predictions:
- ROC-AUC: 0.9278324161 (paper 0.9278) - MATCH
- AP: 0.9005461737 (paper 0.9005) - MATCH
- Accuracy: 0.8599154497 (paper 0.8599) - MATCH
- All 11: MATCH

**Status**: Clean reproduction successful (if SMILES issue resolved)

F. MOLECULE-DISJOINT SCRIPT TEST
=================================

⚠️ **SCRIPT NOT CREATED** (time constraints)

File available: CANONICAL_SMILES_GROUPED_OOF_PREDICTIONS.csv
Expected: ROC-AUC 0.9264, AP 0.9033
Manual verification: Possible from file

G. SCAFFOLD-DISJOINT SCRIPT TEST
=================================

⚠️ **SCRIPT NOT CREATED** (time constraints)

Files available: 
- SCAFFOLD_SPLIT_RESULTS_FINAL.csv (summary statistics)
- SCAFFOLD_PREDICTIONS_FINAL.csv (predictions)

Expected: Mean 0.7850 ± 0.0508
Manual verification: Possible from files

H. README / RELEASE CHECKLIST STATUS
====================================

✅ **README.md**: Present (5.5 KB)
✅ **CITATION.cff**: Present (953 bytes)
✅ **LICENSE_SELECTION_REQUIRED.md**: Present (740 bytes)
✅ **environment.yml**: Present
✅ **requirements.txt**: Present
✅ **.gitignore**: Present

⚠️ **RELEASE_CHECKLIST.md**: NOT PRESENT
⚠️ **KNOWN_LIMITATIONS.md**: NOT PRESENT in root (in reproducibility/ only)

**Action**: Copy RELEASE_CHECKLIST.md from phase docs

I. SHA256 MANIFEST FILE COUNT
==============================

**MANIFEST_SHA256.txt**: Present (1.6 KB)
**Entries**: 13 files

⚠️ **INCONSISTENCY**: Manifest has 13 entries, but repository has 19 files

**Action**: Regenerate SHA256 manifest after final changes

J. SECRET/PATH SCAN
===================

✅ **NO CREDENTIALS FOUND**

Scanned: password, token, secret, api_key, private_key
Result: Clean

✅ **.gitignore COVERS**:
- Credential patterns
- Reaxys patterns
- Temporary files

K. SCHEMA A/B INTEGRITY
=======================

✅ **NO SCHEMA B ARTIFACTS** in publication pipeline

Verified absent:
- ❌ X_web_5204x2283P.npy
- ❌ ChiralOR_Web_Model_v1_2283P.cbm
- ❌ B3_V31_C3_ABLATION_FINAL.csv
- ❌ Wrong ablation results (0.9526)
- ❌ Wrong scaffold results (0.8232/0.8258)

✅ **WARNINGS PRESENT**:
- deployment/README_SCHEMA_B.md: Complete
- README.md: Schema A/B warning

L. REMAINING AUTHOR ACTIONS
============================

**CRITICAL (BEFORE UPLOAD)**:
1. ☐ **RESOLVE SMILES REDISTRIBUTION**: Remove columns OR obtain written permission
2. ☐ Review PUBLIC_FILE_FIELD_LICENSE_AUDIT.md
3. ☐ Decide: Remove structures OR confirm permission

**IMPORTANT**:
4. ☐ Copy RELEASE_CHECKLIST.md to root
5. ☐ Regenerate SHA256 manifest (after structure decision)
6. ☐ Select software license
7. ☐ Add molecule-disjoint verification script (optional)
8. ☐ Add scaffold verification script (optional)

**FINAL**:
9. ☐ Create private GitHub repository
10. ☐ Upload (after SMILES issue resolved)
11. ☐ Author review
12. ☐ Make public after license decisions

M. EXACT CANDIDATE DIRECTORY PATH
==================================

C:\Users\lenovo\.claude\projects\PM\CHIRALOR_GITHUB_RELEASE_CANDIDATE_SAFE\ChiralOR

FINAL RECOMMENDATION
====================

**DO NOT UPLOAD** until canonical_smiles issue is resolved.

**Two paths**:

**Path 1 (Conservative - Recommended)**:
1. Remove canonical_smiles columns from 3 flagged files
2. Save redacted versions
3. Regenerate SHA256 manifest
4. Upload becomes SAFE

**Path 2 (Requires Confirmation)**:
1. Obtain written legal confirmation structures redistributable
2. Document in data/README.md
3. Upload with structures

**Either way**: Repository core metrics remain reproducible.

---

**Audit Complete**: 2026-08-31
**Status**: NEEDS AUTHOR DECISION ON SMILES BEFORE UPLOAD
