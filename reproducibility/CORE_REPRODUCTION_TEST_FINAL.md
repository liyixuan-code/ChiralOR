CORE REPRODUCTION TEST - SANITIZED STRUCTURE-FREE RELEASE
==========================================================
Generated: 2026-08-31

All tests performed using ONLY public structure-free files.

1. PRIMARY GROUPED-OOF (SANITIZED)
===================================

File: results/primary_oof/primary_grouped_oof_predictions.csv
Removed: canonical_smiles, compound_id, solvent_group, PM_label, OR_label

✅ ALL 11 METRICS VERIFIED:
- ROC-AUC: 0.9278324161 (paper 0.9278) ✅
- AP: 0.9005461737 (paper 0.9005) ✅
- Accuracy: 0.8599154497 (paper 0.8599) ✅
- Balanced Accuracy: 0.8495174218 (paper 0.8495) ✅
- Sensitivity: 0.7993213766 (paper 0.7993) ✅
- Specificity: 0.8997134670 (paper 0.8997) ✅
- Precision: 0.8396130346 (paper 0.8396) ✅
- NPV: 0.8722222222 (paper 0.8722) ✅
- F1: 0.8189719394 (paper 0.8190) ✅
- MCC: 0.7054060161 (paper 0.7054) ✅
- Brier: 0.1027364703 (paper 0.1027) ✅

STATUS: ✅ PASS_REPRODUCED

2. MOLECULE-DISJOINT SENSITIVITY (SANITIZED)
=============================================

File: results/molecule_disjoint/molecule_disjoint_oof_predictions.csv
Removed: canonical_smiles, solvent
Added: molecule_group_id (opaque, 3895 unique molecules)

✅ METRICS VERIFIED:
- ROC-AUC: 0.9264441194 (paper 0.9264) ✅
- AP: 0.9033410557 (paper 0.9033) ✅
- Accuracy: 0.8560722521 (paper 0.8561) ✅
- Sensitivity: 0.7867183713 (paper 0.7867) ✅
- Specificity: 0.9016236867 (paper 0.9016) ✅

✅ GROUPING CONSTRAINT VERIFIED:
- Unique molecules: 3895
- Max folds per molecule: 1 (expected 1) ✅
- Grouping violations: 0 ✅

STATUS: ✅ PASS_REPRODUCED

3. SCAFFOLD-DISJOINT EVALUATION (SANITIZED)
============================================

Files: 
- scaffold_split_assignments_public.csv
- scaffold_public_group_leakage_audit.csv
- SCAFFOLD_SPLIT_RESULTS_FINAL.csv

Removed: canonical_smiles, murcko_scaffold, solvent_group
Added: molecule_group_id, scaffold_group_id, structure_solvent_group_id (opaque)

✅ SUMMARY STATISTICS:
- Mean AUC: 0.7850 (paper 0.7850) ✅
- SD: 0.0508 (paper 0.0508) ✅
- 10 splits verified ✅

✅ LEAKAGE AUDIT (OPAQUE IDs):
- All 10 splits: PASS ✅
- Scaffold group overlap: 0 ✅
- Molecule group overlap: 0 ✅
- Structure-solvent group overlap: 0 ✅

Anonymous Group Statistics:
- Unique molecules: 3895
- Unique scaffolds: 709
- Unique structure-solvent groups: 4078

STATUS: ✅ PASS_REPRODUCED

FINAL ASSESSMENT
================

✅ All claimed reproducible results verified from structure-free predictions
✅ All grouping/leakage constraints verified with opaque IDs
✅ No molecular structures in public files
✅ All paper metrics match sanitized outputs

SANITIZATION IMPACT
===================

Removed Columns:
- canonical_smiles (all files)
- murcko_scaffold
- solvent_group
- compound_id
- OR_label
- PM_label
- row_index

Preserved Information:
- All predictions and probabilities
- All fold assignments
- All grouping relationships (via opaque IDs)
- All leakage constraints
- All reported metrics

What Can Be Reproduced:
✅ All 11 primary metrics
✅ Molecule-disjoint metrics
✅ Scaffold-disjoint metrics
✅ Grouping constraint verification
✅ Leakage audit verification

What Cannot Be Reproduced:
❌ Independent model retraining (no structures)
❌ Feature generation (no structures)
❌ SHAP analysis (no structures)

CONCLUSION
==========

Structure-free public release enables complete reproduction of all claimed validation metrics without redistributing potentially restricted molecular structures.

---
Test Date: 2026-08-31
Status: ✅ ALL TESTS PASS
