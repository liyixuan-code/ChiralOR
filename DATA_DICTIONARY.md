# Data dictionary

This repository is a structure-free reproducibility release. Opaque IDs preserve grouping/equivalence without exposing molecular structures.

## Primary grouped-OOF predictions

`results/primary_oof/primary_grouped_oof_predictions.csv`

- `observation_id`: anonymous observation identifier.
- `fold`: grouped-CV fold assignment.
- `observed_label`: binary OR class used for metric verification (0 = OR−, 1 = OR+).
- `oof_probability_or_plus`: held-out P(OR+).
- `predicted_label_t0_5`: thresholded prediction at 0.5.

`data/splits/primary_grouped_fold_assignments_public.csv` contains the public structure-free fold assignment table.

## Molecule-disjoint outputs

`results/molecule_disjoint/molecule_disjoint_oof_predictions.csv` contains anonymous observation IDs, opaque molecule-group IDs, fold assignments, observed labels, and OOF probabilities.

`molecule_group_id` values (`MOL_*`) preserve molecular-identity equivalence but do not encode or reveal a structure. The private ID-to-structure mapping is not released.

`results/molecule_disjoint/molecule_disjoint_group_audit.csv` verifies one-fold-per-molecule grouping.

## Scaffold-disjoint outputs

`data/splits/scaffold_split_assignments_public.csv` and `results/scaffold_disjoint/scaffold_split_assignments_public.csv` use opaque:

- `MOL_*` molecule-group IDs;
- `SCF_*` scaffold-group IDs;
- `SSG_*` structure–solvent-group IDs.

No scaffold or molecular structure strings are included. `SCAFFOLD_LEAKAGE_AUDIT_FINAL.csv` contains overlap **counts only**; column names mentioning canonical SMILES or Murcko scaffolds describe the audited grouping concept and do not contain structure strings.

## Publication Schema A

`data/features_schema_A/feature_schema.csv` contains exactly 2,283 ordered feature names and feature-family labels. It does not contain the private feature matrix or SHAP values.

## Structural domains

`results/structural_domains/structural_rule_table_final_5204.csv` and `figure_source_data/supporting_information/structural_rule_table.csv` contain the 41 structure-free rule names, counts, OR+/OR− descriptive counts, empirical directional consistency, and predominant direction. They do not contain molecule structures.

## Interpretability source tables

`results/interpretability/` and `figure_source_data/supporting_information/` contain public-safe global SHAP summaries, interaction summaries, stability tables, opaque subset IDs, and selected dependence-source data.

## Withheld information

The public release does not redistribute raw Reaxys records, Reaxys identifiers, canonical-SMILES observation tables, Bemis–Murcko structure strings, private identifier-to-structure mappings, or the private 5,204 × 2,283 feature matrix.

Opaque IDs are not direct deterministic SMILES hashes.

Dictionary version: 1.1.0
