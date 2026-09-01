# Public-file structure/licensing audit

Status: **PASS for the assembled structure-free release**.

The current public tables were checked for structure-bearing fields. The release does not include raw Reaxys exports, Reaxys identifiers, canonical-SMILES observation tables, Bemis–Murcko structure strings, or private ID-to-structure mappings.

Opaque `MOL_*`, `SCF_*`, and `SSG_*` identifiers preserve grouping relationships only.

`results/scaffold_disjoint/SCAFFOLD_LEAKAGE_AUDIT_FINAL.csv` contains columns named `murcko_scaffold_overlap_n`, `canonical_smiles_overlap_n`, and `smiles_solvent_overlap_n`; these values are integer overlap counts (all zero), not molecular structures.

The recovered original Python scripts may contain chemistry variable names (for example `canonical_smiles`) and historical local paths. Source-code variable names and chemical algorithms are not redistributed source records.

No separate open-source software license is granted; see the rights statement in the repository README.
