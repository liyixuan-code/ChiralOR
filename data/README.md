# Public data package

This directory contains **structure-free** machine-readable artifacts for Publication Schema A.

## Included

- `features_schema_A/feature_schema.csv`: ordered 2,283-feature publication schema.
- `splits/primary_grouped_fold_assignments_public.csv`: anonymous structure-solvent group IDs and the frozen primary fold assignment.
- `splits/molecule_disjoint_fold_assignments_public.csv`: anonymous molecule group IDs and the molecule-disjoint fold assignment.
- `splits/scaffold_split_assignments_public.csv`: anonymous molecule/scaffold/structure-solvent group IDs for the ten scaffold-disjoint partitions.

## Not redistributed

Molecular structures, canonical SMILES, Murcko scaffold strings, Reaxys identifiers, raw database exports, and private mappings from opaque group IDs to structures are not included. The frozen 5,204 x 2,283 feature matrix and source-record-level inputs are also withheld unless redistribution permission is confirmed.

The public files retain only the identifiers and model outputs needed to verify the released validation analyses. Experimental labels contained in prediction artifacts remain subject to final author/institutional review under the applicable source-data license.
