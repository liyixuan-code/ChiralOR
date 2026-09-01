# ChiralOR v1.1.0 release audit

Status before manifest/package creation: **PASS**.

## Scientific/publication locks

- Publication Schema A ordered feature schema: 2,283 rows — PASS.
- Final benchmark ROC-AUC / Average Precision table — PASS.
- Calibration estimates and CIs, including ECE 0.0153–0.0345 — PASS.
- R01–R41 final source table: 41 classes, n=5,204, OR+=2,063, OR−=3,141 — PASS.
- R01–R41 summary checks: 14 classes ≥90% consistency; 24 classes n≥10; 4 of those ≥90% — PASS.
- Five public CatBoost fold-model hashes match the recorded publication identity table — PASS.
- Script provenance: 38 rows for 38 released Python scripts — PASS.
- Main Figure 1–6 and SI Figure S1–S10 lineage entries present — PASS.

## Public-safe restoration

- Primary, molecule-disjoint, and scaffold public split tables restored — PASS.
- `deployment/README_SCHEMA_B.md` restored — PASS.
- Public-safe main/SI figure source tables restored, including all ten retained SHAP dependence CSVs — PASS.
- Public-safe interaction ranking/stability/subset tables restored — PASS.
- Original curation scripts and recovered structural-domain rule-generation script present — PASS.
- Placeholder R01 reproduction skeleton removed — PASS.
- Non-final `baseline_features.py` isolated under `scripts/historical/nonfinal/` — PASS.

## Structure-free policy

The public tables do not redistribute canonical-SMILES observation tables, Reaxys IDs, Bemis–Murcko structure strings, private structure mappings, or the private 5,204 × 2,283 matrix. Opaque molecule/scaffold/group IDs are retained for leakage and split verification.

Current file count before `MANIFEST_SHA256.txt`: 131. The SHA256 manifest is generated only after all release files are frozen.
