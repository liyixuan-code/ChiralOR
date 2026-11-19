# Revision notes — what each analysis answers

This file maps the analyses in this directory to the points raised in review,
and records the exact numbers quoted in the manuscript and SI. Every number
below is read from a file in `data/`.

## 1. Enantiomer separation in the cross-validation split

**Question.** The grouping unit is canonical SMILES × solvent. For two
enantiomers these differ, so enantiomers can fall into different folds; because
ECFP6 with `useChirality=False` and MACCS are identical for a mirror-image pair,
a model could in principle learn "flip the sign" rather than chemistry.

**Measured.**
- 3,895 unique isomeric SMILES; 3,254 InChIKey skeletons; 641 skeletons occur
  with exactly two stereoisomeric forms (1,282 records, 24.6% of the dataset).
- 567 of those 641 families occupy more than one primary fold.
- ECFP6 (`useChirality=True`) separates a mirror pair at Tanimoto 0.811;
  without chirality the two vectors are identical, and MACCS is identical.

**Control.** A second cross-validation scheme grouped by stereoisomer family
instead of by isomeric SMILES. Pooled OOF AUC is 0.9305, against 0.9278 under
the primary grouping and 0.9264 under molecule-disjoint grouping. Removing the
enantiomer overlap does not raise performance, so the reported AUC is not
inflated by that route. Source: `03_Enantiomer_Validation/enantiomer_family_cv_results.json`
in the revision tree; numbers reproduced in SI Table S6.

## 2. Label reliability

**Question.** How much of the residual error is annotation error, and does it
change the interpretation of the reported AUC?

**Measured.**
- Dataset-internal consistency: among 4,078 structure–solvent groups, one group
  (two records) carries conflicting signs.
- Error rate falls monotonically with rotation magnitude: 21.2% (0–5°), 26.4%
  (5–10°), 21.7% (10–20°), 10.7% (20–50°), 10.9% (50–200°), 12.9% (200–2,000°).
- Pooled OOF AUC rises from 0.9278 (all records) to 0.9309 (|OR| ≥ 5°), 0.9383
  (≥ 10°), 0.9457 (≥ 15°), 0.9511 (≥ 20°), 0.9518 (≥ 30°).
- Sources: `label_magnitude_stratification.csv`, `label_error_rate_by_magnitude.csv`.

**Direction of the argument.** Random label noise moves a measured AUC toward
0.5, not away from it. The all-record AUC is therefore a conservative estimate
of performance on correctly annotated records; the magnitude stratification
quantifies the same effect with the full sample rather than with a hand-audited
subset.

**Quantum-chemical component.** Ten compounds from the model–experiment
discordant pool were studied by Boltzmann-averaged specific rotation
(Gaussian 16, B3LYP/6-31G(d,p), D3(BJ), SMD; ORCA, ωB97X-D3/aug-cc-pVDZ,
CPCM, 589.3 nm). Three compounds (1–3) were selected for detailed case reporting
on the criterion of conformer-level sign instability:
- compound 1: seven conformers, −234.1 to +360.6°, Boltzmann average −223.84°;
- compound 2: four conformers, −100.7 to +71.0°, Boltzmann average −40.31°;
- compound 3: three conformers, −15.7 to +88.9°, no Boltzmann average
  (opposite-sign conformers).

These calculations are diagnostic. They are not used to estimate a label-error
rate, no label was modified on their basis, and they are not validation. Their
function is to motivate the chirality-aware descriptors, which are then tested
by ablation and by scaffold-disjoint and domain-restricted evaluation.

## 3. Single-conformer descriptors and zero-filled geometry values

**Question.** Descriptors depend on one ETKDG conformer (seed 42). What happens
with several conformers, and is zero a legitimate encoding for a signed feature?

**Conformer sensitivity.** Three embedding seeds (42/123/999) and, separately,
six conformers with RMSD pruning 0.5 Å on a stratified 500-molecule sample.
Within-molecule across-conformer SD as a fraction of the between-molecule SD:
signed tetrahedral volume 0.004; substituent–ring dihedral 0.018; N/S-path
dihedral 0.216; substituent-to-ring-plane distance 0.772; ring-normal projection
0.830. The 3D-shape block ranges from 0.13 to 0.62.

Prediction is much less sensitive than the descriptors: pooled OOF AUC 0.9278
(archived single conformer), 0.9277 (conformer mean), 0.9274 (lowest-energy
conformer); across seeding, 0.9293 / 0.9269 / 0.9277 in the frozen models and
0.9282 / 0.9279 / 0.9284 when retrained. Mean |Δp| against the archived matrix
is 0.0045 with 19 of 613 sampled rows changing class at 0.5.

**Missing values.** Only column 2,280 (N/S-path dihedral) contains zeros
(2,040 of 5,204 records). The zero set coincides exactly with the molecules
containing neither nitrogen nor sulphur, so zero is the exact encoding of a
structurally undefined quantity. Controls: train-fold median imputation gives an
identical OOF AUC (0.9278); adding explicit missingness indicators gives 0.9263;
deleting the column gives 0.9244. No other geometry column contains zeros.

Sources: `multiconformer_robustness.json`, `conformer_robustness.json`,
`missingness_experiment.json`.

## 4. Ablation controls

**Question.** The signed-geometry increment is +0.0108 AUC. Could it simply be
the effect of adding seven columns?

**Controls.**
- Permutation: the seven values replaced by permuted geometry blocks across ten
  seeds give 0.9161 ± 0.0014 (range 0.9132–0.9185). The full model (0.9278)
  exceeds all ten.
- Scaffold-disjoint: ten Murcko scaffold splits with no shared scaffold reach
  0.785 ± 0.051 for the full model against 0.7458 for the baseline without
  signed geometry (Δ +0.039 ± 0.032, positive in 9 of 10), larger than the
  +0.011 seen under the grouped protocol.
- Decomposition: the seven columns separate into five conformer-derived
  descriptors and two dihedral sine/cosine terms. Alone, the five reach 0.8585
  (CatBoost) and 0.8655 (RF); the two terms reach 0.6853 and 0.6660; the P/M
  code alone reaches 0.5137 and 0.5178, i.e. chance. Adding the five to a
  2,274-dimensional baseline that excludes dihedral, P/M code and geometry
  raises AUC from 0.9114 to 0.9220 (RF 0.8994 to 0.9194).

Sources: `geometry_block_decomposition.json`, `drift_impact_2022.09.5.json`,
`05_Scaffold_Ablation/scaffold_ablation_summary.json` in the revision tree.

## 5. Same-dataset comparison with prior representations

**Question.** The manuscript compares AUC values taken from studies with
different data, labels and protocols. Is there a comparison that holds data and
protocol fixed?

**Constraint.** Mamede et al. (2021), Chen et al. (2019) and Zhou et al. (2025)
publish no trained weights; Reaxys data cannot be redistributed. A direct
transfer of this dataset into those models is therefore not possible, and any
claim that it was done would be inaccurate.

**Substitute.** Representation-level reconstruction on the identical 5,204
records and the identical published folds (the previously used benchmark
regenerated its splits and reproduced only 20% of the published fold
assignments; this run uses the published assignment file directly).

| Representation | dims | CatBoost AUC | CatBoost acc | CatBoost BA | RF AUC |
|---|---|---|---|---|---|
| PAS | 20 | 0.6607 | 0.6095 | 0.5755 | 0.6268 |
| PAS + sPAS | 40 | 0.6517 | 0.6042 | 0.5726 | 0.6197 |
| Stereocentre-centred (Mamede-style) | 49 | 0.8933 | 0.8240 | 0.8132 | 0.9015 |
| Baseline without signed geometry | 2,276 | 0.9170 | 0.8430 | 0.8323 | 0.9134 |
| Full representation | 2,283 | 0.9278 | 0.8599 | 0.8495 | 0.9263 |

The Mamede-style reconstruction reaches AUC 0.893 / accuracy 0.824, consistent
in magnitude with the 75% accuracy reported for the original independent test
set; the full representation exceeds it by 0.035 AUC and 0.036 balanced
accuracy. Under random forest the ordering is unchanged, so the difference is
not attributable to the classifier.

The Cartesian-coordinate representation of Zhou et al. was also reconstructed.
It is a fixed-length serialisation of atom coordinates and element one-hot
vectors, padded to a constant atom count; on molecules of this size the padding
dominates and the result depends on atom ordering (AUC 0.584–0.631 across three
orderings). It is reported as a lower bound on that representation under
resized molecules, not as the published method's performance.

Sources: `same_dataset_baselines.json`, `same_dataset_baselines.csv`,
`same_dataset_mamede_catboost_oof.csv`, `same_dataset_ohecc.json`.

## 6. Applicability domain and external transfer

An independent Reaxys set (717 curated records; 698 with complete vectors;
443 molecules) was scored by the frozen ensemble. Full-domain AUC is 0.520
[0.448, 0.590], which is chance. Stratification by similarity to the nearest
training molecule shows why: the median nearest-neighbour Tanimoto is 0.434
(81% of records below 0.6), and AUC rises to 0.777 [0.650, 0.881] for Tanimoto
≥ 0.6 (20.9% of records) and 0.812 [0.676, 0.918] when |OR| ≥ 10° is also
required. Within the training set the median nearest-neighbour Tanimoto is
0.826, so the external set lies outside the training manifold rather than being
a sample of the same population. The reported external result therefore applies
to the domain-qualified subset; it is not a full-domain validation.

Source: `07_external_ad_validation.py` → `external_ad_validation.json` in the
revision tree.

## 7. Feature-matrix provenance

The frozen matrix was assembled across RDKit generations. Probes:
- ECFP6 reproduces bit-exactly on 5,204/5,204 records under RDKit 2025.03.6;
  under 2022.09.5 and 2023.03.1 the drift is identical (2,245 records, mean
  22.8 bits, max 29 of 2,048), localising the difference to Morgan hashing.
- NumHAcceptors differs from the frozen column on 256 records under every
  build available for Python 3.9 (identical counts from 2022.03.5, 2022.09.5,
  2025.03.6), i.e. the frozen column predates the current acceptor definition.
  Substituting the current definition changes the pooled OOF AUC by less than
  0.001 and the mean predicted probability by 0.040.
- The 3D-shape block was computed from a conformer cache absent from the public
  release; ETKDGv3 regeneration differs by 8–30%.
- Paired out-of-fold check: the full reconstruction, scored by each sample's own
  held-out model, gives 0.9263 against 0.9278 for the archived matrix
  (probability MAE 0.0395; 197 of 5,204 class flips).

Sources: `ecfp6_probe_*.json`, `paired_oof_reconstruction.json`,
`FEATURE_GATE.json`.
