# Known Reproducibility Limitations

## 1. Publication Schema A source records
Molecular structures, raw Reaxys exports, Reaxys identifiers, and other licensed source-record fields are not redistributed in the public repository.

## 2. Full feature matrix
The frozen 5,204 × 2,283 Publication Schema A feature matrix is not included in the public release unless redistribution permission is confirmed.

## 3. End-to-end feature generation
The complete historical raw-record-to-Publication-Schema-A feature-generation pipeline was not fully recovered as a single executable workflow.

## 4. Primary historical training entry point
The exact historical entry-point script that generated the five frozen Publication Schema A CatBoost fold models was not recovered. The five frozen fold models and structure-free pooled OOF predictions are retained, and the reported primary grouped-OOF metrics are independently reproducible from those artifacts.

## 5. Geometry ablation
The published aggregate results are retained, but the complete historical no-geometry and geometry-only OOF prediction artifacts and the full 5,000-replicate bootstrap vector were not recovered.

## 6. Benchmark
The final publication benchmark table is provided. Historical per-observation prediction artifacts for all non-CatBoost benchmark models were not completely recovered.

## 7. SuperLearner
The SuperLearner is retained as an advanced historical comparator in the publication record; complete historical training/prediction provenance is not claimed where artifacts are unavailable.

## 8. Calibration
Published calibration statistics are provided. Where historical bin-level artifacts are unavailable, any regenerated reliability-bin data must be explicitly labeled as recalculated from frozen OOF predictions.

## 9. SHAP and interaction analyses
Recovered SHAP/interaction artifacts and original scripts are provided where available. The exact historical sampling procedure for some interaction visualization subsets may not be fully recoverable.

## 10. DFT-assisted development analysis
The DFT-assisted cases document feature-development provenance. Complete conformer-level calculation records were not retained for all historical cases, and this analysis is not presented as an independent quantitative validation set.

## 11. Public reproducibility scope
The public structure-free repository supports independent verification of the primary grouped-OOF, molecule-disjoint, and repeated scaffold-disjoint validation analyses. It is not presented as a complete end-to-end recreation of the study from licensed raw database records.

## 12. Schema A versus Schema B
All reported publication performance, calibration, ablation, and SHAP-based analyses apply to Publication Schema A. The web-deployment Schema B is distinct and is not assigned the publication performance estimates.
