# Known limitations

1. **Licensed inputs are not redistributed.** Raw Reaxys exports, Reaxys identifiers, molecular-structure tables/mappings, and the private 5,204 × 2,283 feature matrix are excluded.
2. **Curation wrapper provenance.** The retained multi-stage curation logic and all publication counts are verified, but the exact wrapper used for the initial source-file import and the final serialization of the 5,204-row table was not uniquely identified.
3. **Schema A final assembly wrapper.** The 2,276-feature baseline builder, seven direction-sensitive geometry variables, exact 2,283-column order, frozen models, and downstream outputs are verified. A retained `feature_calculator.py` candidate was demonstrably non-final and is intentionally not released as the final driver.
4. **Primary model first-persistence wrapper.** The exact first historical script that wrote the root `ModelB_fold*.cbm` files was not uniquely identified. The retained historical root models, Figure-13-regenerated models, and public models were verified byte-identical for all five folds.
5. **SuperLearner historical Full8 lineage.** The historical eight-base workflow is retained, while the final published 0.9330/0.9081 comparator is reproduced by the verified six-base GBDT stacking workflow; the exact historical Full8 execution lineage remains partial.
6. **DFT-assisted cases.** The final SI documents three recoverable diagnostic cases; complete conformer-level DFT calculation inputs/workflows are not retained in the public package.
7. **Figure layout.** Some final manuscript figures were renumbered or assembled from recovered scientific source panels; historical script filenames therefore do not always match final manuscript figure numbers.
