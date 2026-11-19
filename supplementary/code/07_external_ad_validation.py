"""
07 — Applicability-domain (AD) external validation.

Standard QSAR practice (OECD principle 3): state the applicability domain and
report external performance inside it. Two AD definitions are applied:

  AD-A  Tanimoto similarity to the nearest training molecule >= tau
        (tau = 0.6 primary, 0.7 strict; Morgan r=2, 1024 bits).
  AD-B  measured |OR| >= 10 deg (weak rotations are condition-sensitive and
        their sign is close to measurement noise).

For each definition: coverage, n, AUC with group-cluster bootstrap CI.
The full-domain result is retained unchanged alongside.

Writes results/external_ad_validation.json.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RESULTS, TRAIN, write_json  # noqa: E402

N_BOOT = 5000


def cluster_boot_auc(y, p, clusters, n_boot=N_BOOT, seed=42):
    rng = np.random.RandomState(seed)
    uniq = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[c] for c in pick])
        yy = y[idx]
        if len(set(yy)) < 2:
            continue
        vals.append(roc_auc_score(yy, p[idx]))
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ml = pd.read_csv(TRAIN / "master_labels_v2.csv")
    pred = pd.read_csv(RESULTS / "external_predictions.csv")
    ev = pred[pred["complete_vector"].astype(bool)
              & pred["prob_ensemble"].notna()
              & pred["OR_class"].notna()].copy()
    ev["y"] = ev["OR_class"].astype(int)
    ev["clusters"] = ev["smiles"].astype(str) + "|" + ev["solvent"].astype(str)

    # nearest-training Tanimoto
    tr_fps = []
    for s in ml["canonical_smiles"].unique():
        m = Chem.MolFromSmiles(s)
        if m is not None:
            tr_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024))
    sims = {}
    for s in ev["smiles"].unique():
        m = Chem.MolFromSmiles(s)
        if m is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
            sims[s] = float(max(DataStructs.BulkTanimotoSimilarity(fp, tr_fps)))
    ev["max_tanimoto"] = ev["smiles"].map(sims)
    ev["absOR"] = ev["OR_value"].abs()

    def subset_result(mask, name):
        g = ev[mask]
        if len(g) < 10 or g["y"].nunique() < 2:
            r = {"name": name, "n": int(len(g)), "coverage": float(mask.mean()),
                 "auc": None, "auc_ci95": None}
        else:
            r = {"name": name,
                 "n": int(len(g)),
                 "coverage": float(mask.mean()),
                 "n_pos": int(g["y"].sum()),
                 "auc": float(roc_auc_score(g["y"], g["prob_ensemble"])),
                 "auc_ci95": cluster_boot_auc(g["y"].values,
                                              g["prob_ensemble"].values,
                                              g["clusters"].values)}
        print(f"  {name:32s} n={r['n']:4d} ({r['coverage']:5.1%} coverage)"
              + (f"  AUC={r['auc']:.3f} [{r['auc_ci95'][0]:.3f}, {r['auc_ci95'][1]:.3f}]"
                 if r["auc"] is not None else ""))
        return r

    print("External validation by applicability-domain definition:")
    results = {
        "all_records": subset_result(pd.Series(True, index=ev.index), "full_domain"),
        "AD_similarity_0.6": subset_result(ev["max_tanimoto"] >= 0.6,
                                           "tanimoto>=0.6"),
        "AD_similarity_0.7": subset_result(ev["max_tanimoto"] >= 0.7,
                                           "tanimoto>=0.7"),
        "AD_OR10": subset_result(ev["absOR"] >= 10, "|OR|>=10"),
        "AD_OR15": subset_result(ev["absOR"] >= 15, "|OR|>=15"),
        "AD_combined_0.6_OR10": subset_result(
            (ev["max_tanimoto"] >= 0.6) & (ev["absOR"] >= 10),
            "tanimoto>=0.6 & |OR|>=10"),
    }

    print("\nAll definitions (all-records comparison):")
    tr_sims = []
    print(f"  external median nearest-training Tanimoto: {ev['max_tanimoto'].median():.3f}")

    out = {
        "ad_definitions": {
            "similarity": "max Tanimoto (Morgan r=2, 1024 bits) to any unique "
                          "training molecule; thresholds 0.6 / 0.7",
            "signal": "measured |OR| thresholds 10 / 15 degrees "
                      "(weak rotations are condition-sensitive)",
            "bootstrap": f"group-cluster (smiles x solvent), {N_BOOT} replicates",
        },
        "results": results,
        "note": ("Full-domain AUC is reported unchanged in external_metrics.json. "
                 "AD-restricted metrics quantify generalization inside the domain "
                 "the paper's claim actually covers; coverage is stated with each."),
    }
    write_json(RESULTS / "external_ad_validation.json", out)
    print(f"\nsaved: {RESULTS / 'external_ad_validation.json'}")


if __name__ == "__main__":
    main()
