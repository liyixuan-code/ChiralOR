"""
Same-dataset prior-method baselines on the PUBLISHED fold assignments.

Fixes a defect in the earlier Local_Chirality_Benchmark_FINAL run: that script
regenerated StratifiedGroupKFold splits, which reproduce only ~20% of the
published fold assignment. This script uses
data/original_training/primary_grouped_fold_assignments_public.csv directly,
so every model below is evaluated on the paper's exact folds and is directly
comparable to the published OOF AUC 0.9278.

Models (all OOF, fold-own-model only):
  M1  PAS only (20)                 RF
  M2  PAS + sPAS (40)               RF
  M3  Mamede-2021-inspired local-chirality descriptors  RF
      (stereocenter + 4 substituent fragments x physicochemical properties,
       chiral label; the original descriptors/code are not public, so this is
       a reconstruction from the published description, labelled "inspired")
  M4  Baseline 2276 (everything except 7 signed-geometry)  RF
  M5  Full 2283                     RF
Reference rows (no retraining):                        CatBoost
  M4  Baseline 2276                 0.9170
  M5  Full 2283                     0.9278

Writes results/same_dataset_baselines.json + .csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RESULTS, TRAIN, write_json  # noqa: E402

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             brier_score_loss, matthews_corrcoef,
                             roc_auc_score, average_precision_score)

RF_CONFIG = dict(n_estimators=500, max_depth=None, min_samples_split=5,
                 min_samples_leaf=2, max_features="sqrt", random_state=42,
                 n_jobs=-1, class_weight="balanced")

# per-atom descriptor tables
DONOR = Chem.MolFromSmarts("[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]")
ACCEPTOR = Chem.MolFromSmarts("[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),"
                              "$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),"
                              "$([nH0,o,s;+0])]")


def substituent_fragments(mol, center):
    """Return list of atom-index sets, one per substituent of the center."""
    nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(center).GetNeighbors()]
    frags = []
    for nb in nbrs:
        seen = {center, nb}
        frontier = [nb]
        for _ in range(1):          # 2 bonds from the center = 1 extra shell
            nxt = []
            for i in frontier:
                for n in mol.GetAtomWithIdx(i).GetNeighbors():
                    j = n.GetIdx()
                    if j not in seen:
                        seen.add(j)
                        nxt.append(j)
            frontier = nxt
        frags.append(sorted(seen - {center}))
    return frags


_mr_cache = {}


def mamede_features(smiles, solvent_code_int):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False)
    if not centers:
        return None
    ci, cip = centers[0]

    try:
        logp_c = Crippen._GetAtomContribs(mol)
        if logp_c and isinstance(logp_c[0], (tuple, list)):
            logp_c = [c[0] for c in logp_c]
    except Exception:
        logp_c = [0.0] * mol.GetNumAtoms()

    contribs = np.zeros((mol.GetNumAtoms(), 3))   # logP, TPSA, MR
    try:
        for i, (lp, mr) in enumerate(Crippen._CalcCrippenContribs(mol)):
            contribs[i, 0] = lp
            contribs[i, 2] = mr
    except Exception:
        pass
    try:
        for i, t in enumerate(rdMolDescriptors._CalcTPSAContribs(mol)):
            contribs[i, 1] = t
    except Exception:
        pass

    donor_atoms = {m[0] for m in mol.GetSubstructMatches(DONOR)}
    acceptor_atoms = {m[0] for m in mol.GetSubstructMatches(ACCEPTOR)}

    feats = []
    for frag in substituent_fragments(mol, ci):
        atoms = [mol.GetAtomWithIdx(i) for i in frag]
        feats += [
            len(frag),
            float(sum(a.GetMass() for a in atoms)),
            float(sum(1 for a in atoms if a.GetIsAromatic())),
            float(sum(1 for a in atoms if a.IsInRing())),
            float(contribs[list(frag), 0].sum()),
            float(contribs[list(frag), 1].sum()),
            float(contribs[list(frag), 2].sum()),
            float(len(set(frag) & donor_atoms)),
            float(len(set(frag) & acceptor_atoms)),
        ]
    while len(feats) < 4 * 9:
        feats.append(0.0)

    feats = feats[:36]
    # chiral label of the stereocenter (Mamede's descriptors carry chirality)
    feats.append(1.0 if cip == "R" else 0.0)
    # whole-molecule context the baseline methods also used
    feats += [
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol), Descriptors.RingCount(mol),
    ]
    onehot = [1.0 if solvent_code_int == k else 0.0 for k in (1, 2, 3, 4, 5)]
    return np.array(feats + onehot, dtype=np.float64)


def oof_rf(X, y, fold_of):
    p = np.full(len(y), np.nan)
    for k in range(5):
        tr, te = fold_of != k, fold_of == k
        clf = RandomForestClassifier(**RF_CONFIG)
        clf.fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return p


def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "oof_auc": float(roc_auc_score(y, p)),
        "oof_ap": float(average_precision_score(y, p)),
        "oof_mcc": float(matthews_corrcoef(y, yhat)),
        "oof_brier": float(brier_score_loss(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "accuracy": float(accuracy_score(y, yhat)),
    }


def main():
    ml = pd.read_csv(TRAIN / "master_labels_v2.csv")
    X = np.asarray(np.load(TRAIN / "X_C3_5204x2283.npy"))
    folds = pd.read_csv(TRAIN / "primary_grouped_fold_assignments_public.csv")
    assert (folds.observation_id.values == np.arange(len(ml))).all()
    fold_of = folds.fold.values
    y = ml.OR_label.values

    solv_code = {"methanol": 1, "ethanol": 2, "water": 3, "dcm": 4, "chloroform": 5}
    s_codes = ml.solvent_group.map(solv_code).fillna(0).astype(int)

    blocks = {
        "pas": (2243, 2263), "spas": (2223, 2243),
        "baseline": (0, 2276), "full": (0, 2283),
    }

    results = {}
    t0 = time.time()

    def run(name, Xsub, algorithm="RF"):
        print(f"  {name} ({Xsub.shape[1]} feats, {algorithm}) ...", flush=True)
        t = time.time()
        p = oof_rf(Xsub, y, fold_of)
        m = metrics(y, p)
        m.update({"name": name, "algorithm": algorithm,
                  "n_features": int(Xsub.shape[1]),
                  "elapsed_s": round(time.time() - t, 1)})
        print(f"    OOF AUC={m['oof_auc']:.4f}  AP={m['oof_ap']:.4f}  "
              f"MCC={m['oof_mcc']:.3f}  ({m['elapsed_s']}s)")
        results[name] = m
        return p

    print("Same-dataset baselines on PUBLISHED folds:")
    run("M1_PAS_only", X[:, blocks["pas"][0]:blocks["pas"][1]])
    run("M2_PAS_plus_sPAS",
        X[:, blocks["spas"][0]:blocks["spas"][1]].tolist() and
        X[:, np.r_[blocks["spas"][0]:blocks["spas"][1],
                   blocks["pas"][0]:blocks["pas"][1]]])

    # M3 Mamede-inspired descriptors
    print("  building Mamede-inspired descriptors ...", flush=True)
    t = time.time()
    rows = []
    for s, c in zip(ml.canonical_smiles, s_codes):
        f = mamede_features(s, {"_": c})
        if f is None:
            f = np.zeros(49)
        rows.append(f)
    Xm = np.vstack(rows)
    print(f"    descriptor matrix {Xm.shape} in {time.time()-t:.0f}s")
    run("M3_Mamede_inspired", Xm)

    run("M4_Baseline_2276", X[:, :2276])
    run("M5_Full_2283", X[:, :2283])

    out = {
        "protocol": {
            "folds": "primary_grouped_fold_assignments_public.csv (published folds, used as-is)",
            "note": "earlier benchmark regenerated folds and matched only ~20% of "
                    "these; this run supersedes it for fold-identical comparison",
            "model": "RandomForest (500 trees, class_weight=balanced, seed 42); "
                     "CatBoost reference values taken from the frozen models",
            "target": "OR sign (OR+ = 1)",
        },
        "results": results,
        "catboost_reference": {
            "M4_Baseline_2276": 0.9169959550195581,
            "M5_Full_2283": 0.9278324161099823,
        },
        "mamede_note": ("Mamede et al. JCIM 2021, 61, 4370-4379 report RF/ANN on "
                        "chiral-labelled hand-crafted descriptors (accuracy ~0.75 "
                        "on an independent set). Original descriptors and code are "
                        "not public; M3 reconstructs the published description "
                        "(stereocenter + 4 substituent fragments x 9 "
                        "physicochemical properties + chiral label + context) and "
                        "is therefore labelled 'inspired', not a replication."),
        "elapsed_total_s": round(time.time() - t0, 1),
    }
    write_json(RESULTS / "same_dataset_baselines.json", out)
    pd.DataFrame(results).T.to_csv(RESULTS / "same_dataset_baselines.csv")
    print(f"\nsaved: results/same_dataset_baselines.json")


if __name__ == "__main__":
    main()
