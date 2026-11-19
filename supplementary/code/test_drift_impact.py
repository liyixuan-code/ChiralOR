"""
Drift impact test: how much do the frozen fold models depend on feature blocks
that cannot be reproduced exactly for external molecules?

Two independent measurements:

1. BLOCK IMPORTANCE — sum CatBoost feature importances per feature block on the
   five frozen fold models. Direct measure of decision weight per block.

2. PERTURBATION — prediction changes when unreproducible blocks are destroyed
   (zeroed / shuffled) or replaced by current-RDKit recomputation (ECFP6 and
   rdkit2d under the installed RDKit). Metrics: mean/95th percentile |dp|,
   Spearman rho, class flips at 0.5, AUC on training labels (optimistic in
   absolute terms, but comparable across variants).

Env: needs rdkit + numpy + pandas + catboost. Run under the interpreter that
has catboost installed.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")

BASE = Path(__file__).resolve().parents[1]
TRAIN = BASE / "data" / "original_training"
MODELS = BASE / "models"
RESULTS = BASE / "results"

BLOCKS = {
    "ecfp6": (0, 2048),
    "maccs": (2048, 2215),
    "rdkit2d": (2215, 2223),
    "spas": (2223, 2243),
    "pas": (2243, 2263),
    "shape3d": (2263, 2273),
    "solvent_code": (2273, 2274),
    "dihedral": (2274, 2275),
    "pm_code": (2275, 2276),
    "geometry": (2276, 2283),
}
N_FEATURES = 2283


def rankdata(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ties
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    return float(np.dot(ra, rb) / (np.linalg.norm(ra) * np.linalg.norm(rb)))


def auc(score, label):
    r = rankdata(score)
    n_pos = int(label.sum())
    n_neg = len(label) - n_pos
    return float((r[label == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main():
    ml = pd.read_csv(TRAIN / "master_labels_v2.csv")
    X = np.asarray(np.load(TRAIN / "X_C3_5204x2283.npy"))
    y = ml["OR_label"].to_numpy(dtype=np.int64)

    models = []
    for i in range(5):
        m = CatBoostClassifier()
        m.load_model(str(MODELS / f"catboost_schemaA_fold{i}.cbm"))
        models.append(m)

    print(f"rows={X.shape[0]}  features={X.shape[1]}  positives={y.sum()}")

    # ---- 1. block importance ----
    imp = np.zeros(N_FEATURES)
    for m in models:
        fi = np.asarray(m.get_feature_importance(), dtype=np.float64)
        assert len(fi) == N_FEATURES, f"importance length {len(fi)} != {N_FEATURES}"
        imp += fi / len(models)
    imp = imp / imp.sum()
    block_imp = {k: float(imp[lo:hi].sum()) for k, (lo, hi) in BLOCKS.items()}
    print("\nBlock importance (mean over folds):")
    for k, v in sorted(block_imp.items(), key=lambda kv: -kv[1]):
        print(f"  {k:14s} {v:7.4f}")

    # ---- baseline predictions ----
    def predict_all(mat):
        C = np.ascontiguousarray(mat, dtype=np.float32)
        return np.mean([m.predict_proba(C)[:, 1] for m in models], axis=0)

    p0 = predict_all(X)

    # ---- 2. perturbations ----
    rng = np.random.RandomState(42)
    variants = {}

    def variant(name, mat):
        p = predict_all(mat)
        d = np.abs(p - p0)
        variants[name] = {
            "mean_abs_dp": float(d.mean()),
            "p95_abs_dp": float(np.percentile(d, 95)),
            "max_abs_dp": float(d.max()),
            "spearman_vs_baseline": spearman(p0, p),
            "n_class_flips": int(np.sum((p0 >= 0.5) != (p >= 0.5))),
            "auc": auc(p, y),
        }
        v = variants[name]
        print(f"  {name:22s} mean|dp|={v['mean_abs_dp']:.4f}  "
              f"p95={v['p95_abs_dp']:.4f}  flips={v['n_class_flips']:4d}  "
              f"rho={v['spearman_vs_baseline']:.4f}  auc={v['auc']:.4f}")

    print("\nPerturbations:")
    variants["baseline"] = {
        "mean_abs_dp": 0.0, "p95_abs_dp": 0.0, "max_abs_dp": 0.0,
        "spearman_vs_baseline": 1.0, "n_class_flips": 0, "auc": auc(p0, y)}

    def zero_copy(blocks):
        M = X.copy()
        for b in blocks:
            lo, hi = BLOCKS[b]
            M[:, lo:hi] = 0.0
        return M

    def shuffle_copy(blocks):
        M = X.copy()
        perm = rng.permutation(len(M))
        for b in blocks:
            lo, hi = BLOCKS[b]
            M[:, lo:hi] = X[perm, lo:hi]
        return M

    variant("shape3d_zero", zero_copy(["shape3d"]))
    variant("shape3d_shuffle", shuffle_copy(["shape3d"]))
    variant("geometry_zero", zero_copy(["geometry"]))
    variant("geometry_shuffle", shuffle_copy(["geometry"]))
    variant("tail_zero", zero_copy(["shape3d", "geometry"]))
    variant("tail_shuffle", shuffle_copy(["shape3d", "geometry"]))

    # recompute ecfp6 + rdkit2d with the installed RDKit
    mols = [Chem.MolFromSmiles(s) for s in ml["canonical_smiles"]]
    M = X.copy()
    import rdkit
    for i, m in enumerate(mols):
        fp = AllChem.GetMorganFingerprintAsBitVect(
            m, radius=3, nBits=2048, useChirality=True)
        M[i, 0:2048] = np.asarray(fp, dtype=np.float64)
        M[i, 2215:2223] = [
            Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m),
            Descriptors.NumHAcceptors(m), Descriptors.NumHDonors(m),
            Descriptors.NumRotatableBonds(m), Descriptors.RingCount(m),
            Descriptors.NumAromaticRings(m)]
    variant(f"recomputed_rdkit_{rdkit.__version__}", M)

    # realistic external vector: recomputed chemistry + destroyed tail
    M2 = M.copy()
    perm = rng.permutation(len(M2))
    for b in ("shape3d", "geometry"):
        lo, hi = BLOCKS[b]
        M2[:, lo:hi] = X[perm, lo:hi]
    variant("realistic_external", M2)

    out = {
        "rdkit_version": rdkit.__version__,
        "n_rows": int(X.shape[0]),
        "block_importance": block_imp,
        "variants": variants,
        "note": ("AUC is computed on training labels and is therefore optimistic "
                 "in absolute terms; it is comparable across variants. "
                 "realistic_external = current-RDKit chemistry blocks + "
                 "shuffled shape3d/geometry (stand-in for unreproducible tail)."),
    }
    outpath = RESULTS / f"drift_impact_{rdkit.__version__}.json"
    outpath.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved: {outpath}")


if __name__ == "__main__":
    main()
