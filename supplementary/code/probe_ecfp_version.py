"""
ECFP6 RDKit-version probe.

Self-contained: needs only numpy + rdkit + pandas, so it can be run under any
interpreter/conda env. Computes ECFP6 (Morgan r=3, 2048 bits, useChirality=True)
for training molecules and compares bit-for-bit with the frozen matrix.

Purpose: identify which RDKit version reproduced the stored fingerprints by
running this script under each candidate interpreter, e.g.

    C:/python9/python.exe src/probe_ecfp_version.py --n 0
    C:/Users/lenovo/.conda/envs/my-rdkit-env/python.exe src/probe_ecfp_version.py --n 0

--n 0 verifies all 5204 rows (ECFP6 only, fast). Results are written to
results/ecfp6_probe_<rdkit version>.json so versions can be compared later.

Exit code 0 always; read the JSON or stdout for the verdict.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

BASE = Path(__file__).resolve().parents[1]
TRAIN = BASE / "data" / "original_training"
RESULTS = BASE / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0,
                    help="0 = all rows, else random sample of this size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    version = rdkit.__version__
    ml = pd.read_csv(TRAIN / "master_labels_v2.csv")
    X = np.load(TRAIN / "X_C3_5204x2283.npy", mmap_mode="r")

    if args.n and args.n < len(ml):
        rng = np.random.RandomState(args.seed)
        idx = np.sort(rng.choice(len(ml), size=args.n, replace=False))
    else:
        idx = np.arange(len(ml))

    n_exact = 0
    n_drift = 0
    bits = []
    worst = []

    for i in idx:
        smi = ml.iloc[i]["canonical_smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=3, nBits=2048, useChirality=True)
        gen = np.asarray(fp, dtype=np.float64)
        ref = np.asarray(X[i, 0:2048], dtype=np.float64)
        d = int(np.sum(gen != ref))
        if d == 0:
            n_exact += 1
        else:
            n_drift += 1
            bits.append(d)
            if len(worst) < 30:
                worst.append({"idx": int(i), "n_diff": d, "smiles": smi})
            elif d > max(w["n_diff"] for w in worst):
                worst.sort(key=lambda w: w["n_diff"])
                worst[0] = {"idx": int(i), "n_diff": d, "smiles": smi}

    bits = np.array(bits) if bits else np.array([0])
    out = {
        "rdkit_version": version,
        "python": sys.executable,
        "n_checked": int(len(idx)),
        "n_exact": int(n_exact),
        "n_drift": int(n_drift),
        "frac_drift": float(n_drift / max(len(idx), 1)),
        "mean_bits_diff": float(bits.mean()),
        "max_bits_diff": int(bits.max()),
        "perfect_match": bool(n_drift == 0),
        "worst_examples": sorted(worst, key=lambda w: -w["n_diff"]),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    outpath = Path(args.out) if args.out else RESULTS / f"ecfp6_probe_{version}.json"
    outpath.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"RDKit {version}  ({sys.executable})")
    print(f"  rows checked      : {out['n_checked']}")
    print(f"  exact             : {n_exact}")
    print(f"  drifted           : {n_drift}  ({out['frac_drift']:.2%})")
    print(f"  mean bits diff    : {bits.mean():.1f}")
    print(f"  max  bits diff    : {bits.max()}")
    print(f"  PERFECT MATCH     : {out['perfect_match']}")
    print(f"  saved             : {outpath}")


if __name__ == "__main__":
    main()
