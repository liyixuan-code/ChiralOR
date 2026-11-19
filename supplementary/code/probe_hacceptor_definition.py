"""
Identify which RDKit H-bond-acceptor definition produced frozen column 2218.

Tests candidate implementations against all training rows and reports exact
match counts. Run under any interpreter with rdkit + numpy + pandas.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

BASE = Path(__file__).resolve().parents[1]
ml = pd.read_csv(BASE / "data/original_training/master_labels_v2.csv")
X = np.load(BASE / "data/original_training/X_C3_5204x2283.npy", mmap_mode="r")
ref = np.asarray(X[:, 2218], dtype=np.float64)

candidates = {
    "Descriptors.NumHAcceptors": Descriptors.NumHAcceptors,
    "Lipinski.NumHAcceptors": Lipinski.NumHAcceptors,
    "CalcNumHBA": rdMolDescriptors.CalcNumHBA,
    "CalcNumLipinskiHBA": rdMolDescriptors.CalcNumLipinskiHBA,
    "NO_count": lambda m: sum(1 for a in m.GetAtoms() if a.GetSymbol() in ("N", "O")),
}

mols = [Chem.MolFromSmiles(s) for s in ml["canonical_smiles"]]

print(f"python={sys.executable}")
import rdkit
print(f"rdkit={rdkit.__version__}  n={len(mols)}")
for name, fn in candidates.items():
    vals = np.array([fn(m) for m in mols], dtype=np.float64)
    n_exact = int(np.sum(vals == ref))
    d = vals - ref
    print(f"  {name:28s} exact {n_exact}/{len(ref)}  "
          f"({n_exact/len(ref):.4%})  mean_diff {d.mean():+.3f}")
