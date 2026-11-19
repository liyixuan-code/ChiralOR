"""
11 — Single-conformer vs multi-conformer descriptors (reviewer point #4, corrected).

Previous run (src/10) varied the EMBEDDING SEED with one conformer each; that
answers "is the descriptor seed-stable" but NOT "does using several conformers
change anything". This script does the real comparison:

  For a stratified sample of molecules, embed N conformers (ETKDGv3,
  pruneRmsThresh=0.5), MMFF94s-optimise each, and compute for every conformer
  * the 5 conformer-derived signed-geometry descriptors (cols 2276-2280)
  * the 10 shape3d descriptors (cols 2263-2272)
  (cols 2281/2282 are sin/cos of the stored dihedral and are conformer-invariant)

  Aggregates:
    shape3d      : mean over conformers (standard practice)
    geometry-A   : mean over conformers of the signed values
    geometry-B   : values from the LOWEST-MMFF-ENERGY conformer

  Reports
    1. within-molecule across-conformer SD per descriptor (the variance the
       reviewer asks for), relative to the across-molecule SD;
    2. 5-fold OOF AUC with the sample's blocks replaced by the multi-conformer
       aggregation, compared with the single-conformer frozen matrix — overall
       and on the sampled molecules.

Writes results/multiconformer_robustness.json
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors3D

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RESULTS, TRAIN, add_original_modules, write_json  # noqa: E402

add_original_modules()
from feature_generation import block2_signed_geometry  # noqa: E402

SHAPE3D_NAMES = ["PMI1", "PMI2", "PMI3", "NPR1", "NPR2",
                 "RadiusOfGyration", "InertialShapeFactor",
                 "Eccentricity", "Asphericity", "SpherocityIndex"]
GEO5_NAMES = ["signed_tetra_volume", "subst_to_ringplane_signed_dist",
              "ringnormal_dot_substvec", "signed_dihedral_subst_c_rn1_rn2",
              "signed_dihedral_NS_path"]
N_CONFS = 6
PRUNE = 0.5
SEED = 42


def embed_multi(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    molH = Chem.AddHs(mol)
    p = AllChem.ETKDGv3()
    p.randomSeed = SEED
    p.pruneRmsThresh = PRUNE
    cids = list(AllChem.EmbedMultipleConfs(molH, numConfs=N_CONFS, params=p))
    if not cids:
        p.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(molH, numConfs=N_CONFS, params=p))
    if not cids:
        return None
    props = AllChem.MMFFGetMoleculeProperties(molH, mmffVariant="MMFF94s")
    energies = []
    for cid in cids:
        try:
            ff = (AllChem.MMFFGetMoleculeForceField(molH, props, confId=cid)
                  if props is not None else
                  AllChem.UFFGetMoleculeForceField(molH, confId=cid))
            ff.Minimize(maxIts=500)
            energies.append(float(ff.CalcEnergy()))
        except Exception:
            energies.append(float("inf"))
    return molH, cids, np.array(energies)


def descriptors_for_conf(molH, cid, smiles, chiral_idx, dihedral):
    m2 = Chem.Mol(molH)
    conf = Chem.Conformer(molH.GetConformer(cid))
    m2.RemoveAllConformers()
    m2.AddConformer(conf, assignId=True)
    shape = [float(getattr(Descriptors3D, n)(m2)) for n in SHAPE3D_NAMES]
    # block2.compute() only evaluates geometry when audit["embed_success"] == 1
    geo, ok = block2_signed_geometry.compute(
        smiles, chiral_idx, dihedral, m2, {"embed_success": 1})
    geo = np.asarray(geo, dtype=np.float64) if geo is not None else np.full(7, np.nan)
    return shape, geo[:5], bool(ok)


def main():
    from catboost import CatBoostClassifier
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 roc_auc_score)

    ml = pd.read_csv(TRAIN / "master_labels_v2.csv")
    X0 = np.asarray(np.load(TRAIN / "X_C3_5204x2283.npy"))
    folds = pd.read_csv(TRAIN / "primary_grouped_fold_assignments_public.csv")
    fold_of = folds.fold.values
    y = ml.OR_label.values

    # stratified sample of unique molecules: 500, balanced by label of first record
    first = ml.groupby("canonical_smiles").agg(y=("OR_label", "first")).reset_index()
    rng = np.random.RandomState(13)
    pos = first[first.y == 1].sample(250, random_state=13)
    neg = first[first.y == 0].sample(250, random_state=13)
    sample_smiles = set(pos.canonical_smiles) | set(neg.canonical_smiles)
    sample_rows = np.where(ml.canonical_smiles.isin(sample_smiles))[0]
    print(f"sample: {len(sample_smiles)} unique molecules -> {len(sample_rows)} rows")

    # reference per molecule: stored dihedral + chiral index
    dihedral_by, center_by = {}, {}
    for i, s in enumerate(ml.canonical_smiles):
        if s in sample_smiles and s not in dihedral_by:
            dihedral_by[s] = float(X0[i, 2274])
            m = Chem.MolFromSmiles(s)
            cs = Chem.FindMolChiralCenters(m, includeUnassigned=False) if m else []
            center_by[s] = cs[0][0] if cs else None

    shape_all, geoA_all, geoB_all, nmol = {}, {}, {}, {}
    t0 = time.time()
    done = 0
    for s in sorted(sample_smiles):
        if center_by.get(s) is None:
            continue
        r = embed_multi(s)
        if r is None:
            continue
        molH, cids, energies = r
        shapes, geos = [], []
        for cid in cids:
            sh, ge, ok = descriptors_for_conf(molH, cid, s, center_by[s],
                                              dihedral_by[s])
            if ok:
                shapes.append(sh)
                geos.append(ge)
        if not shapes:
            continue
        shapes = np.array(shapes)
        geos = np.array(geos)
        shape_all[s] = shapes.mean(axis=0)
        geoA_all[s] = np.nanmean(geos, axis=0)
        best = int(np.argmin(energies[:len(geos)]))
        geoB_all[s] = geos[best]
        nmol[s] = {"n_confs": len(shapes), "conf_var_shape": shapes.var(axis=0, ddof=1)
                   if len(shapes) > 1 else np.zeros(10),
                   "conf_var_geo": geos.var(axis=0, ddof=1)
                   if len(geos) > 1 else np.zeros(5),
                   "shapes": shapes, "geos": geos}
        done += 1
        if done % 50 == 0:
            print(f"  {done} molecules  {(time.time()-t0)/60:.1f} min")

    print(f"embedded {len(shape_all)} molecules in {(time.time()-t0)/60:.1f} min")

    # ---------- 1. within-molecule across-conformer SD ----------
    def agg_var(store, ncols, names, key):
        sds = np.array([v[key] for v in store.values()])          # (mols, cols)
        sd = np.sqrt(sds)                                          # SD per molecule
        ref_sd = np.nanstd(np.vstack([v["shapes" if ncols == 10 else "geos"]
                                      for v in store.values()]), axis=0, ddof=1)
        out = {}
        for c in range(ncols):
            col = sd[:, c]
            out[names[c]] = {
                "mean_within_mol_sd": float(np.nanmean(col)),
                "median_within_mol_sd": float(np.nanmedian(col)),
                "p95_within_mol_sd": float(np.nanpercentile(col, 95)),
                "across_molecule_sd": float(ref_sd[c]),
                "relative_sd": float(np.nanmean(col) / (ref_sd[c] + 1e-12)),
                "median_confs": float(np.median([v["n_confs"] for v in store.values()])),
            }
        return out

    shape_var = agg_var(nmol, 10, SHAPE3D_NAMES, "conf_var_shape")
    geo_var = agg_var(nmol, 5, GEO5_NAMES, "conf_var_geo")
    print("\nwithin-molecule across-conformer SD (relative to across-molecule SD):")
    for k, v in geo_var.items():
        print(f"  {k:36s} rel={v['relative_sd']:.3f}  mean_sd={v['mean_within_mol_sd']:.4g}")

    # ---------- 2. AUC impact ----------
    models = []
    for k in range(5):
        m = CatBoostClassifier()
        m.load_model(str(Path(__file__).resolve().parents[1] / "models"
                          / f"catboost_schemaA_fold{k}.cbm"))
        models.append(m)

    def oof_frozen(Xm, mask):
        p = np.full(len(Xm), np.nan)
        for k in range(5):
            mk = fold_of == k
            if mk.any():
                p[mk] = models[k].predict_proba(Xm[mk])[:, 1]
        return p

    XA = X0.copy()
    XB = X0.copy()
    for i, s in enumerate(ml.canonical_smiles):
        if s in shape_all:
            XA[i, 2263:2273] = shape_all[s]
            XB[i, 2263:2273] = shape_all[s]
        if s in geoA_all:
            XA[i, 2276:2281] = geoA_all[s]
            XB[i, 2276:2281] = geoB_all[s]

    p0 = oof_frozen(X0, sample_rows)
    pA = oof_frozen(XA, sample_rows)
    pB = oof_frozen(XB, sample_rows)

    mask = np.isin(np.arange(len(ml)), sample_rows)
    res = {
        "sample": {"n_unique_molecules": len(shape_all), "n_rows": int(mask.sum()),
                   "n_conformers_requested": N_CONFS, "prune_rms": PRUNE},
        "overall_auc": {
            "single_conformer_frozen": float(
                roc_auc_score(y, oof_frozen(X0, mask))),
            "multi_mean": float(roc_auc_score(y, pA)),
            "multi_lowest_energy": float(roc_auc_score(y, pB)),
        },
        "sampled_rows_auc": {
            "single_conformer": float(roc_auc_score(y[mask], p0[mask])),
            "multi_mean": float(roc_auc_score(y[mask], pA[mask])),
            "multi_lowest_energy": float(roc_auc_score(y[mask], pB[mask])),
        },
        "delta_p_vs_single": {
            "multi_mean": {"mean_abs": float(np.mean(np.abs(pA - p0))),
                           "flips": int(np.sum((pA >= 0.5) != (p0 >= 0.5)))},
            "multi_lowest_energy": {"mean_abs": float(np.mean(np.abs(pB - p0))),
                                    "flips": int(np.sum((pB >= 0.5) != (p0 >= 0.5)))},
        },
        "within_molecule_conformer_sd": {"shape3d": shape_var, "geometry5": geo_var},
        "note": ("cols 2281/2282 are sin/cos of the stored dihedral and are "
                 "conformer-invariant by construction; the comparison covers the "
                 "5 conformer-derived geometry descriptors and the 10 shape3d "
                 "descriptors."),
    }
    write_json(RESULTS / "multiconformer_robustness.json", res)
    print("\nAUC (all records) single={:.4f} multi-mean={:.4f} multi-lowestE={:.4f}".format(
        res["overall_auc"]["single_conformer_frozen"],
        res["overall_auc"]["multi_mean"],
        res["overall_auc"]["multi_lowest_energy"]))
    print("saved: results/multiconformer_robustness.json")


if __name__ == "__main__":
    main()
