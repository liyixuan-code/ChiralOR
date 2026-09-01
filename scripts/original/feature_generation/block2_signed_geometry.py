"""
V31 Block 2 — Signed local geometry features (REQUIRES 3D conformer).

3D conformers are generated FRESH for V31 only (ETKDG + MMFF94s, seed=42,
UFF fallback). They are used ONLY for these signed-geometry features and do
NOT replace baseline dihedral / P/M / sPAS.

baseline_pm_dihedral_sin/cos come from the EXISTING processed_data_v2 dihedral
column (NOT recomputed) — see source_note in feature names.

Missing geometry -> NaN imputed to 0 by the driver + missing_geometry_flag.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from .ring_neighbor_shared import make_rule_mol, get_fused_ring_system

FEATURE_NAMES = [
    "signed_tetra_volume",
    "subst_to_ringplane_signed_dist",
    "ringnormal_dot_substvec",
    "signed_dihedral_subst_c_rn1_rn2",
    "signed_dihedral_NS_path",
    "baseline_pm_dihedral_sin",
    "baseline_pm_dihedral_cos",
]


def feature_names():
    return list(FEATURE_NAMES)


def embed_conformer(smiles, seed=42, max_attempts=10):
    """Return (mol_with_H_and_conf, audit_dict).

    audit_dict keys: embed_success, n_embed_attempts, forcefield_used,
    forcefield_success, final_energy, stereochemistry_preserved.
    """
    audit = dict(embed_success=0, n_embed_attempts=0, forcefield_used="fail",
                 forcefield_success=0, final_energy=float("nan"),
                 stereochemistry_preserved=0)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, audit
    # record stereo before
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    stereo_before = Chem.FindMolChiralCenters(mol, useLegacyImplementation=False)

    molH = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cid = -1
    for attempt in range(1, max_attempts + 1):
        audit["n_embed_attempts"] = attempt
        params.randomSeed = seed + attempt - 1
        try:
            cid = AllChem.EmbedMolecule(molH, params)
        except Exception:
            cid = -1
        if cid == 0:
            break
    if cid != 0:
        # last-resort: random-coords embed (robust for awkward systems)
        try:
            p2 = AllChem.ETKDGv3()
            p2.randomSeed = seed
            p2.useRandomCoords = True
            cid = AllChem.EmbedMolecule(molH, p2)
        except Exception:
            cid = -1
    if cid != 0:
        return None, audit
    audit["embed_success"] = 1

    # MMFF94s first
    energy = float("nan")
    used = "fail"
    ok = 0
    try:
        if AllChem.MMFFHasAllMoleculeParams(molH):
            res = AllChem.MMFFOptimizeMolecule(molH, mmffVariant="MMFF94s", maxIters=1000)
            ff = AllChem.MMFFGetMoleculeForceField(
                molH, AllChem.MMFFGetMoleculeProperties(molH, mmffVariant="MMFF94s"))
            energy = float(ff.CalcEnergy()) if ff is not None else float("nan")
            used = "MMFF"
            ok = 1 if res == 0 else 0
        else:
            raise ValueError("no MMFF params")
    except Exception:
        try:
            res = AllChem.UFFOptimizeMolecule(molH, maxIters=1000)
            ff = AllChem.UFFGetMoleculeForceField(molH)
            energy = float(ff.CalcEnergy()) if ff is not None else float("nan")
            used = "UFF"
            ok = 1 if res == 0 else 0
        except Exception:
            used = "fail"
            ok = 0
    audit["forcefield_used"] = used
    audit["forcefield_success"] = ok
    audit["final_energy"] = energy

    # stereo after
    try:
        Chem.AssignStereochemistryFrom3D(molH)
        molH_noH = Chem.RemoveHs(molH)
        stereo_after = Chem.FindMolChiralCenters(molH_noH, useLegacyImplementation=False)
        audit["stereochemistry_preserved"] = 1 if (
            len(stereo_before) == len(stereo_after)) else 0
    except Exception:
        audit["stereochemistry_preserved"] = 0

    return molH, audit


def _signed_dihedral(p0, p1, p2, p3):
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / (np.linalg.norm(b1) + 1e-12)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def compute(smiles, chiral_idx, baseline_dihedral, mol_conf, audit):
    """Compute the 7-dim Block-2 vector.

    mol_conf: AddHs mol with embedded conformer (or None).
    baseline_dihedral: float from processed_data_v2 (may be NaN).
    Returns (vector(np.float64,7), geometry_ok(bool)).
    """
    n = len(FEATURE_NAMES)
    vec = np.full(n, np.nan, dtype=np.float64)

    # baseline_pm_dihedral sin/cos from EXISTING dihedral (independent of 3D)
    if baseline_dihedral is not None and np.isfinite(baseline_dihedral):
        rad = np.radians(float(baseline_dihedral))
        vec[5] = float(np.sin(rad))
        vec[6] = float(np.cos(rad))

    geometry_ok = False
    if mol_conf is not None and audit.get("embed_success", 0) == 1:
        try:
            conf = mol_conf.GetConformer()
            pos = conf.GetPositions()  # (nAtoms_with_H, 3)
            ci = int(chiral_idx)
            atom = mol_conf.GetAtomWithIdx(ci)
            nbrs = [nb.GetIdx() for nb in atom.GetNeighbors()]
            c = pos[ci]

            # signed tetrahedral volume over up to 4 neighbors (sorted by idx)
            nb_sorted = sorted(nbrs)
            if len(nb_sorted) >= 4:
                a, b, cc, d = [pos[i] for i in nb_sorted[:4]]
                vol = np.dot(np.cross(b - a, cc - a), d - a) / 6.0
                vec[0] = float(vol)
            elif len(nb_sorted) == 3:
                a, b, cc = [pos[i] for i in nb_sorted[:3]]
                vol = np.dot(np.cross(b - c, cc - c), a - c) / 6.0
                vec[0] = float(vol)

            # fused-ring plane (use achiral graph mapping; heavy-atom ring set)
            mol_rule = make_rule_mol(Chem.MolFromSmiles(smiles))
            fused = get_fused_ring_system(mol_rule, ci)
            fused = [i for i in fused if i < pos.shape[0]]
            ext = [i for i in nbrs if i not in set(fused)]
            if len(fused) >= 3:
                ring_pts = pos[list(fused)]
                centroid = ring_pts.mean(axis=0)
                u, s, vt = np.linalg.svd(ring_pts - centroid)
                normal = vt[2]
                normal = normal / (np.linalg.norm(normal) + 1e-12)
                if ext:
                    subst = pos[ext[0]]
                    sv = subst - c
                    svn = sv / (np.linalg.norm(sv) + 1e-12)
                    vec[1] = float(np.dot(subst - centroid, normal))   # signed dist
                    vec[2] = float(np.dot(normal, svn))                 # normal . substvec
                    # signed dihedral subst-c-rn1-rn2
                    rn = sorted([i for i in nbrs if i in set(fused)])
                    if len(rn) >= 2:
                        vec[3] = _signed_dihedral(subst, c, pos[rn[0]], pos[rn[1]])

            # N/S-path signed dihedral: nearest N or S neighbor path
            hetero = [i for i in range(mol_conf.GetNumAtoms())
                      if mol_conf.GetAtomWithIdx(i).GetSymbol() in ("N", "S")]
            if hetero and len(nbrs) >= 1:
                # pick nearest hetero by 3D distance to chiral center
                hd = sorted(hetero, key=lambda i: np.linalg.norm(pos[i] - c))
                h = hd[0]
                # dihedral h - c - nb0 - nb1 if possible
                others = [i for i in nbrs if i != h]
                if len(others) >= 2:
                    vec[4] = _signed_dihedral(pos[h], c, pos[others[0]], pos[others[1]])

            geometry_ok = np.isfinite(vec[0])  # tetra volume as the geometry anchor
        except Exception:
            geometry_ok = False

    return vec, geometry_ok
