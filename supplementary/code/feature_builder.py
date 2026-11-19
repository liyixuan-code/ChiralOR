"""
ChiralOR Publication V1 — 2283-dimensional feature builder.

Builds the full verified feature vector and returns a per-block audit so that
callers can see exactly which blocks were computed from structure and which
required an unavailable historical source.

Block provenance (schema/cleaning_rules_provenance.csv):
  ecfp6 / maccs / rdkit2d ...... computed from structure
  spas / pas ................... computed from structure
  shape3d ...................... regenerated; does NOT match the legacy v13 cache
  solvent_code ................. computed from solvent label (exact)
  dihedral / pm_code ........... only from a supplied/corrected source
  geometry ..................... reproducible GIVEN the true dihedral

Zero-fill rule (verified on the frozen matrix):
  col 2280 (signed_dihedral_NS_path) is 0 for 2040/5204 training records because
  the N/S reference path is structurally undefined for those molecules.
  The other six geometry columns have no zeros at all in the frozen matrix.
  Therefore:
    * a structural zero is permitted ONLY for cols listed in
      common.STRUCTURAL_ZERO_COLUMNS, and only when the geometry was actually
      computed (geometry_ok is True);
    * a zero anywhere else means "not computed" and the block is reported failed;
    * NaN/Inf are never acceptable in an output vector — they are trapped and
      reported, never silently converted.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, MACCSkeys

from common import (
    ALL_BLOCKS, BLOCKS, N_FEATURES, NO_NATURAL_ZERO_COLUMNS, SHAPE3D_NAMES,
    STRUCTURAL_ZERO_COLUMNS, add_original_modules, solvent_code,
)

add_original_modules()

from feature_generation import compute_signed_pas, block2_signed_geometry  # noqa: E402
import pm_calculator_correct  # noqa: E402


def _blank_audit():
    a = {b: False for b in ALL_BLOCKS}
    a.update({
        "chiral_idx": None,
        "errors": [],
        "pm_label": None,
        "pm_source": None,
        "dihedral_source": None,
        "needs_blind_review": False,
        "structural_zero_cols": [],
        "nan_inf_trapped": [],
    })
    return a


def _sanitize(vec, audit):
    """
    Trap NaN/Inf produced by any upstream computation.

    Returns (vec_clean, n_bad). Any offending column is zeroed for storage but
    recorded, so the block that produced it is marked failed by its own check.
    """
    bad = ~np.isfinite(vec)
    n_bad = int(bad.sum())
    if n_bad:
        idx = np.where(bad)[0]
        audit["nan_inf_trapped"].extend(int(i) for i in idx)
        vec[bad] = 0.0
    return vec, n_bad


def build_vector(smiles, solvent, dihedral=None, pm_label=None,
                 allow_derived_dihedral=False):
    """
    Build one 2283-dim feature vector.

    Parameters
    ----------
    smiles : str                Isomeric SMILES.
    solvent : str               Solvent label.
    dihedral : float or None    Known baseline P/M dihedral (degrees).
    pm_label : 'P' | 'M' | None Known P/M label.
    allow_derived_dihedral : bool
        Derive dihedral/PM via pm_calculator_correct and mark them APPROXIMATE.

    Returns
    -------
    (vec, audit)
        vec is None only when the molecule cannot be parsed at all.
        Otherwise vec is a finite float32 array, with per-block audit flags.
    """
    audit = _blank_audit()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        audit["errors"].append("invalid_smiles")
        return None, audit

    vec = np.zeros(N_FEATURES, dtype=np.float64)

    # ---- 0:2048  ECFP6 ----
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=3, nBits=2048, useChirality=True)
        vec[0:2048] = np.asarray(fp, dtype=np.float64)
        audit["ecfp6"] = True
    except Exception as e:
        audit["errors"].append(f"ecfp6: {e}")

    # ---- 2048:2215  MACCS, all 167 bits ----
    try:
        maccs = np.asarray(MACCSkeys.GenMACCSKeys(mol), dtype=np.float64)
        if maccs.shape[0] != 167:
            raise ValueError(f"MACCS length {maccs.shape[0]} != 167")
        vec[2048:2215] = maccs
        audit["maccs"] = True
    except Exception as e:
        audit["errors"].append(f"maccs: {e}")

    # ---- 2215:2223  RDKit 2D ----
    try:
        vec[2215:2223] = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.RingCount(mol),
            Descriptors.NumAromaticRings(mol),
        ]
        audit["rdkit2d"] = True
    except Exception as e:
        audit["errors"].append(f"rdkit2d: {e}")

    # ---- chiral center ----
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False)
    if not centers:
        audit["errors"].append("no_assigned_chiral_center")
    else:
        chiral_idx = centers[0][0]
        audit["chiral_idx"] = chiral_idx
        if len(centers) > 1:
            audit["needs_blind_review"] = True
            audit["errors"].append(f"multiple_chiral_centers={len(centers)}")

        try:
            spas = compute_signed_pas.compute_signed_pas(mol, chiral_idx)
            vec[2223:2243] = np.asarray(spas, dtype=np.float64)
            audit["spas"] = True
        except Exception as e:
            audit["errors"].append(f"spas: {e}")

        try:
            pas = compute_signed_pas.compute_original_pas(mol, chiral_idx)
            vec[2243:2263] = np.asarray(pas, dtype=np.float64)
            audit["pas"] = True
        except Exception as e:
            audit["errors"].append(f"pas: {e}")

    # ---- conformer ----
    mol_h, conf_audit = block2_signed_geometry.embed_conformer(smiles, seed=42)
    embedded = bool(conf_audit.get("embed_success", 0) == 1) and mol_h is not None

    if embedded:
        try:
            vec[2263:2273] = [float(getattr(Descriptors3D, n)(mol_h))
                              for n in SHAPE3D_NAMES]
            audit["shape3d"] = True
            audit["shape3d_source"] = "regenerated_ETKDGv3_seed42"
        except Exception as e:
            audit["errors"].append(f"shape3d: {e}")
    else:
        audit["errors"].append("shape3d: conformer_embed_failed")

    # ---- 2273 solvent ----
    sc = solvent_code(solvent)
    if sc is None:
        audit["errors"].append(f"unsupported_solvent={solvent}")
    else:
        vec[2273] = float(sc)
        audit["solvent_code"] = True

    # ---- 2274 dihedral / 2275 pm_code ----
    dih, lab = dihedral, pm_label
    if dih is not None:
        audit["dihedral_source"] = "supplied"
        audit["pm_source"] = "supplied"
    elif allow_derived_dihedral:
        try:
            r = pm_calculator_correct.calc_pm_helix(smiles)
            if isinstance(r, tuple) and len(r) >= 2:
                lab_auto, deg_auto = r[0], r[1]
                if lab_auto in ("P", "M") and deg_auto is not None:
                    lab, dih = lab_auto, float(deg_auto)
                    audit["dihedral_source"] = "derived_approximate"
                    audit["pm_source"] = "derived_approximate"
        except Exception as e:
            audit["errors"].append(f"pm_calc: {e}")

    if dih is not None and np.isfinite(dih):
        vec[2274] = float(dih)
        audit["dihedral"] = True

    if lab in ("P", "M"):
        vec[2275] = 1.0 if lab == "P" else -1.0
        audit["pm_label"] = lab
        audit["pm_code"] = True
    elif lab is None:
        pass  # absent by construction; recorded via audit['pm_code'] == False
    else:
        audit["errors"].append(f"pm_label_unrecognized={lab!r}")

    # ---- 2276:2283 geometry ----
    if dih is not None and embedded and audit["chiral_idx"] is not None:
        try:
            geom, ok = block2_signed_geometry.compute(
                smiles, audit["chiral_idx"], float(dih), mol_h, conf_audit)
            if ok and geom is not None:
                g = np.asarray(geom, dtype=np.float64)
                if g.shape[0] != 7:
                    raise ValueError(f"geometry length {g.shape[0]} != 7")
                # NaN here means "structurally undefined" for the N/S column and
                # "computation failure" everywhere else; distinguish before fill.
                nan_cols = [2276 + k for k in range(7) if not np.isfinite(g[k])]
                illegitimate = [c for c in nan_cols
                                if c not in STRUCTURAL_ZERO_COLUMNS]
                if illegitimate:
                    audit["errors"].append(
                        f"geometry_not_computed_cols={illegitimate}")
                else:
                    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                    audit["structural_zero_cols"] = sorted(
                        c for c in nan_cols if c in STRUCTURAL_ZERO_COLUMNS)
                    vec[2276:2283] = g
                    audit["geometry"] = True
            else:
                audit["errors"].append("geometry: compute_returned_not_ok")
        except Exception as e:
            audit["errors"].append(f"geometry: {e}")
    elif dih is None:
        audit["errors"].append("geometry: skipped_no_dihedral")
    else:
        audit["errors"].append("geometry: skipped_no_conformer_or_center")

    # ---- trap NaN/Inf anywhere ----
    vec, n_bad = _sanitize(vec, audit)
    if n_bad:
        # A trapped value means the block that owns it must not be trusted.
        lo_hi = BLOCKS
        for b, (lo, hi) in lo_hi.items():
            if any(lo <= i < hi for i in audit["nan_inf_trapped"]):
                audit[b] = False

    return vec.astype(np.float32), audit


def check_vector_integrity(vec, audit):
    """
    Final per-record completeness check.

    Returns (ok, problems). `ok` is True only when every block reproduced AND the
    stored values are finite AND no illegitimate zero was written into a
    geometry column that has no natural zeros in the frozen matrix.
    """
    problems = []

    if vec is None:
        return False, ["vector_is_none"]
    if vec.shape[0] != N_FEATURES:
        problems.append(f"n_features={vec.shape[0]}")
    if not np.all(np.isfinite(vec)):
        problems.append("non_finite_values")

    for b in ALL_BLOCKS:
        if not audit.get(b, False):
            problems.append(f"block_missing:{b}")

    # geometry columns that must never be zero unless structurally undefined
    if audit.get("geometry", False):
        for c in NO_NATURAL_ZERO_COLUMNS:
            if vec[c] == 0.0:
                problems.append(f"illegitimate_zero_col_{c}")

    return (len(problems) == 0), problems


def fatal_blocks(require_full_vector=True):
    """Blocks whose absence invalidates a row for Publication V1 strict evaluation."""
    base = ["ecfp6", "maccs", "rdkit2d", "spas", "pas", "solvent_code"]
    if require_full_vector:
        base += ["shape3d", "dihedral", "pm_code", "geometry"]
    return base
