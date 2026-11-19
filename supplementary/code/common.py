"""
ChiralOR Publication V1 — External Validation: shared constants and helpers.

All paths are relative to the package root (no absolute paths, no Windows paths).
"""
import hashlib
import json
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
DATA     = PKG_ROOT / "data"
RAW      = DATA / "raw_reaxys"
TRAIN    = DATA / "original_training"
MODELS   = PKG_ROOT / "models"
SCHEMA   = PKG_ROOT / "schema"
MODULES  = PKG_ROOT / "original_modules"
RESULTS  = PKG_ROOT / "results"
LOGS     = PKG_ROOT / "logs"
CONFIGS  = PKG_ROOT / "configs"

for _d in (RESULTS, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- features
N_FEATURES = 2283

# Column layout verified against schema/feature_schema.csv
BLOCKS = {
    "ecfp6":        (0,    2048),
    "maccs":        (2048, 2215),
    "rdkit2d":      (2215, 2223),
    "spas":         (2223, 2243),
    "pas":          (2243, 2263),
    "shape3d":      (2263, 2273),
    "solvent_code": (2273, 2274),
    "dihedral":     (2274, 2275),
    "pm_code":      (2275, 2276),
    "geometry":     (2276, 2283),
}

# Every block must reproduce for a strict Publication V1 external run.
ALL_BLOCKS = list(BLOCKS.keys())

# Column indices whose structural zero is legitimate (verified on the frozen matrix):
#   col 2280 (geo:signed_dihedral_NS_path) = 0 for 2040/5204 records, because the
#   N/S-reference path is structurally undefined for those molecules.
#   The other six geometry columns contain NO zeros in the frozen matrix, so a zero
#   there can only mean "not computed" and must never be treated as a value.
STRUCTURAL_ZERO_COLUMNS = {2280}
NO_NATURAL_ZERO_COLUMNS = [2276, 2277, 2278, 2279, 2281, 2282]

SHAPE3D_NAMES = ["PMI1", "PMI2", "PMI3", "NPR1", "NPR2",
                 "RadiusOfGyration", "InertialShapeFactor",
                 "Eccentricity", "Asphericity", "SpherocityIndex"]

GEOMETRY_NAMES = [
    "geo:signed_tetra_volume",
    "geo:subst_to_ringplane_signed_dist",
    "geo:ringnormal_dot_substvec",
    "geo:signed_dihedral_subst_c_rn1_rn2",
    "geo:signed_dihedral_NS_path",
    "geo:baseline_pm_dihedral_sin",
    "geo:baseline_pm_dihedral_cos",
]

# ------------------------------------------------- solvent encoding (VERIFIED)
# Derived from X_C3_5204x2283.npy column 2273 vs training solvent labels:
#   methanol=1 (1482)  ethanol=2 (355)  water=3 (20)  dcm=4 (388)  chloroform=5 (2959)
SOLVENT_CODE = {
    "methanol": 1, "meoh": 1, "ch3oh": 1,
    "ethanol": 2, "etoh": 2, "c2h5oh": 2,
    "water": 3, "h2o": 3,
    "dcm": 4, "dichloromethane": 4, "ch2cl2": 4,
    "chloroform": 5, "chcl3": 5,
}

ALLOWED_SOLVENTS = ("methanol", "ethanol", "water", "dcm", "chloroform")

# Molecular-weight range actually present in the frozen training matrix.
# NOTE: no MW filter has any evidenced basis in the recovered curation code;
# the range below is descriptive only (see schema/cleaning_rules_provenance.csv).
TRAIN_MW_RANGE = (136.2, 788.0)


def solvent_code(name):
    """Return integer solvent code, or None if the solvent is not one of the five."""
    if name is None:
        return None
    key = str(name).strip().lower()
    if key in SOLVENT_CODE:
        return SOLVENT_CODE[key]
    for token, code in SOLVENT_CODE.items():
        if token in key:
            return code
    return None


def solvent_name(name):
    """Normalize a solvent label to one of the five allowed names, or None."""
    c = solvent_code(name)
    if c is None:
        return None
    return {1: "methanol", 2: "ethanol", 3: "water", 4: "dcm", 5: "chloroform"}[c]


# ---------------------------------------------------------------- modules
def add_original_modules():
    """Make the vendored Publication V1 feature modules importable."""
    p = str(MODULES)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_feature_modules():
    """Import the vendored feature modules (PAS/sPAS, geometry, P/M)."""
    add_original_modules()
    from feature_generation import compute_signed_pas, block2_signed_geometry
    import pm_calculator_correct
    return compute_signed_pas, block2_signed_geometry, pm_calculator_correct


# ---------------------------------------------------------------- models
MODEL_FILES = [f"catboost_schemaA_fold{i}.cbm" for i in range(5)]


def model_paths():
    return [MODELS / f for f in MODEL_FILES]


# ------------------------------------------------- external dihedral table
DIHEDRAL_TABLE = CONFIGS / "external_dihedral.csv"
DIHEDRAL_REQUIRED_COLUMNS = ["smiles", "solvent", "dihedral", "pm_label"]


def load_external_dihedral(strict_validate=True):
    """
    Load configs/external_dihedral.csv.

    Returns
    -------
    (table, report)
        table  : dict {(canonical_smiles, solvent_name): (dihedral, pm_label)}
        report : dict with validation outcome
    """
    from rdkit import Chem

    report = {
        "path": str(DIHEDRAL_TABLE),
        "exists": DIHEDRAL_TABLE.exists(),
        "n_rows": 0,
        "n_valid": 0,
        "n_rejected": 0,
        "rejected": [],
        "columns": [],
        "usable": False,
        "reason": None,
    }
    if not DIHEDRAL_TABLE.exists():
        report["reason"] = "file_not_found"
        return {}, report

    import csv as _csv
    with open(DIHEDRAL_TABLE, "r", encoding="utf-8-sig", newline="") as f:
        reader = _csv.DictReader(f)
        report["columns"] = reader.fieldnames or []
        missing = [c for c in DIHEDRAL_REQUIRED_COLUMNS if c not in report["columns"]]
        if missing:
            report["reason"] = f"missing_columns={missing}"
            return {}, report

        table = {}
        for i, row in enumerate(reader, start=2):
            report["n_rows"] += 1
            raw_smi = (row.get("smiles") or "").strip()
            raw_solv = (row.get("solvent") or "").strip()
            raw_dih = (row.get("dihedral") or "").strip()
            raw_pm = (row.get("pm_label") or "").strip().upper()

            problems = []
            mol = Chem.MolFromSmiles(raw_smi) if raw_smi else None
            if mol is None:
                problems.append("invalid_smiles")
                canon = None
            else:
                canon = Chem.MolToSmiles(mol, isomericSmiles=True)

            sname = solvent_name(raw_solv)
            if sname is None:
                problems.append("unsupported_solvent")

            try:
                dih = float(raw_dih)
                if not (dih == dih) or abs(dih) == float("inf"):
                    problems.append("dihedral_not_finite")
            except (TypeError, ValueError):
                problems.append("dihedral_not_numeric")
                dih = None

            if raw_pm not in ("P", "M"):
                problems.append("pm_label_not_P_or_M")

            if problems:
                report["n_rejected"] += 1
                report["rejected"].append({"line": i, "problems": problems,
                                           "smiles": raw_smi[:80]})
                continue

            key = (canon, sname)
            if key in table and table[key] != (dih, raw_pm):
                problems.append("conflicting_duplicate_key")
                report["n_rejected"] += 1
                report["rejected"].append({"line": i, "problems": problems,
                                           "smiles": raw_smi[:80]})
                continue

            table[key] = (dih, raw_pm)
            report["n_valid"] += 1

    report["n_unique_keys"] = len(table)
    if report["n_valid"] == 0:
        report["reason"] = "no_valid_rows"
    elif strict_validate and report["n_rejected"] > 0:
        report["reason"] = f"{report['n_rejected']}_rows_rejected"
    else:
        report["usable"] = True

    return table, report


# ---------------------------------------------------------------- io helpers
def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def data_fingerprint():
    """Fingerprint of the inputs a feature matrix depends on (resume safety)."""
    parts = {}
    for p in (RESULTS / "external_curated.csv",
              TRAIN / "master_labels_v2.csv",
              CONFIGS / "model_manifest.json"):
        parts[p.name] = file_sha256(p) if p.exists() else None
    parts["dihedral_table"] = (
        file_sha256(DIHEDRAL_TABLE) if DIHEDRAL_TABLE.exists() else None)
    return parts


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serializable: {type(o)}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_default)
    return path


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
