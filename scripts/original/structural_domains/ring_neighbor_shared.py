"""
Shared ring-neighbor detection for V31 — single source of truth.

Imports the EXACT ring-neighbor / fused-ring-system logic from
generate_stereo_invariant_rules.py so that V31 ring_pair_onehot categories
are guaranteed identical to classification_rules_stereo_invariant.csv.

DO NOT reimplement ring-neighbor detection here.
"""
import os
import sys
import importlib.util

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_SCRIPTS_DIR, "generate_stereo_invariant_rules.py")

# Load the module by path without executing its __main__ block.
_spec = importlib.util.spec_from_file_location("_v31_stereo_rules_src", _SRC)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-export the canonical functions.
make_rule_mol = _mod.make_rule_mol
get_fused_ring_system = _mod.get_fused_ring_system
detect_ring_neighbors = _mod.detect_ring_neighbors          # returns e.g. "C-C", "N-S", "Unknown"
detect_substituent = _mod.detect_substituent
detect_aryl_category = _mod.detect_aryl_category

# Canonical ring-pair categories (sorted element pairs) observed in the
# stereo-invariant rule set, plus Unknown. Used for one-hot encoding.
RING_PAIR_CATEGORIES = [
    "C-C", "C-N", "C-O", "C-S",
    "N-N", "N-O", "N-S",
    "O-O", "O-S", "S-S",
    "Unknown",
]


def ring_pair_label(mol_rule, chiral_idx):
    """Return the canonical ring-neighbor label using the shared detector.

    detect_ring_neighbors already returns a sorted "A-B" string or "Unknown".
    We normalize any unseen pair to "Unknown" so the one-hot stays fixed-width.
    """
    label = detect_ring_neighbors(mol_rule, chiral_idx)
    if label in RING_PAIR_CATEGORIES:
        return label
    return "Unknown"
