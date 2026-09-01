#!/usr/bin/env python3
"""
Generate Stereochemistry-Invariant Classification Rules

This script fixes the stereo-dependent classification rule generation by:
1. Creating achiral molecular graphs (mol_rule) for all rule detection
2. Ensuring enantiomer pairs receive identical classification rules
3. Generating audit reports for enantiomer consistency
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from collections import defaultdict
import os

BASE = 'C:/Users/lenovo/.claude/projects/PM/V12混合系统优化版（4.1V12可行版本1的results）/phase1_dl_fix/V13_results/基于V13revised再次优化方案'

# ============================================================================
# Core Functions
# ============================================================================

def make_rule_mol(mol_stereo):
    """Create achiral molecular graph for rule generation"""
    mol_rule = Chem.Mol(mol_stereo)
    Chem.RemoveStereochemistry(mol_rule)
    return mol_rule

def get_fused_ring_system(mol, atom_idx):
    """Get all atoms in the fused ring system containing atom_idx"""
    ri = mol.GetRingInfo()
    rings = [set(ring) for ring in ri.AtomRings()]
    # Find rings containing atom_idx
    my_rings = [r for r in rings if atom_idx in r]
    if not my_rings:
        return set()
    fused = set()
    for r in my_rings:
        fused.update(r)
    changed = True
    while changed:
        changed = False
        for r in rings:
            r_set = set(r)
            if r_set & fused and not r_set.issubset(fused):
                fused.update(r_set)
                changed = True
    return fused

def detect_substituent(mol_rule, chiral_idx):
    """Detect substituent type on achiral molecular graph"""
    atom = mol_rule.GetAtomWithIdx(chiral_idx)
    fused_system = get_fused_ring_system(mol_rule, chiral_idx)
    if not fused_system:
        return 'Unknown'

    external_atoms = []
    for neighbor in atom.GetNeighbors():
        if neighbor.GetIdx() not in fused_system:
            external_atoms.append(neighbor)

    if not external_atoms:
        return 'Unknown'

    # Prioritize heteroatoms over carbon
    priority = {'O': 0, 'N': 1, 'S': 2, 'F': 3, 'Cl': 3, 'Br': 3, 'I': 3, 'C': 4}
    external_atoms.sort(key=lambda a: priority.get(a.GetSymbol(), 99))

    element = external_atoms[0].GetSymbol()
    if element == 'O': return 'O'
    elif element == 'N': return 'N'
    elif element == 'C': return 'C'
    elif element == 'S': return 'S'
    elif element in ['F','Cl','Br','I']: return 'Halogen'
    else: return 'Other'

def detect_ring_neighbors(mol_rule, chiral_idx):
    """Detect ring neighbor types on achiral molecular graph"""
    atom = mol_rule.GetAtomWithIdx(chiral_idx)
    fused_system = get_fused_ring_system(mol_rule, chiral_idx)
    if not fused_system:
        return 'Unknown'

    ring_nbs = []
    for neighbor in atom.GetNeighbors():
        if neighbor.GetIdx() in fused_system:
            ring_nbs.append(neighbor.GetSymbol())

    if len(ring_nbs) < 2:
        return 'Unknown'

    ring_nbs_sorted = sorted(ring_nbs[:2])
    return f'{ring_nbs_sorted[0]}-{ring_nbs_sorted[1]}'

def detect_aryl_category(mol_rule, chiral_idx):
    """Detect aryl category on achiral molecular graph"""
    fused_system = get_fused_ring_system(mol_rule, chiral_idx)
    atom = mol_rule.GetAtomWithIdx(chiral_idx)

    # Check neighbors not in fused system
    for neighbor in atom.GetNeighbors():
        nid = neighbor.GetIdx()
        if nid in fused_system:
            continue
        if neighbor.GetIsAromatic():
            return 'Aryl0'
        for nb2 in neighbor.GetNeighbors():
            if nb2.GetIdx() == chiral_idx: continue
            if nb2.GetIsAromatic():
                return 'Aryl1'
            for nb3 in nb2.GetNeighbors():
                if nb3.GetIdx() in (chiral_idx, neighbor.GetIdx()): continue
                if nb3.GetIsAromatic():
                    return 'Aryl2'

    # Check if fused system itself contains aromatic atoms
    for aidx in fused_system:
        if aidx == chiral_idx: continue
        a = mol_rule.GetAtomWithIdx(aidx)
        if a.GetIsAromatic():
            return 'Aryl0'

    return 'NoAryl'

def detect_salt(smi):
    """Detect if molecule is a salt or charged species"""
    if '.' in smi:
        return True, 'Salt'
    mol = Chem.MolFromSmiles(smi)
    if mol:
        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() != 0:
                return True, 'Charged'
    return False, 'NoSalt'

def generate_new_rule(smi, chiral_idx):
    """Generate stereochemistry-invariant classification rule"""
    mol_stereo = Chem.MolFromSmiles(smi)
    if mol_stereo is None:
        return 'Invalid_SMILES'

    mol_rule = make_rule_mol(mol_stereo)

    is_salt, salt_type = detect_salt(smi)
    aryl = detect_aryl_category(mol_rule, chiral_idx)
    sub = detect_substituent(mol_rule, chiral_idx)
    ring_nb = detect_ring_neighbors(mol_rule, chiral_idx)

    parts = []
    if is_salt:
        parts.append('Salt')
    parts.append(aryl)
    parts.append(f'Sub_{sub}')
    parts.append(f'Ring_{ring_nb}')

    return '_'.join(parts)

# ============================================================================
# Main
# ============================================================================

def main():
    print('='*70)
    print('  Generating Stereochemistry-Invariant Classification Rules')
    print('='*70)

    # Load data
    df = pd.read_csv(os.path.join(BASE, 'scripts/processed_data_v2.csv'))
    mask = np.load(os.path.join(BASE, 'outputs/v24_audit/keep_mask_v24.npy'))
    print(f'  Loaded: {len(df)} samples, mask keeps {mask.sum()}')

    # Generate new rules
    print('\n  Generating new rules...')
    new_rules = []
    achiral_smiles_list = []

    for idx, row in df.iterrows():
        smi = row['smi']
        chiral_idx = int(row['chiral_idx'])
        new_rule = generate_new_rule(smi, chiral_idx)
        new_rules.append(new_rule)

        mol = Chem.MolFromSmiles(smi)
        if mol:
            mol_a = make_rule_mol(mol)
            achiral_smiles_list.append(Chem.MolToSmiles(mol_a))
        else:
            achiral_smiles_list.append('')

        if (idx+1) % 1000 == 0:
            print(f'    {idx+1}/{len(df)}...')

    df['new_rule'] = new_rules
    df['old_rule'] = df['classification']
    df['achiral_smiles'] = achiral_smiles_list
    print(f'  Unique new rules: {df["new_rule"].nunique()}')

    # Find enantiomer pairs
    print('\n  Finding enantiomer pairs...')
    achiral_groups = defaultdict(list)
    for idx, asmi in enumerate(achiral_smiles_list):
        if asmi:
            achiral_groups[asmi].append(idx)

    pairs = []
    for asmi, indices in achiral_groups.items():
        if len(indices) >= 2:
            for i in range(0, len(indices)-1, 2):
                pairs.append((indices[i], indices[i+1]))

    print(f'  Found {len(pairs)} enantiomer pairs')

    # Check consistency
    old_inconsistent = 0
    new_inconsistent = 0
    audit_rows = []

    for idx1, idx2 in pairs:
        old1, old2 = df.iloc[idx1]['old_rule'], df.iloc[idx2]['old_rule']
        new1, new2 = df.iloc[idx1]['new_rule'], df.iloc[idx2]['new_rule']
        achiral_same = (df.iloc[idx1]['achiral_smiles'] == df.iloc[idx2]['achiral_smiles'])
        old_ok = (old1 == old2)
        new_ok = (new1 == new2)

        if not old_ok: old_inconsistent += 1
        if not new_ok: new_inconsistent += 1

        audit_rows.append({
            'idx1': idx1, 'idx2': idx2,
            'smi1': df.iloc[idx1]['smi'], 'smi2': df.iloc[idx2]['smi'],
            'old_rule1': old1, 'old_rule2': old2,
            'new_rule1': new1, 'new_rule2': new2,
            'old_consistent': old_ok, 'new_consistent': new_ok,
            'achiral_same': achiral_same,
            'notes': '' if new_ok else ('structural_mismatch' if not achiral_same else 'BUG_NEEDS_FIX')
        })

    print(f'  Old rules inconsistent: {old_inconsistent}/{len(pairs)} ({100*old_inconsistent/max(len(pairs),1):.1f}%)')
    print(f'  New rules inconsistent: {new_inconsistent}/{len(pairs)} ({100*new_inconsistent/max(len(pairs),1):.1f}%)')

    # Key pairs check
    print('\n  === Key Enantiomer Pairs ===')
    key_pairs = [(5392,5393), (5372,5373), (1314,1315)]
    for a, b in key_pairs:
        if a < len(df) and b < len(df):
            r1, r2 = df.iloc[a], df.iloc[b]
            print(f'  #{a}/{b}:')
            print(f'    Old: {r1["old_rule"]} vs {r2["old_rule"]} -> {"SAME" if r1["old_rule"]==r2["old_rule"] else "DIFFERENT"}')
            print(f'    New: {r1["new_rule"]} vs {r2["new_rule"]} -> {"SAME" if r1["new_rule"]==r2["new_rule"] else "DIFFERENT"}')

    # Save outputs
    out_dir = os.path.join(BASE, 'scripts')

    # 1. Main CSV
    df.to_csv(os.path.join(out_dir, 'processed_data_v3_stereo_invariant_rules.csv'), index=False)
    print(f'\n  Saved: processed_data_v3_stereo_invariant_rules.csv')

    # 2. Rule stats
    rule_stats = []
    for rule in sorted(df['new_rule'].unique()):
        sub = df[df['new_rule']==rule]
        n = len(sub)
        orp = int((sub['OR']>0).sum())
        orn = int((sub['OR']<0).sum())
        conf = max(orp,orn)/n*100 if n>0 else 0
        direction = 'OR+' if orp>=orn else 'OR-'
        rule_stats.append({
            'rule':rule, 'n':n, 'OR+':orp, 'OR-':orn,
            'confidence':round(conf,1), 'direction':direction
        })

    pd.DataFrame(rule_stats).sort_values('n',ascending=False).to_csv(
        os.path.join(out_dir,'classification_rules_stereo_invariant.csv'), index=False)
    print(f'  Saved: classification_rules_stereo_invariant.csv')

    # 3. Audit
    pd.DataFrame(audit_rows).to_csv(
        os.path.join(out_dir,'enantiomer_rule_consistency_audit.csv'), index=False)
    print(f'  Saved: enantiomer_rule_consistency_audit.csv')

    # 4. Difficult cases
    diff_ids = [5392,5393,5372,5373,1314,1315,5374,5388]
    diff_rows = []
    for did in diff_ids:
        if did < len(df):
            r = df.iloc[did]
            diff_rows.append({
                'id':did, 'smi':r['smi'], 'OR':r['OR'], 'PM':r['PM'],
                'old_rule':r['old_rule'], 'new_rule':r['new_rule'],
                'achiral_smiles':r['achiral_smiles']
            })

    pd.DataFrame(diff_rows).to_csv(
        os.path.join(out_dir,'difficult_cases_rule_context_v2.csv'), index=False)
    print(f'  Saved: difficult_cases_rule_context_v2.csv')

    # 5. Summary
    summary = f"""# Rule Fix Summary

## Problem
Original classification rules were stereo-dependent: @/@@ in SMILES caused different
substituent detection for enantiomers, splitting them into different rules.

## Solution
Created stereochemistry-invariant rules using achiral molecular graph (mol_rule).
All rule generation now operates on molecules with stereochemistry removed.

## Results

### Enantiomer Consistency
- Total enantiomer pairs: {len(pairs)}
- Old rules inconsistent: {old_inconsistent}/{len(pairs)} ({100*old_inconsistent/max(len(pairs),1):.1f}%)
- New rules inconsistent: {new_inconsistent}/{len(pairs)} ({100*new_inconsistent/max(len(pairs),1):.1f}%)

### Key Pairs
"""

    for a, b in key_pairs:
        if a < len(df) and b < len(df):
            r1, r2 = df.iloc[a], df.iloc[b]
            same_new = r1['new_rule'] == r2['new_rule']
            summary += f"""
**#{a} / #{b}**
- Old: `{r1['old_rule']}` vs `{r2['old_rule']}` ({'same' if r1['old_rule']==r2['old_rule'] else 'DIFFERENT'})
- New: `{r1['new_rule']}` vs `{r2['new_rule']}` ({'SAME' if same_new else 'DIFFERENT'})
- Fixed: {'YES' if same_new and r1['old_rule']!=r2['old_rule'] else 'N/A (was already same)' if r1['old_rule']==r2['old_rule'] else 'NO - needs investigation'}
"""

    summary += f"""
### Impact on Model Features
Classification rules are NOT part of the 2,276-dimensional SuperLearner feature space.
They are used only for:
- Classification explorer (HTML)
- Rule-level error analysis
- Difficult sample interpretation

**No model retraining is required.**

### Unique Rules
- Old unique rules: {df['old_rule'].nunique()}
- New unique rules: {df['new_rule'].nunique()}

### Files Generated
1. processed_data_v3_stereo_invariant_rules.csv
2. classification_rules_stereo_invariant.csv
3. enantiomer_rule_consistency_audit.csv
4. difficult_cases_rule_context_v2.csv
5. rule_fix_summary.md
"""

    with open(os.path.join(out_dir, 'rule_fix_summary.md'), 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f'  Saved: rule_fix_summary.md')

    print('\n' + '='*70)
    print('  DONE')
    print('='*70)

if __name__ == '__main__':
    main()
