#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OR Sign Conflict & Enantiomer Audit Script
===========================================
全数据集审计: 溶剂效应 vs 对映体混淆 vs 真正数据错误

输出:
  outputs/reports/audit_or_conflicts_report.txt  — 详细文本报告
  outputs/reports/audit_or_conflicts.csv         — 结构化CSV (可导入Excel)

Usage: python scripts/audit_or_conflicts.py
"""
import warnings; warnings.filterwarnings('ignore')
import csv, json, os, sys
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE / 'outputs' / 'reports'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("  OR Sign Conflict & Enantiomer Audit")
print("=" * 70)

pdf = {}
with open(BASE / 'scripts' / 'processed_data.csv', 'r', encoding='utf-8-sig') as f:
    for i, r in enumerate(csv.DictReader(f)): pdf[i] = r
N = len(pdf)
print(f"  Dataset: {N} samples")

# ref_db
ref_db = {}
for p in [Path('/root/pm_v15/data/ref_database_cache.json'),
          Path(r'C:\Users\lenovo\.claude\projects\PM\data\ref_database_cache.json'),
          BASE / 'data' / 'ref_database_cache.json']:
    if p.exists():
        with open(p, 'r', encoding='utf-8') as f: ref_db = json.load(f)
        print(f"  ref_db: {len(ref_db)} SMILES entries")
        break

# V23 predictions
v23_pred, prob_cols = {}, []
V23_CSV = REPORT_DIR / 'v23_superlearner_predictions.csv'
if V23_CSV.exists():
    with open(V23_CSV, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            v23_pred[int(r['sample_id'])] = r
            if not prob_cols: prob_cols = [k for k in r.keys() if k.startswith('prob_')]
n_models = len(prob_cols)
v23_dis_map = {}
for sid, r in v23_pred.items():
    yt = int(r['y_true'])
    v23_dis_map[sid] = sum(1 for pc in prob_cols if (1 if float(r[pc]) > 0.5 else 0) != yt)
print(f"  V23: {len(v23_pred)} samples, {n_models} models")

# Masks
keep_v21 = np.load(BASE / 'data' / 'v21_cleaned' / 'keep_mask_v21.npy')
keep_v22 = np.load(BASE / 'data' / 'v22_final' / 'keep_mask_v22.npy')
y_v18 = np.load(BASE / 'data' / 'v18_corrected' / 'y_full.npy').astype(int)
y_v21 = np.load(BASE / 'data' / 'v21_cleaned' / 'y_v21.npy').astype(int)

# ═══════════════════════════════════════════════════════════
# 2. BUILD GROUPS & DETECT CONFLICTS
# ═══════════════════════════════════════════════════════════
smi_groups = defaultdict(list)
for i, pr in pdf.items():
    smi_groups[pr['smi']].append((i, float(pr['OR']), pr.get('solvent', '')))

print(f"  Unique SMILES: {len(smi_groups)}")

# Find all sign conflicts (same SMILES, both OR>0 and OR<=0)
sign_conflicts = []
for smi, group in smi_groups.items():
    if len(group) < 2: continue
    pos = [g for g in group if g[1] > 0]
    neg = [g for g in group if g[1] <= 0]
    if not pos or not neg: continue
    sign_conflicts.append((smi, group, pos, neg))

print(f"  SMILES with sign conflicts: {len(sign_conflicts)}")

# ═══════════════════════════════════════════════════════════
# 3. ENANTIOMER DETECTION
# ═══════════════════════════════════════════════════════════
print("\n  Scanning for enantiomer pairs (@<->@@)...")

def swap_chirality(smi):
    """Swap @ and @@ in SMILES to get potential enantiomer."""
    return smi.replace('@@', '\x00').replace('@', '@@').replace('\x00', '@')

enantiomer_pairs = []
checked = set()
for smi in smi_groups:
    if smi in checked: continue
    if '@' not in smi: continue
    enan_smi = swap_chirality(smi)
    if enan_smi != smi and enan_smi in smi_groups:
        enantiomer_pairs.append((smi, enan_smi))
        checked.add(smi)
        checked.add(enan_smi)

print(f"  Enantiomer pairs found: {len(enantiomer_pairs)}")

# ═══════════════════════════════════════════════════════════
# 4. CLASSIFY EACH SIGN CONFLICT
# ═══════════════════════════════════════════════════════════
print("\n  Classifying sign conflicts...")

# Cat C indices (8/8 disagree + |OR|>=5 + opposite records)
cat_c_idx = set()
for sid in v23_pred:
    if v23_dis_map[sid] < n_models: continue
    ov = float(pdf[sid]['OR'])
    if abs(ov) < 5: continue
    smi = pdf[sid]['smi']
    group = smi_groups[smi]
    cs = '+' if ov > 0 else '-'
    opp = sum(1 for i,v,_ in group if (v<=0 if cs=='+' else v>0) and i!=sid)
    if opp > 0: cat_c_idx.add(sid)

# Enantiomer SMILES lookup
enan_map = {}
for s1, s2 in enantiomer_pairs:
    enan_map[s1] = s2
    enan_map[s2] = s1

results = []
for smi, group, pos, neg in sign_conflicts:
    # Solvent analysis
    sol_or = defaultdict(list)
    for idx, ov, sol in group:
        sol_or[sol].append((idx, ov))

    n_solvents = len(sol_or)

    # Intra-solvent conflict?
    intra_solvents = []
    for sol, recs in sol_or.items():
        signs = set(1 if v > 0 else 0 for _, v in recs)
        if len(signs) > 1:
            intra_solvents.append(sol)

    # Has enantiomer in dataset?
    has_enantiomer = smi in enan_map
    enan_smi = enan_map.get(smi, '')
    enan_group = smi_groups.get(enan_smi, []) if enan_smi else []

    # Any record in Cat C?
    catc_records = [idx for idx, _, _ in group if idx in cat_c_idx]

    # Classify
    if intra_solvents:
        conflict_type = 'INTRA_SOLVENT'
        severity = 'P0_CRITICAL'
        action = 'CHECK_LITERATURE'
    elif n_solvents >= 2 and not intra_solvents:
        conflict_type = 'CROSS_SOLVENT'
        severity = 'P3_LOW'
        action = 'LIKELY_SOLVENT_EFFECT'
    else:
        conflict_type = 'SINGLE_SOLVENT_MIXED'
        severity = 'P1_HIGH'
        action = 'CHECK_LITERATURE'

    # Ref analysis
    refs = ref_db.get(smi, [])
    ref_cas = set()
    ref_names = set()
    for r in refs:
        c = str(r.get('cas','')).strip()
        if c and c != 'nan': ref_cas.add(c)
        n = str(r.get('name','')).strip()
        if n and n != 'nan': ref_names.add(n.split(';')[0].strip())

    # V23 model info per record
    rec_details = []
    for idx, ov, sol in sorted(group, key=lambda x: x[0]):
        v23r = v23_pred.get(idx)
        ep = float(v23r['ensemble_prob']) if v23r else -1
        ms = '+' if ep > 0.5 else ('-' if ep >= 0 else '?')
        mc = round(abs(ep - 0.5) * 2, 3) if ep >= 0 else 0
        dis = v23_dis_map.get(idx, -1)
        in_catc = idx in cat_c_idx
        rec_details.append({
            'idx': idx, 'OR': ov, 'solvent': sol,
            'model_sign': ms, 'model_conf': mc, 'disagree': dis,
            'in_v22': bool(keep_v22[idx]), 'in_catc': in_catc
        })

    results.append({
        'smi': smi, 'n_records': len(group), 'n_pos': len(pos), 'n_neg': len(neg),
        'n_solvents': n_solvents, 'conflict_type': conflict_type,
        'severity': severity, 'action': action,
        'intra_solvents': intra_solvents,
        'has_enantiomer': has_enantiomer, 'enan_smi': enan_smi,
        'enan_n_records': len(enan_group),
        'catc_records': catc_records,
        'ref_count': len(refs), 'cas': '; '.join(sorted(ref_cas)[:3]),
        'name': '; '.join(sorted(ref_names)[:2]),
        'records': rec_details
    })

results.sort(key=lambda x: (
    {'P0_CRITICAL':0,'P1_HIGH':1,'P2_MEDIUM':2,'P3_LOW':3}.get(x['severity'],9),
    -x['n_records']
))

# ═══════════════════════════════════════════════════════════
# 5. ENANTIOMER PAIR DETAILED ANALYSIS
# ═══════════════════════════════════════════════════════════
print("  Analyzing enantiomer pairs...")

enan_analysis = []
for smi_a, smi_b in enantiomer_pairs:
    ga = smi_groups[smi_a]
    gb = smi_groups[smi_b]
    refs_a = ref_db.get(smi_a, [])
    refs_b = ref_db.get(smi_b, [])

    cas_a = set()
    cas_b = set()
    name_a = set()
    name_b = set()
    for r in refs_a:
        c = str(r.get('cas','')).strip()
        if c and c != 'nan': cas_a.add(c)
        n = str(r.get('name','')).strip()
        if n and n != 'nan': name_a.add(n.split(';')[0].strip())
    for r in refs_b:
        c = str(r.get('cas','')).strip()
        if c and c != 'nan': cas_b.add(c)
        n = str(r.get('name','')).strip()
        if n and n != 'nan': name_b.add(n.split(';')[0].strip())

    # Check if OR signs are consistent within each enantiomer
    or_a = [v for _,v,_ in ga]
    or_b = [v for _,v,_ in gb]
    a_signs = set(1 if v > 0 else 0 for v in or_a)
    b_signs = set(1 if v > 0 else 0 for v in or_b)

    a_consistent = len(a_signs) == 1
    b_consistent = len(b_signs) == 1

    # Expected: enantiomers should have opposite OR signs
    if a_consistent and b_consistent:
        a_pos = 1 in a_signs
        b_pos = 1 in b_signs
        if a_pos != b_pos:
            pair_status = 'CORRECT_ENANTIOMERS'
        else:
            pair_status = 'SUSPICIOUS_SAME_SIGN'
    elif not a_consistent or not b_consistent:
        pair_status = 'MIXED_SIGNS_WITHIN'
    else:
        pair_status = 'UNKNOWN'

    enan_analysis.append({
        'smi_a': smi_a, 'smi_b': smi_b,
        'n_a': len(ga), 'n_b': len(gb),
        'or_a': or_a, 'or_b': or_b,
        'cas_a': '; '.join(sorted(cas_a)), 'cas_b': '; '.join(sorted(cas_b)),
        'name_a': '; '.join(sorted(name_a)[:2]), 'name_b': '; '.join(sorted(name_b)[:2]),
        'a_consistent': a_consistent, 'b_consistent': b_consistent,
        'pair_status': pair_status,
        'records_a': [(i, v, s) for i,v,s in ga],
        'records_b': [(i, v, s) for i,v,s in gb],
    })

# ═══════════════════════════════════════════════════════════
# 6. WRITE TEXT REPORT
# ═══════════════════════════════════════════════════════════
rpt_path = REPORT_DIR / 'audit_or_conflicts_report.txt'
with open(rpt_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("  OR SIGN CONFLICT & ENANTIOMER AUDIT REPORT\n")
    f.write("=" * 80 + "\n\n")

    # Summary
    n_intra = sum(1 for r in results if r['conflict_type'] == 'INTRA_SOLVENT')
    n_cross = sum(1 for r in results if r['conflict_type'] == 'CROSS_SOLVENT')
    n_single = sum(1 for r in results if r['conflict_type'] == 'SINGLE_SOLVENT_MIXED')
    n_with_enan = sum(1 for r in results if r['has_enantiomer'])
    n_with_catc = sum(1 for r in results if r['catc_records'])

    f.write(f"SUMMARY\n")
    f.write(f"  Total dataset: {N} samples, {len(smi_groups)} unique SMILES\n")
    f.write(f"  SMILES with sign conflicts: {len(results)}\n")
    f.write(f"    - Intra-solvent (P0 Critical): {n_intra}\n")
    f.write(f"    - Cross-solvent only (P3 Low, likely solvent effect): {n_cross}\n")
    f.write(f"    - Single solvent mixed (P1 High): {n_single}\n")
    f.write(f"  With enantiomer (@<->@@) in dataset: {n_with_enan}\n")
    f.write(f"  With former Cat C records (8/8 disagree): {n_with_catc}\n")
    f.write(f"  Enantiomer pairs detected: {len(enantiomer_pairs)}\n\n")

    # Section A: Former Cat C detailed analysis
    f.write("=" * 80 + "\n")
    f.write("  SECTION A: FORMER CAT C SAMPLES (reclassified)\n")
    f.write("=" * 80 + "\n\n")

    catc_results = [r for r in results if r['catc_records']]
    for r in catc_results:
        f.write(f"--- {r['smi'][:70]} ---\n")
        f.write(f"  CAS: {r['cas']}  Name: {r['name']}\n")
        f.write(f"  Conflict type: {r['conflict_type']}  Severity: {r['severity']}\n")
        f.write(f"  Records: {r['n_records']} ({r['n_pos']}+ / {r['n_neg']}-), {r['n_solvents']} solvents\n")
        f.write(f"  Has enantiomer: {r['has_enantiomer']}")
        if r['has_enantiomer']:
            f.write(f" ({r['enan_n_records']} records)")
        f.write(f"\n  Cat C record indices: {r['catc_records']}\n")
        f.write(f"  Literature: {r['ref_count']} entries\n")
        for rec in r['records']:
            tag = " <-- 8/8 DISAGREE (former Cat C)" if rec['in_catc'] else ""
            f.write(f"    #{rec['idx']}: OR={rec['OR']:+.1f} sol={rec['solvent']} "
                    f"model=({rec['model_sign']}) conf={rec['model_conf']:.0%} "
                    f"dis={rec['disagree']}/{n_models} "
                    f"{'V22' if rec['in_v22'] else 'Removed'}{tag}\n")

        # Diagnosis
        if r['conflict_type'] == 'CROSS_SOLVENT' and not r['intra_solvents']:
            f.write(f"  DIAGNOSIS: Solvent-dependent OR sign reversal. "
                    f"All opposite-sign records are in different solvents. "
                    f"This is a known physical phenomenon, NOT a label error.\n")
        if r['has_enantiomer']:
            f.write(f"  WARNING: Enantiomer exists in dataset. "
                    f"Verify that records are correctly assigned to R/S configurations.\n")
        f.write("\n")

    # Section B: Enantiomer pairs
    f.write("=" * 80 + "\n")
    f.write("  SECTION B: ENANTIOMER PAIRS (@<->@@)\n")
    f.write("=" * 80 + "\n\n")

    for ea in enan_analysis:
        f.write(f"--- Pair ---\n")
        f.write(f"  A (@@): {ea['smi_a'][:70]}\n")
        f.write(f"  B (@):  {ea['smi_b'][:70]}\n")
        f.write(f"  CAS A: {ea['cas_a']}  Name A: {ea['name_a']}\n")
        f.write(f"  CAS B: {ea['cas_b']}  Name B: {ea['name_b']}\n")
        f.write(f"  Status: {ea['pair_status']}\n")
        f.write(f"  A records ({ea['n_a']}): OR = {[f'{v:+.1f}' for v in ea['or_a']]}\n")
        f.write(f"  B records ({ea['n_b']}): OR = {[f'{v:+.1f}' for v in ea['or_b']]}\n")
        f.write(f"  A consistent sign: {ea['a_consistent']}  B consistent sign: {ea['b_consistent']}\n")

        f.write(f"  A details:\n")
        for i, v, s in ea['records_a']:
            dis = v23_dis_map.get(i, -1)
            f.write(f"    #{i}: OR={v:+.1f} sol={s} dis={dis}/{n_models} v22={bool(keep_v22[i])}\n")
        f.write(f"  B details:\n")
        for i, v, s in ea['records_b']:
            dis = v23_dis_map.get(i, -1)
            f.write(f"    #{i}: OR={v:+.1f} sol={s} dis={dis}/{n_models} v22={bool(keep_v22[i])}\n")

        if ea['pair_status'] == 'MIXED_SIGNS_WITHIN':
            f.write(f"  ACTION REQUIRED: One or both enantiomers have mixed OR signs. "
                    f"Check if records are correctly assigned to R/S configurations.\n")
        elif ea['pair_status'] == 'SUSPICIOUS_SAME_SIGN':
            f.write(f"  ACTION REQUIRED: Both enantiomers show same OR sign direction. "
                    f"This is physically unexpected — verify chirality assignment.\n")
        f.write("\n")

    # Section C: Intra-solvent conflicts (most critical)
    f.write("=" * 80 + "\n")
    f.write("  SECTION C: INTRA-SOLVENT CONFLICTS (P0 Critical)\n")
    f.write("=" * 80 + "\n\n")

    intra_results = [r for r in results if r['conflict_type'] == 'INTRA_SOLVENT']
    for r in intra_results:
        f.write(f"--- {r['smi'][:70]} ---\n")
        f.write(f"  CAS: {r['cas']}  Name: {r['name']}\n")
        f.write(f"  Intra-solvent conflicts in: {r['intra_solvents']}\n")
        f.write(f"  Has enantiomer: {r['has_enantiomer']}\n")
        for rec in r['records']:
            f.write(f"    #{rec['idx']}: OR={rec['OR']:+.1f} sol={rec['solvent']} "
                    f"model=({rec['model_sign']}) conf={rec['model_conf']:.0%} "
                    f"dis={rec['disagree']}/{n_models}\n")
        f.write("\n")

    # Section D: All cross-solvent (for reference)
    f.write("=" * 80 + "\n")
    f.write(f"  SECTION D: CROSS-SOLVENT ONLY ({n_cross} molecules, likely solvent effect)\n")
    f.write("=" * 80 + "\n\n")

    cross_results = [r for r in results if r['conflict_type'] == 'CROSS_SOLVENT']
    for r in cross_results:
        solvents = set()
        for rec in r['records']: solvents.add(rec['solvent'])
        f.write(f"  {r['smi'][:60]}  |  {r['n_pos']}+/{r['n_neg']}-  |  "
                f"solvents: {', '.join(sorted(solvents))}  |  "
                f"enan: {'Y' if r['has_enantiomer'] else 'N'}  |  "
                f"CAS: {r['cas'][:30]}\n")

    f.write(f"\n\nTotal: {len(results)} molecules with sign conflicts\n")

print(f"\n  Report: {rpt_path}")

# ═══════════════════════════════════════════════════════════
# 7. WRITE CSV
# ═══════════════════════════════════════════════════════════
csv_path = REPORT_DIR / 'audit_or_conflicts.csv'
with open(csv_path, 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['SMILES', 'n_records', 'n_pos', 'n_neg', 'n_solvents',
                'conflict_type', 'severity', 'action',
                'has_enantiomer', 'enan_n_records',
                'has_catc', 'intra_solvents',
                'ref_count', 'CAS', 'name',
                'record_details'])
    for r in results:
        details = '; '.join(
            f"#{rec['idx']}:OR={rec['OR']:+.1f}/sol={rec['solvent']}/model=({rec['model_sign']}){rec['model_conf']:.0%}/dis={rec['disagree']}/{n_models}"
            for rec in r['records']
        )
        w.writerow([
            r['smi'], r['n_records'], r['n_pos'], r['n_neg'], r['n_solvents'],
            r['conflict_type'], r['severity'], r['action'],
            r['has_enantiomer'], r['enan_n_records'],
            bool(r['catc_records']), ', '.join(r['intra_solvents']),
            r['ref_count'], r['cas'], r['name'],
            details
        ])

print(f"  CSV: {csv_path}")
print(f"  Size: {os.path.getsize(rpt_path)/1024:.1f} KB (report), {os.path.getsize(csv_path)/1024:.1f} KB (csv)")
print("  DONE")
