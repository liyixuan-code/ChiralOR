#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V24 Audit-Based Data Correction
================================
Apply manual audit results from 48 MIXED_SIGNS_WITHIN enantiomer pairs.

Actions:
  1. DELETE individual records (26 samples)
  2. DELETE entire enantiomer pairs (3 pairs, ~27 samples)
  3. FIX OR values (2 samples: Reaxys verified sign correction)
  4. FIX chirality assignment (3 samples: label flip)
  5. FLAG "cannot judge" records (metadata only, kept in dataset)

Input:  data/v22_final/   (keep_mask_v22.npy)
        data/v21_cleaned/ (y_v21.npy, weights_v21.npy)
        data/v18_corrected/ (X_2254_full.npy)
        v13_revised/v13_results/processed_data.csv

Output: data/v24_audit/   (keep_mask_v24, y_v24, weights_v24, X/y/w_kept,
                           audit_changelog.json, flagged_cannot_judge.json)

Usage:  python scripts/apply_audit_fixes.py
"""
import warnings; warnings.filterwarnings('ignore')
import csv, json, numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / 'data' / 'v24_audit'
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("  V24 Audit-Based Data Correction")
print("=" * 65)

# ================================================================
# 1. LOAD
# ================================================================
X_full  = np.load(BASE / 'data' / 'v18_corrected' / 'X_2254_full.npy')
y_v21   = np.load(BASE / 'data' / 'v21_cleaned' / 'y_v21.npy').astype(int)
w_v21   = np.load(BASE / 'data' / 'v21_cleaned' / 'weights_v21.npy')
mask_22 = np.load(BASE / 'data' / 'v22_final' / 'keep_mask_v22.npy').astype(bool)
N = len(y_v21)
print(f"Total samples: {N},  V22 kept: {mask_22.sum()}")

pdf = {}
with open(BASE / 'v13_revised' / 'v13_results' / 'processed_data.csv',
          'r', encoding='utf-8') as f:
    for i, row in enumerate(csv.DictReader(f)):
        pdf[i] = row
print(f"Loaded {len(pdf)} records from processed_data.csv")

# ================================================================
# 2. AUDIT CORRECTIONS  (indices = processed_data.csv row, 0-based)
# ================================================================

# ── 2a  DELETE individual records ──
DEL = {
    37:   "OR=+2.0, 数值太小考虑测试偏差",
    1288: "OR=-28.3, reaxys无此数据，归类错误",
    1305: "OR=+57.8, 与多条-57.8相反，作者混淆对映体",
    1561: "OR=-29.2, reaxys无此数据",
    1600: "OR=-30.5, reaxys无此数据",
    1708: "OR=+59.9, 作者实验问题",
    1794: "OR=-71.4, 温度/波长不一致",
    2345: "OR=+33.7, 原文无符号无法判断",
    2349: "OR=+69.9, 未注明R/S构型",
    2360: "OR=-42.0, 未注明R/S构型",
    2389: "OR=+7.1, reaxys未找到此数据",
    2405: "OR=-23.8, 浓度过低(0.1-0.17g/100ml)",
    2408: "OR=-23.8, 浓度过低",
    2409: "OR=-19.5, 浓度过低",
    2439: "OR=-21.0, 测试条件不全且年代久远",
    2445: "OR=+29.0, 测试条件不全且年代久远",
    2565: "OR=-64.0, 纯度太低",
    2706: "OR=-64.1, 纯度不高",
    2816: "OR=-2.8, 数值太小",
    2818: "OR=-2.8, 数值太小",
    2867: "OR=+27.4, 测试浓度过低",
    2916: "OR=-33.0, 可删除",
    3101: "OR=+6.5, 数值较小",
    3769: "OR=-8.6, 数值较小",
    4229: "OR=-4.0, 数值较小",
    4233: "OR=+3.0, 数值较小",
    4367: "OR=+1.1, 数值较小",
}

# ── 2b  DELETE entire enantiomer pairs ──
DEL_PAIRS = [
    ("diethylaminomethyl-benzodioxane",
     "reaxys命名歧义，未明确R/S构型",
     list(range(1561, 1567))),          # #1561-#1566
    ("2-(p-tolyl)-THQx",
     "明显不对称数据",
     list(range(2536, 2549))),          # #2536-#2548
    ("4-chlorophenyl-THQx",
     "建议删除此两个分子的数据",
     list(range(2935, 2943))),          # #2935-#2942
]

# ── 2c  FIX OR values (Reaxys verified) ──
FIX_OR = {
    1435: (-38.5, +45.3, "reaxys验证应为+45.3"),
    1436: (-30.7, +35.2, "reaxys验证应为+35.2"),
}

# ── 2d  FIX chirality (label flip) ──
FIX_CHIRAL = {
    228: "实则为S构型，翻转标签",
    263: "文献明确说明是R构型数据，翻转标签",
    487: "原文说明是R构型，reaxys下属于S构型错误，翻转标签",
}

# ── 2e  FLAG "cannot judge" (kept, metadata only) ──
FLAG = {
    1284: "OR=-67.8 vs +23.2 同溶剂chloroform，无法判断",
    1285: "OR=-13.1 vs +25.5 同溶剂ethanol，无法判断",
    1287: "OR=+25.5, 无法判断",
    2345: "OR=+33.7, 无法判断是否实验误差",
    2365: "OR=-69.8, methanol溶剂测试无对比结果",
    2390: "OR=+95.3, 文献无明显问题",
    2499: "OR=-61.6, 无法从文献判断",
    2528: "OR=+181.0, 无法从文献判断",
    2567: "OR=-46.2, 无法从文献判断",
    2568: "OR=-89.5, 无法从文献判断",
    2612: "OR=+86.3, 无法根据文献判断",
    2664: "OR=+80.0, 无法判断",
    2803: "OR=+3.7, 无法判断",
    2804: "OR=-18.1, 无法判断",
    2846: "OR=-66.0, 无法判断",
    2852: "OR=+44.5, 无法判断",
    3028: "OR=+49.7, 无法判断",
    3043: "OR=-23.0, 无法判断",
    3117: "OR=-47.3, 无法判断",
    3146: "OR=+31.6, 无法判断",
    3187: "OR, 无法判断",
    3188: "OR, 无法判断",
}

# ================================================================
# 3. APPLY
# ================================================================
mask_24 = mask_22.copy()
y_v24   = y_v21.copy()
w_v24   = w_v21.copy()
log     = []

def _info(idx):
    r = pdf.get(idx, {})
    return r.get('smi', '?'), r.get('OR', '?')

# ── 3a. Deletions ──
all_del = set(DEL.keys())
for name, reason, indices in DEL_PAIRS:
    all_del.update(indices)

n_new, n_old = 0, 0
for idx in sorted(all_del):
    smi, orv = _info(idx)
    was = bool(mask_24[idx])
    mask_24[idx] = False
    reason = DEL.get(idx, "part of deleted pair")
    log.append(dict(action='DELETE', index=idx, smi=smi, or_val=orv,
                    reason=reason, was_in_v22=was))
    if was:
        n_new += 1
    else:
        n_old += 1

print(f"\nDELETE: {len(all_del)} targets  (new={n_new}, already_gone={n_old})")

# ── 3b. Fix OR values ──
n_or = 0
for idx, (old_or, new_or, reason) in FIX_OR.items():
    old_y = y_v24[idx]
    new_y = 1 if new_or > 0 else 0
    y_v24[idx] = new_y
    changed = (old_y != new_y)
    if changed:
        n_or += 1
    smi, _ = _info(idx)
    log.append(dict(action='FIX_OR', index=idx, smi=smi,
                    old_or=old_or, new_or=new_or,
                    old_label=int(old_y), new_label=int(new_y),
                    reason=reason))
    print(f"  FIX_OR #{idx}: y {old_y}->{new_y}  (OR {old_or}->{new_or})")

# ── 3c. Fix chirality ──
n_ch = 0
for idx, reason in FIX_CHIRAL.items():
    old_y = y_v24[idx]
    new_y = 1 - old_y
    y_v24[idx] = new_y
    n_ch += 1
    smi, orv = _info(idx)
    log.append(dict(action='FIX_CHIRAL', index=idx, smi=smi, or_val=orv,
                    old_label=int(old_y), new_label=int(new_y), reason=reason))
    print(f"  FIX_CHIRAL #{idx}: y {old_y}->{new_y}  ({reason[:40]})")

# ================================================================
# 4. BUILD & SAVE
# ================================================================
X_kept = X_full[mask_24]
y_kept = y_v24[mask_24]
w_kept = w_v24[mask_24]

n22 = int(mask_22.sum())
n24 = int(mask_24.sum())

print(f"\n{'='*65}")
print(f"  V24 SUMMARY")
print(f"{'='*65}")
print(f"  V22 kept:           {n22}")
print(f"  Newly removed:      {n_new}")
print(f"  V24 kept:           {n24}")
print(f"  Labels flipped(OR): {n_or}")
print(f"  Labels flipped(chi):{n_ch}")
print(f"  Flagged(kept):      {len(FLAG)}")
print(f"  X_kept shape:       {X_kept.shape}")
print(f"  OR+ ratio:          {y_kept.mean():.4f}")
print(f"  Weighted (w<1):     {(w_kept < 1.0).sum()}")

np.save(OUT / 'keep_mask_v24.npy', mask_24)
np.save(OUT / 'y_v24.npy',         y_v24)
np.save(OUT / 'weights_v24.npy',   w_v24)
np.save(OUT / 'X_kept.npy',        X_kept)
np.save(OUT / 'y_kept.npy',        y_kept)
np.save(OUT / 'w_kept.npy',        w_kept)

class Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.bool_,)):    return bool(o)
        if isinstance(o, np.ndarray):     return o.tolist()
        return super().default(o)

with open(OUT / 'audit_changelog.json', 'w', encoding='utf-8') as f:
    json.dump(dict(
        version='V24_audit', timestamp=datetime.now().isoformat(),
        v22=n22, v24=n24, removed=n_new, or_fixed=n_or, chiral_fixed=n_ch,
        flagged=len(FLAG), changelog=log,
    ), f, indent=2, ensure_ascii=False, cls=Enc)

with open(OUT / 'flagged_cannot_judge.json', 'w', encoding='utf-8') as f:
    fl = []
    for idx, reason in FLAG.items():
        smi, orv = _info(idx)
        fl.append(dict(index=idx, smi=smi, or_val=orv, reason=reason,
                       in_v24=bool(mask_24[idx])))
    json.dump(fl, f, indent=2, ensure_ascii=False, cls=Enc)

print(f"\n  Saved to {OUT}")
print(f"{'='*65}")
print(f"  DONE. Ready for V24 retraining.")
print(f"{'='*65}")