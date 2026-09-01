#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V23/V24 数据流水线完整审计脚本
逐层追踪每个数字的来源，输出详细报告
Usage: python scripts/audit_pipeline.py
"""
import json, csv, sys
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

BASE = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE / 'outputs' / 'reports'
OOF_DIR    = BASE / 'outputs' / 'probs' / 'v19_oof'
V21_DIR    = BASE / 'data' / 'v21_cleaned'
V22_DIR    = BASE / 'data' / 'v22_final'
V18C       = BASE / 'data' / 'v18_corrected'

REF_PATH = None
for p in [Path('/root/pm_v15/data/ref_database_cache.json'),
          BASE / 'data' / 'ref_database_cache.json',
          Path(r'C:\Users\lenovo\.claude\projects\PM\data\ref_database_cache.json')]:
    if p.exists(): REF_PATH = p; break

S = "=" * 70
def sec(t): print(f"\n{S}\n  {t}\n{S}")

# ═══════════════════════════════════════════════════════════════
sec("LAYER 1: 原始数据集")
# ═══════════════════════════════════════════════════════════════
pdf = {}
with open(BASE / 'scripts' / 'processed_data.csv', 'r', encoding='utf-8-sig') as f:
    for i, r in enumerate(csv.DictReader(f)): pdf[i] = r
N = len(pdf)
or_vals = [float(pdf[i]['OR']) for i in range(N)]
print(f"  总样本: {N}")
print(f"  OR>0: {sum(1 for v in or_vals if v>0)}, OR<=0: {sum(1 for v in or_vals if v<=0)}")
print(f"  |OR|<5: {sum(1 for v in or_vals if abs(v)<5)}")
print(f"  |OR|<3: {sum(1 for v in or_vals if abs(v)<3)}")
print(f"  |OR|<10: {sum(1 for v in or_vals if abs(v)<10)}")

smi_groups = defaultdict(list)
for i, pr in pdf.items():
    smi_groups[pr['smi']].append((i, float(pr['OR']), pr.get('solvent','')))
print(f"  唯一SMILES: {len(smi_groups)}, 多记录SMILES: {sum(1 for g in smi_groups.values() if len(g)>1)}")

# ═══════════════════════════════════════════════════════════════
sec("LAYER 2: 文献数据库 ref_db")
# ═══════════════════════════════════════════════════════════════
ref_db = {}
if REF_PATH:
    with open(REF_PATH, 'r', encoding='utf-8') as f: ref_db = json.load(f)
    print(f"  ref_db SMILES条目: {len(ref_db)}")
    # 统计有多少 processed_data 的 SMILES 在 ref_db 中有记录
    n_in_ref = sum(1 for smi in smi_groups if smi in ref_db)
    print(f"  数据集SMILES在ref_db中有记录: {n_in_ref}/{len(smi_groups)}")
else:
    print("  [MISSING] ref_database_cache.json")

# ═══════════════════════════════════════════════════════════════
sec("LAYER 3: V18→V21 标签翻转 (Cat A)")
# ═══════════════════════════════════════════════════════════════
y_v18 = np.load(V18C / 'y_full.npy').astype(int)
y_v21 = np.load(V21_DIR / 'y_v21.npy').astype(int)
flipped_idx = np.where(y_v18 != y_v21)[0]
print(f"  y_v18: shape={y_v18.shape}, sum={y_v18.sum()}")
print(f"  y_v21: shape={y_v21.shape}, sum={y_v21.sum()}")
print(f"  翻转样本数 (Cat A): {len(flipped_idx)}")
for idx in flipped_idx:
    pr = pdf[idx]
    print(f"    idx={idx}: OR={pr['OR']}, y: {y_v18[idx]}→{y_v21[idx]}, smi={pr['smi'][:50]}")

# ═══════════════════════════════════════════════════════════════
sec("LAYER 4: V21 MGA JSON 审计")
# ═══════════════════════════════════════════════════════════════
with open(REPORT_DIR / 'v21_model_guided_analysis.json', 'r', encoding='utf-8') as f:
    mga = json.load(f)

print("  --- Header 声明 ---")
hdr = {k: mga.get(k, '?') for k in ['cat_a_flip','cat_b_small_or','cat_c_partial','cat_d_suspect','total_candidates']}
for k, v in hdr.items(): print(f"    {k}: {v}")
hdr_sum = mga.get('cat_a_flip',0)+mga.get('cat_b_small_or',0)+mga.get('cat_c_partial',0)+mga.get('cat_d_suspect',0)
print(f"    A+B+C+D = {hdr_sum}, total = {mga.get('total_candidates',0)}")
if hdr_sum != mga.get('total_candidates',0):
    print(f"    ⚠️ 不一致! {hdr_sum} ≠ {mga.get('total_candidates',0)}")

print("\n  --- Detail 数组 ---")
for k in ['cat_a_details','cat_b_details','cat_c_details','cat_d_details']:
    arr = mga.get(k, [])
    print(f"    {k}: {len(arr)} 条")
    if arr and len(arr) <= 3:
        for s in arr: print(f"      {s}")

# Cat A details 的 idx
cat_a_mga_idx = {s.get('idx', s.get('index')) for s in mga.get('cat_a_details', [])}
cat_b_mga_idx = {s.get('idx', s.get('index')) for s in mga.get('cat_b_details', [])}
cat_c_mga_idx = {s.get('idx', s.get('index')) for s in mga.get('cat_c_details', [])}
cat_d_mga_idx = {s.get('idx', s.get('index')) for s in mga.get('cat_d_details', [])}
print(f"\n  Cat A idx: {sorted(cat_a_mga_idx)}")
print(f"  Cat A idx vs flipped_idx 交集: {sorted(cat_a_mga_idx & set(flipped_idx.tolist()))}")

# ═══════════════════════════════════════════════════════════════
sec("LAYER 5: V21 Cleaning Report")
# ═══════════════════════════════════════════════════════════════
cp = REPORT_DIR / 'v21_deep_cleaning_report.json'
if cp.exists():
    with open(cp, 'r', encoding='utf-8') as f: cr = json.load(f)
    for k, v in cr.items():
        if isinstance(v, (int, float, str, bool)):
            print(f"    {k}: {v}")
        elif isinstance(v, list):
            print(f"    {k}: list[{len(v)}]")
        elif isinstance(v, dict):
            print(f"    {k}: {v}")
else:
    print("  [MISSING]")

# ═══════════════════════════════════════════════════════════════
sec("LAYER 6: V21/V22 Mask & Weights")
# ═══════════════════════════════════════════════════════════════
keep_v21 = np.load(V21_DIR / 'keep_mask_v21.npy')
keep_v22 = np.load(V22_DIR / 'keep_mask_v22.npy')
weights  = np.load(V21_DIR / 'weights_v21.npy')

print(f"  V21 kept: {keep_v21.sum()}, removed: {N - keep_v21.sum()}")
print(f"  V22 kept: {keep_v22.sum()}, removed: {N - keep_v22.sum()}")
print(f"  V21→V22 额外移除: {keep_v21.sum() - keep_v22.sum()}")

# Weight 分布
wc = Counter(round(float(w), 2) for w in weights)
print(f"  Weight 分布:")
for wv in sorted(wc.keys()): print(f"    w={wv}: {wc[wv]}")

# V21→V22 差异
v21_not_v22 = set(np.where(keep_v21 & ~keep_v22)[0].tolist())
print(f"  V21保留但V22移除: {len(v21_not_v22)} 个, idx={sorted(v21_not_v22)}")
print(f"  与 cat_d_details idx 交集: {len(v21_not_v22 & cat_d_mga_idx)}")

# ═══════════════════════════════════════════════════════════════
sec("LAYER 7: V19 OOF 33模型 (V21 MGA 的基础)")
# ═══════════════════════════════════════════════════════════════
oof_files = sorted(OOF_DIR.glob('*.npy')) if OOF_DIR.exists() else []
print(f"  OOF 文件数: {len(oof_files)}")
if oof_files:
    # 用 V21 的 OOF 模型重新计算 MGA — 只保留 shape=(N,) 的文件
    all_probs = []
    skipped = []
    for fp in oof_files:
        prob = np.load(fp)
        print(f"    {fp.name}: shape={prob.shape}")
        if prob.shape == (N,):
            all_probs.append(prob)
        else:
            skipped.append((fp.name, prob.shape))
    if skipped:
        print(f"  ⚠️ 跳过 {len(skipped)} 个 shape 不匹配的文件:")
        for name, shape in skipped:
            print(f"      {name}: {shape} (期望 ({N},))")
    all_probs = np.stack(all_probs)  # (n_models, N)
    n_v19_models = len(all_probs)
    print(f"  有效模型数: {n_v19_models}, 样本数: {all_probs.shape[1]}")

    # 对每个样本计算 disagree 数
    # y_v18 是原始标签 (V21 MGA 用的是 V18 标签做比较)
    disagree_counts_v18 = np.zeros(N, dtype=int)
    for m in range(n_v19_models):
        preds = (all_probs[m] > 0.5).astype(int)
        disagree_counts_v18 += (preds != y_v18)

    # 分布
    print(f"\n  V19 OOF 对 y_v18 的 disagree 分布:")
    dc = Counter(disagree_counts_v18.tolist())
    for d in sorted(dc.keys()):
        pct = d / n_v19_models * 100
        marker = " ← ≥90% threshold" if d >= int(n_v19_models * 0.9) else ""
        print(f"    {d}/{n_v19_models} ({pct:.1f}%): {dc[d]} 个样本{marker}")

    thresh_90 = int(n_v19_models * 0.9)
    n_above_90 = sum(dc[d] for d in dc if d >= thresh_90)
    print(f"\n  ≥90% ({thresh_90}/{n_v19_models}) disagree 的样本: {n_above_90}")

    # 在 keep_v21=True 的样本中
    kept_mask = keep_v21.astype(bool)
    n_above_90_kept = int((disagree_counts_v18[kept_mask] >= thresh_90).sum())
    n_above_90_removed = int((disagree_counts_v18[~kept_mask] >= thresh_90).sum())
    print(f"    其中 V21 kept: {n_above_90_kept}")
    print(f"    其中 V21 removed (Plan A): {n_above_90_removed}")

    # 重新分类 (模拟 V21 MGA 逻辑)
    print(f"\n  --- 重新模拟 V21 MGA 分类 (≥{thresh_90}/{n_v19_models} disagree) ---")
    re_a, re_b, re_c, re_d = [], [], [], []
    for i in range(N):
        if disagree_counts_v18[i] < thresh_90: continue
        smi = pdf[i]['smi']; ov = float(pdf[i]['OR'])
        cs = '+' if ov > 0 else '-'
        # 文献证据
        refs = ref_db.get(smi, [])
        opp_refs = 0
        same_refs = 0
        for rec in refs:
            rv = rec.get('or_val', 0)
            if rv is None: continue
            try: rv = float(rv)
            except: continue
            if rv == 0: continue
            ref_sign = '+' if rv > 0 else '-'
            if ref_sign != cs: opp_refs += 1
            else: same_refs += 1
        # 数据集内部证据
        group = smi_groups[smi]
        po = sum(1 for j,v,_ in group if v > 0 and j != i)
        no = sum(1 for j,v,_ in group if v <= 0 and j != i)
        opp_data = no if cs == '+' else po
        same_data = po if cs == '+' else no
        total_opp = opp_refs + opp_data
        total_same = same_refs + same_data

        if total_opp > total_same and total_opp >= 2:
            re_a.append(i)
        elif abs(ov) < 5:
            re_b.append(i)
        elif total_opp > 0:
            re_c.append(i)
        else:
            re_d.append(i)

    print(f"    重算 Cat A: {len(re_a)} (MGA header: {mga.get('cat_a_flip',0)})")
    print(f"    重算 Cat B: {len(re_b)} (MGA header: {mga.get('cat_b_small_or',0)})")
    print(f"    重算 Cat C: {len(re_c)} (MGA header: {mga.get('cat_c_partial',0)})")
    print(f"    重算 Cat D: {len(re_d)} (MGA header: {mga.get('cat_d_suspect',0)})")
    print(f"    重算 Total: {len(re_a)+len(re_b)+len(re_c)+len(re_d)} (MGA header: {mga.get('total_candidates',0)})")
    print(f"    Cat A idx: {sorted(re_a)}")
    print(f"    Cat B idx (前20): {sorted(re_b)[:20]}...")
    print(f"    Cat C idx: {sorted(re_c)}")

# ═══════════════════════════════════════════════════════════════
sec("LAYER 8: V23 SuperLearner 预测分析")
# ═══════════════════════════════════════════════════════════════
V23_CSV = REPORT_DIR / 'v23_superlearner_predictions.csv'
v23_pred = {}
prob_cols = []
if V23_CSV.exists():
    with open(V23_CSV, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            v23_pred[int(r['sample_id'])] = r
            if not prob_cols:
                prob_cols = [k for k in r.keys() if k.startswith('prob_')]
print(f"  V23 预测样本数: {len(v23_pred)}")
print(f"  V23 模型数: {len(prob_cols)}")
print(f"  模型列名: {prob_cols}")

# 计算每个样本的 disagree 数
v23_disagree = {}
for sid, r in v23_pred.items():
    yt = int(r['y_true'])
    n_dis = sum(1 for pc in prob_cols if (1 if float(r[pc]) > 0.5 else 0) != yt)
    v23_disagree[sid] = n_dis

# disagree 分布
n_v23_models = len(prob_cols)
print(f"\n  V23 8模型 disagree 分布 (对 y_true):")
dc23 = Counter(v23_disagree.values())
for d in sorted(dc23.keys()):
    pct = d / n_v23_models * 100
    print(f"    {d}/{n_v23_models} ({pct:.1f}%): {dc23[d]} 个样本")

# 不同阈值下的 QC 候选数
print(f"\n  --- 不同阈值下的 QC 候选数 ---")
for thresh in [5, 6, 7, 8]:
    n_cand = sum(1 for d in v23_disagree.values() if d >= thresh)
    pct = thresh / n_v23_models * 100
    print(f"    ≥{thresh}/{n_v23_models} ({pct:.1f}%): {n_cand} 个样本")

# ═══════════════════════════════════════════════════════════════
sec("LAYER 9: V23 QC 分类 (当前 gen_html_v4.py 的逻辑, 8/8)")
# ═══════════════════════════════════════════════════════════════
# 完全复现 gen_html_v4.py 的分类逻辑
cat_b_8, cat_c_8, cat_d_8 = [], [], []
for sid, r in v23_pred.items():
    if v23_disagree[sid] < n_v23_models: continue  # 8/8
    pr = pdf[sid]; smi = pr['smi']; ov = float(pr['OR'])
    group = smi_groups[smi]; cs = '+' if ov > 0 else '-'
    po = sum(1 for j,v,_ in group if v > 0 and j != sid)
    no = sum(1 for j,v,_ in group if v <= 0 and j != sid)
    opp = no if cs == '+' else po
    if abs(ov) < 5: cat_b_8.append(sid)
    elif opp > 0: cat_c_8.append(sid)
    else: cat_d_8.append(sid)

print(f"  阈值 8/8 (100%):")
print(f"    Cat B (|OR|<5): {len(cat_b_8)}")
print(f"    Cat C (opp>0): {len(cat_c_8)}")
print(f"    Cat D (其余): {len(cat_d_8)}")
print(f"    Total B+C+D: {len(cat_b_8)+len(cat_c_8)+len(cat_d_8)}")
print(f"    + Cat A (历史翻转): {len(flipped_idx)}")
print(f"    Grand Total: {len(cat_b_8)+len(cat_c_8)+len(cat_d_8)+len(flipped_idx)}")

# ═══════════════════════════════════════════════════════════════
sec("LAYER 10: V23 QC 分类 (≥7/8 = 87.5%)")
# ═══════════════════════════════════════════════════════════════
cat_b_7, cat_c_7, cat_d_7 = [], [], []
for sid, r in v23_pred.items():
    if v23_disagree[sid] < 7: continue  # ≥7/8
    pr = pdf[sid]; smi = pr['smi']; ov = float(pr['OR'])
    group = smi_groups[smi]; cs = '+' if ov > 0 else '-'
    po = sum(1 for j,v,_ in group if v > 0 and j != sid)
    no = sum(1 for j,v,_ in group if v <= 0 and j != sid)
    opp = no if cs == '+' else po
    if abs(ov) < 5: cat_b_7.append(sid)
    elif opp > 0: cat_c_7.append(sid)
    else: cat_d_7.append(sid)

print(f"  阈值 ≥7/8 (87.5%):")
print(f"    Cat B (|OR|<5): {len(cat_b_7)}")
print(f"    Cat C (opp>0): {len(cat_c_7)}")
print(f"    Cat D (其余): {len(cat_d_7)}")
print(f"    Total B+C+D: {len(cat_b_7)+len(cat_c_7)+len(cat_d_7)}")
print(f"    + Cat A: {len(flipped_idx)}")
print(f"    Grand Total: {len(cat_b_7)+len(cat_c_7)+len(cat_d_7)+len(flipped_idx)}")

# 7/8 但不在 8/8 中的 (灰色地带)
only7 = set(cat_b_7+cat_c_7+cat_d_7) - set(cat_b_8+cat_c_8+cat_d_8)
print(f"\n  仅 7/8 (不在 8/8 中): {len(only7)} 个样本")

# ═══════════════════════════════════════════════════════════════
sec("LAYER 11: Solvent Conflicts 溶剂冲突分析")
# ═══════════════════════════════════════════════════════════════
# 复现 gen_html_v4.py 的 solvent conflict 逻辑
sol_conflicts = []
sol_conflict_indices = set()  # 所有涉及溶剂冲突的样本 idx
for smi, group in smi_groups.items():
    if len(group) < 2: continue
    sol_or = defaultdict(list)
    for idx, ov, sol in group:
        sol_or[sol].append((idx, ov))
    if len(sol_or) < 2: continue
    all_signs = set()
    for recs in sol_or.values():
        for _, v in recs:
            all_signs.add(1 if v > 0 else 0)
    if len(all_signs) > 1:
        indices_in_conflict = [idx for idx, _, _ in group]
        sol_conflict_indices.update(indices_in_conflict)
        sol_conflicts.append({
            'smi': smi, 'n_solvents': len(sol_or),
            'n_records': len(group), 'indices': indices_in_conflict
        })

sol_conflicts.sort(key=lambda x: -x['n_records'])
print(f"  溶剂冲突分子数: {len(sol_conflicts)}")
print(f"  涉及的样本总数: {len(sol_conflict_indices)}")

# 溶剂冲突样本在 V22 kept 中的数量
sol_in_v22 = sol_conflict_indices & set(np.where(keep_v22)[0].tolist())
sol_not_in_v22 = sol_conflict_indices - sol_in_v22
print(f"  在 V22 kept 中: {len(sol_in_v22)}")
print(f"  已被移除 (Plan A/B/D): {len(sol_not_in_v22)}")

# ═══════════════════════════════════════════════════════════════
sec("LAYER 12: 交叉分析 — Solvent vs QC Categories")
# ═══════════════════════════════════════════════════════════════
# 8/8 QC 样本集合
qc_8_set = set(cat_b_8 + cat_c_8 + cat_d_8)
qc_a_set = set(flipped_idx.tolist())
qc_all_8 = qc_8_set | qc_a_set

# 7/8 QC 样本集合
qc_7_set = set(cat_b_7 + cat_c_7 + cat_d_7)
qc_all_7 = qc_7_set | qc_a_set

print(f"  --- 8/8 阈值 ---")
print(f"  QC 总样本 (A+B+C+D): {len(qc_all_8)}")
print(f"  Solvent conflict 样本 (V22 kept): {len(sol_in_v22)}")
overlap_8 = qc_all_8 & sol_in_v22
print(f"  交集 (既是QC又是溶剂冲突): {len(overlap_8)}")
sol_only_8 = sol_in_v22 - qc_all_8
print(f"  仅溶剂冲突 (不在QC中): {len(sol_only_8)}")
qc_only_8 = qc_all_8 - sol_in_v22
print(f"  仅QC (不在溶剂冲突中): {len(qc_only_8)}")

# 交集样本的 QC 类别分布
if overlap_8:
    print(f"\n  交集样本的 QC 类别:")
    for idx in sorted(overlap_8):
        cat = 'A' if idx in qc_a_set else ('B' if idx in set(cat_b_8) else ('C' if idx in set(cat_c_8) else 'D'))
        pr = pdf[idx]
        print(f"    idx={idx}: Cat {cat}, OR={pr['OR']}, solvent={pr.get('solvent','')}")

print(f"\n  --- 7/8 阈值 ---")
overlap_7 = qc_all_7 & sol_in_v22
sol_only_7 = sol_in_v22 - qc_all_7
print(f"  QC 总样本: {len(qc_all_7)}")
print(f"  交集: {len(overlap_7)}")
print(f"  仅溶剂冲突: {len(sol_only_7)}")

# ═══════════════════════════════════════════════════════════════
sec("LAYER 13: 溶剂冲突样本的 V23 模型 disagree 分布")
# ═══════════════════════════════════════════════════════════════
# 溶剂冲突样本中，V23 模型有多少个反对？
sol_dis_dist = Counter()
for idx in sol_in_v22:
    nd = v23_disagree.get(idx, -1)
    if nd >= 0:
        sol_dis_dist[nd] += 1

print(f"  溶剂冲突样本 (V22 kept) 的 V23 disagree 分布:")
for nd in range(n_v23_models + 1):
    cnt = sol_dis_dist.get(nd, 0)
    if cnt > 0:
        marker = ""
        if nd == 8: marker = " ← 在当前QC中"
        elif nd == 7: marker = " ← 7/8阈值可捕获"
        elif nd >= 5: marker = " ← 多数模型反对"
        print(f"    {nd}/{n_v23_models}: {cnt} 个样本{marker}")

# ═══════════════════════════════════════════════════════════════
sec("LAYER 14: V23 Ensemble 错误分析")
# ═══════════════════════════════════════════════════════════════
v23_err_set = set()
for sid, r in v23_pred.items():
    yt = int(r['y_true']); ep = float(r['ensemble_prob'])
    yp = 1 if ep > 0.5 else 0
    if yp != yt: v23_err_set.add(sid)

print(f"  V23 ensemble 错误: {len(v23_err_set)} / {len(v23_pred)} = {len(v23_err_set)/len(v23_pred)*100:.2f}%")
print(f"  V23 ensemble 正确: {len(v23_pred) - len(v23_err_set)}")

# 错误与 QC 的交集
err_qc_8 = v23_err_set & qc_all_8
err_qc_7 = v23_err_set & qc_all_7
err_sol = v23_err_set & sol_in_v22
print(f"\n  错误 ∩ QC(8/8): {len(err_qc_8)}")
print(f"  错误 ∩ QC(7/8): {len(err_qc_7)}")
print(f"  错误 ∩ 溶剂冲突: {len(err_sol)}")
print(f"  错误 ∩ (QC(8/8) ∪ 溶剂冲突): {len(v23_err_set & (qc_all_8 | sol_in_v22))}")
print(f"  错误 ∩ (QC(7/8) ∪ 溶剂冲突): {len(v23_err_set & (qc_all_7 | sol_in_v22))}")

# 纯错误 (不在任何 QC 或溶剂冲突中)
pure_err_8 = v23_err_set - qc_all_8 - sol_in_v22
pure_err_7 = v23_err_set - qc_all_7 - sol_in_v22
print(f"\n  纯错误 (不在QC(8/8)也不在溶剂冲突): {len(pure_err_8)}")
print(f"  纯错误 (不在QC(7/8)也不在溶剂冲突): {len(pure_err_7)}")

# ═══════════════════════════════════════════════════════════════
sec("LAYER 15: 综合对比表")
# ═══════════════════════════════════════════════════════════════
print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │                    V21 vs V23 QC 对比                       │
  ├──────────────┬──────────┬──────────┬──────────┬─────────────┤
  │   Category   │  V21 MGA │ V23 8/8  │ V23 ≥7/8 │    说明     │
  │              │ (33模型  │ (8模型   │ (8模型   │             │
  │              │  ≥90%)   │  100%)   │  87.5%)  │             │
  ├──────────────┼──────────┼──────────┼──────────┼─────────────┤
  │ Cat A (flip) │    {mga.get('cat_a_flip',0):>4}  │    {len(flipped_idx):>4}  │    {len(flipped_idx):>4}  │ 历史翻转    │
  │ Cat B (|OR|) │    {mga.get('cat_b_small_or',0):>4}  │    {len(cat_b_8):>4}  │    {len(cat_b_7):>4}  │ 小旋光值    │
  │ Cat C (part) │    {mga.get('cat_c_partial',0):>4}  │    {len(cat_c_8):>4}  │    {len(cat_c_7):>4}  │ 部分证据    │
  │ Cat D (susp) │    {mga.get('cat_d_suspect',0):>4}  │    {len(cat_d_8):>4}  │    {len(cat_d_7):>4}  │ 纯模型可疑  │
  ├──────────────┼──────────┼──────────┼──────────┼─────────────┤
  │ Total        │    {mga.get('total_candidates',0):>4}  │    {len(qc_all_8):>4}  │    {len(qc_all_7):>4}  │             │
  ├──────────────┼──────────┼──────────┼──────────┼─────────────┤
  │ Solvent冲突  │     --   │    {len(sol_conflicts):>4}  │    {len(sol_conflicts):>4}  │ {len(sol_conflict_indices)}个样本    │
  │ Sol∩QC       │     --   │    {len(overlap_8):>4}  │    {len(overlap_7):>4}  │ 交集        │
  │ Sol独有      │     --   │    {len(sol_only_8):>4}  │    {len(sol_only_7):>4}  │ 额外问题    │
  ├──────────────┼──────────┼──────────┼──────────┼─────────────┤
  │ V23 errors   │     --   │    {len(v23_err_set):>4}  │    {len(v23_err_set):>4}  │ ensemble    │
  │ Err∩QC       │     --   │    {len(err_qc_8):>4}  │    {len(err_qc_7):>4}  │             │
  │ Err∩Sol      │     --   │    {len(err_sol):>4}  │    {len(err_sol):>4}  │             │
  └──────────────┴──────────┴──────────┴──────────┴─────────────┘

  关键问题:
  1. V21 MGA header 声明 A={mga.get('cat_a_flip',0)} B={mga.get('cat_b_small_or',0)} C={mga.get('cat_c_partial',0)} D={mga.get('cat_d_suspect',0)} = {hdr_sum}
     但 total_candidates = {mga.get('total_candidates',0)} → {'一致' if hdr_sum == mga.get('total_candidates',0) else '不一致!'}
  2. V23 当前用 8/8 (100%) 阈值, 比 V21 的 90% 严格得多
  3. 溶剂冲突 {len(sol_conflicts)} 个分子中, {len(sol_only_8)} 个样本不在 8/8 QC 中
  4. V23 的 {len(v23_err_set)} 个错误中, {len(pure_err_8)} 个既不在 QC 也不在溶剂冲突中
""")

# ═══════════════════════════════════════════════════════════════
sec("LAYER 16: 建议 — 应该用什么数字?")
# ═══════════════════════════════════════════════════════════════
print(f"""
  方案A: 保持 8/8 (100%) — 最保守
    Cat A={len(flipped_idx)}, B={len(cat_b_8)}, C={len(cat_c_8)}, D={len(cat_d_8)}, Total={len(qc_all_8)}
    + Solvent独有={len(sol_only_8)}
    综合覆盖: {len(qc_all_8 | sol_in_v22)} 个样本

  方案B: 用 ≥7/8 (87.5%) — 接近V21的90%
    Cat A={len(flipped_idx)}, B={len(cat_b_7)}, C={len(cat_c_7)}, D={len(cat_d_7)}, Total={len(qc_all_7)}
    + Solvent独有={len(sol_only_7)}
    综合覆盖: {len(qc_all_7 | sol_in_v22)} 个样本

  方案C: 用 V21 MGA 原始数字 (33模型, ≥90%)
    Cat A={mga.get('cat_a_flip',0)}, B={mga.get('cat_b_small_or',0)}, C={mga.get('cat_c_partial',0)}, D={mga.get('cat_d_suspect',0)}, Total={mga.get('total_candidates',0)}
    注意: 这些数字基于 V21 的 33 模型, 不是 V23 的 8 模型
""")

print(f"\n{'='*70}")
print(f"  审计完成")
print(f"{'='*70}")