#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
SERVER_compute_oof_interaction_v2.py  (修正版 — 修复 OOF 泄漏)
================================================================================
v1 问题: 重新运行 StratifiedGroupKFold.split() 得到训练索引, 但不同 sklearn
版本 split 结果不一致 → 与 parquet 存储的 fold 错位 → 101/128 泄漏。

v2 修复: **不再重新 split**, 直接读取 oof_predictions_master_labels_v2.parquet
的 `fold` 列作为唯一 fold 真值。fold-k 模型训练集 = {fold != k}, 只解释 {fold == k}。
逻辑上保证泄漏 = 0, 与 sklearn 版本无关。

科学规范同 v1: master_labels_v2 / Model B artifact 参数 / OR+=1 / RawFormulaVal log-odds。

输入 (同目录):
  X_C3_5204x2283.npy
  master_labels_v2.csv
  feature_order_check.csv
  oof_shap_global_importance.csv
  interaction_subset_manifest_seed42.csv
  oof_predictions_master_labels_v2.parquet   <-- fold 真值来源(关键)

输出:
  oof_shap_interaction_modelB_subset128_v2.npy      # (128,2283,2283) f32
  oof_interaction_pair_importance_v2.csv
  oof_interaction_fold_stability_v2.csv
  SERVER_interaction_manifest_v2.json

依赖: numpy pandas scikit-learn catboost pyarrow
运行: python SERVER_compute_oof_interaction_v2.py
预计: 5 fold 训练(~40s) + 128 样本 interaction(单样本约44s, 共~95min)
================================================================================
"""
import numpy as np, pandas as pd, hashlib, json, os, sys, time
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "X":      os.path.join(HERE, "X_C3_5204x2283.npy"),
    "labels": os.path.join(HERE, "master_labels_v2.csv"),
    "feat":   os.path.join(HERE, "feature_order_check.csv"),
    "global": os.path.join(HERE, "oof_shap_global_importance.csv"),
    "subset": os.path.join(HERE, "interaction_subset_manifest_seed42.csv"),
    "oof":    os.path.join(HERE, "oof_predictions_master_labels_v2.parquet"),
}
for k, p in FILES.items():
    if not os.path.exists(p):
        sys.exit(f"[BLOCKED] missing {k}: {p}")

def sha256(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()

X       = np.load(FILES["X"])
master  = pd.read_csv(FILES["labels"]).sort_values("sample_id").reset_index(drop=True)
feat    = pd.read_csv(FILES["feat"]).feature_name.tolist()
sub     = pd.read_csv(FILES["subset"])
oof     = pd.read_parquet(FILES["oof"]).sort_values("sample_id").reset_index(drop=True)

y = master.OR_label.values.astype(int)
# ---- 关键: fold 直接来自 parquet, 按 sample_id 对齐 ----
assert (oof.sample_id.values == np.arange(5204)).all(), "oof sample_id 必须是 0..5203"
FOLD = oof.fold.values.astype(int)                    # 每个 sample_id 的验证折 (唯一真值)
sub_ids = sub.sample_id.values.astype(int)

assert X.shape == (5204, 2283)
assert len(feat) == 2283
assert len(sub_ids) == 128
print(f"[OK] fold来源=parquet stored; fold分布={pd.Series(FOLD).value_counts().sort_index().to_dict()}")

ART = dict(
    iterations=288, depth=8, learning_rate=0.1, l2_leaf_reg=1, random_seed=42,
    loss_function="Logloss", bootstrap_type="MVS", subsample=0.8, random_strength=1,
    border_count=254, nan_mode="Min", boost_from_average=False,
    leaf_estimation_method="Newton", leaf_estimation_iterations=10, min_data_in_leaf=1,
    grow_policy="SymmetricTree", feature_border_type="GreedyLogSum", rsm=1,
    auto_class_weights=None, verbose=0, thread_count=-1,
)

# ---- 用 stored fold 训练 5 个模型: fold-k 训练集 = {FOLD != k} ----
fold_models = {}
for k in range(5):
    tr = np.where(FOLD != k)[0]
    te = np.where(FOLD == k)[0]
    m = CatBoostClassifier(**ART)
    m.fit(X[tr], y[tr])
    fold_models[k] = (m, set(tr.tolist()))
    auc = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
    print(f"[fold{k}] trained on {len(tr)} (val {len(te)}), val_AUC={auc:.4f} ({time.time()-t0:.0f}s)", flush=True)

# ---- 逐样本 OOF interaction: 用其 stored fold 模型 ----
inter = np.zeros((len(sub_ids), 2283, 2283), dtype=np.float32)
leak = 0
for j, sid in enumerate(sub_ids):
    k = int(FOLD[sid])                     # 该样本的验证折
    m, tr_set = fold_models[k]
    if sid in tr_set:                      # 逻辑上不该发生
        leak += 1
    iv = m.get_feature_importance(Pool(X[sid:sid+1]), type="ShapInteractionValues")
    inter[j] = iv[0, :-1, :-1].astype(np.float32)
    if (j + 1) % 16 == 0 or j == 0:
        print(f"  interaction {j+1}/128 (fold{k}) ({time.time()-t0:.0f}s)", flush=True)

print(f"\n[OOF校验] leak = {leak}/128 (应=0)")
if leak != 0:
    print("[WARN] 仍有泄漏 — 请检查 parquet fold 与训练是否一致")

np.save(os.path.join(HERE, "oof_shap_interaction_modelB_subset128_v2.npy"), inter)

# pair importance
mabs = np.abs(inter).mean(axis=0)
rows = []
for i in range(2283):
    for jj in range(i + 1, 2283):
        v = mabs[i, jj]
        if v > 1e-5:
            rows.append((feat[i], feat[jj], float(v)))
pair = pd.DataFrame(rows, columns=["feat_i", "feat_j", "mean_abs_interaction"]).sort_values(
    "mean_abs_interaction", ascending=False).reset_index(drop=True)
pair.head(300).to_csv(os.path.join(HERE, "oof_interaction_pair_importance_v2.csv"), index=False)

# fold stability
sub_fold = FOLD[sub_ids]
srows = []
for k in range(5):
    mask = sub_fold == k
    if mask.sum() >= 2:
        mp = np.abs(inter[mask]).mean(axis=0)
        tri = np.triu_indices(2283, k=1)
        vals = mp[tri]; top = np.argsort(vals)[::-1][:5]
        srows.append({"fold": k, "n": int(mask.sum()),
                      "top5": ";".join(f"{feat[tri[0][t]]}|{feat[tri[1][t]]}" for t in top)})
    else:
        srows.append({"fold": k, "n": int(mask.sum()), "top5": "too_few"})
pd.DataFrame(srows).to_csv(os.path.join(HERE, "oof_interaction_fold_stability_v2.csv"), index=False)

man = {
    "inputs_sha256": {k: sha256(p) for k, p in FILES.items()},
    "output_sha256": sha256(os.path.join(HERE, "oof_shap_interaction_modelB_subset128_v2.npy")),
    "shape": list(inter.shape), "dtype": "float32", "scale": "RawFormulaVal log-odds",
    "fold_source": "parquet stored fold (NOT re-split)",
    "oof_leak": int(leak), "n_subset": int(len(sub_ids)),
    "top10_pairs": pair.head(10).to_dict("records"),
    "elapsed_s": round(time.time() - t0, 1),
}
json.dump(man, open(os.path.join(HERE, "SERVER_interaction_manifest_v2.json"), "w"),
          indent=2, ensure_ascii=False)

print(f"\n=== v2 完成 ({time.time()-t0:.0f}s) leak={leak}/128 ===")
print("top10 pair:"); print(pair.head(10).to_string(index=False))
print("\n回传: oof_shap_interaction_modelB_subset128_v2.npy, oof_interaction_pair_importance_v2.csv,")
print("      oof_interaction_fold_stability_v2.csv, SERVER_interaction_manifest_v2.json")
