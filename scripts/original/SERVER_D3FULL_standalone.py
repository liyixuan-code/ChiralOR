# -*- coding: utf-8 -*-
"""
D3-FULL — SHAP-interaction sensitivity analysis (SERVER standalone, self-contained).

这是一个 newly prespecified sensitivity analysis(不是原 seed42 subset 的 replication;
原 subset-selection 脚本不可恢复)。

HARD RULES(已内建):
 1. 不重新训练/不 reseed 正式模型 —— 直接 load 冻结的 5 个 fold 模型。
 2. 用现有 frozen 5-fold CatBoost OOF 模型(ModelB_fold{0..4}.cbm)。
 3. 每个被解释样本只用它对应 OOF fold 的模型(fold 来自 parquet 存储列)。
 4. 不修改 seed42 官方 interaction tensor/result(只读作参考)。
 5. 用真实 SHAP interaction(ShapInteractionValues),不用 CatBoost native Interaction。
 6. 不跑 D4 LOWESS。
 7. Fig10 在本分析完成前继续 deferred。
 流式累加:逐样本算 interaction 矩阵 → 累计 mean|Phi_ij|,绝不落盘 2.6GB 张量。

============================================================================
使用方法(服务器):
  1) 建一个目录,把下面 10 个输入文件放进同一目录(扁平,无需子目录):
       X_C3_5204x2283.npy
       feature_order_check.csv
       oof_predictions_master_labels_v2.parquet
       ModelB_fold0.cbm  ModelB_fold1.cbm  ModelB_fold2.cbm  ModelB_fold3.cbm  ModelB_fold4.cbm
       oof_interaction_pair_importance_v2.csv         (seed42 官方参考;缺失也能跑,聚合里 seed42 列标 NA)
       interaction_sensitivity_sampling_manifest.csv  (已冻结的抽样清单;缺失则脚本按同一规则重建)
  2) 依赖: pip install numpy pandas scipy catboost pyarrow
  3) 运行: python SERVER_D3FULL_standalone.py
     (可选)只跑部分 seed: python SERVER_D3FULL_standalone.py --seeds 101,202
  4) 预计: 每 seed ~60-70 分钟(单进程),5 seed 顺序约 5-6 小时。
     不要 5 个 seed 并行 —— 每进程要分配 2283x2283 float 矩阵,并行会 OOM。
  5) 产出(同目录 ./D3_FULL_OUTPUT/):
       seed_<seed>/top50_pairs_seed<seed>.csv
       seed_<seed>/tracked_pairs_seed<seed>.csv
       seed_<seed>/meta_seed<seed>.json
       interaction_sensitivity_sampling_manifest.csv   (若脚本重建)
       interaction_stability_FULL_aggregate.csv
       interaction_rank_correlations_FULL.csv
       interaction_tracked_pairs_FULL.csv
       _d3full_summary.json
  把 D3_FULL_OUTPUT/ 整个传回来即可。
============================================================================
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, Pool

HERE = os.path.dirname(os.path.abspath(__file__))
def P(fn): return os.path.join(HERE, fn)
OUT = os.path.join(HERE, "D3_FULL_OUTPUT"); os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="101,202,303,404,505")
args = ap.parse_args()
SEEDS = [int(s) for s in args.seeds.split(",")]
T0 = time.time()

# ---- inputs ----------------------------------------------------------------
X    = np.load(P("X_C3_5204x2283.npy"))
feat = pd.read_csv(P("feature_order_check.csv")).feature_name.tolist()
NF   = len(feat)                       # 2283
oof  = pd.read_parquet(P("oof_predictions_master_labels_v2.parquet")).sort_values("sample_id").reset_index(drop=True)
FOLD = oof.fold.values.astype(int); y = oof.y_true.values.astype(int); SID = oof.sample_id.values
assert X.shape == (5204, NF), X.shape
def fname(i): return feat[i] if i < NF else "BIAS"
def idx(n):   return feat.index(n)

# ---- frozen fold models (verify they reproduce stored OOF => leak-free) -----
fold_models = {}
for k in range(5):
    m = CatBoostClassifier(); m.load_model(P(f"ModelB_fold{k}.cbm")); fold_models[k] = m
maxerr = 0.0
for k in range(5):
    te = np.where(FOLD == k)[0]
    pk = fold_models[k].predict_proba(X[te])[:, 1]
    maxerr = max(maxerr, float(np.abs(pk - oof.oof_prob.values[te]).max()))
print(f"[check] frozen fold models vs stored OOF max err = {maxerr:.2e} "
      f"({'OK leak-free' if maxerr < 1e-6 else 'WARNING mismatch'})", flush=True)

# ---- frozen sampling manifest (load if present else rebuild same rule) ------
man_path = P("interaction_sensitivity_sampling_manifest.csv")
if os.path.exists(man_path):
    man = pd.read_csv(man_path)
    print("[manifest] loaded frozen manifest", flush=True)
else:
    RULE = ("fold-stratified(target[26,26,26,25,25] over folds0-4)+class-stratified"
            "(within-fold OR-/OR+ matched to fold true ratio, remainder->majority); "
            "sampling WITHOUT replacement; NEW prespecified sensitivity (NOT seed42 replication)")
    fold_target = {0: 26, 1: 26, 2: 26, 3: 25, 4: 25}
    rows = []
    for seed in [101, 202, 303, 404, 505]:
        rng = np.random.RandomState(seed)
        for k in range(5):
            idx_f = np.where(FOLD == k)[0]; yk = y[idx_f]; nf = fold_target[k]
            frac_pos = (yk == 1).mean()
            n_pos = int(round(nf * frac_pos)); n_neg = nf - n_pos
            pos_ids = idx_f[yk == 1]; neg_ids = idx_f[yk == 0]
            n_pos = min(n_pos, len(pos_ids)); n_neg = min(n_neg, len(neg_ids))
            while n_pos + n_neg < nf:
                if len(neg_ids) - n_neg >= len(pos_ids) - n_pos: n_neg += 1
                else: n_pos += 1
            sel = np.concatenate([rng.choice(pos_ids, n_pos, replace=False),
                                  rng.choice(neg_ids, n_neg, replace=False)])
            for i in sel:
                rows.append(dict(seed=seed, sample_id=int(SID[i]), fold=int(k),
                                 true_label=int(y[i]), selection_rule=RULE))
    man = pd.DataFrame(rows)
    man.to_csv(os.path.join(OUT, "interaction_sensitivity_sampling_manifest.csv"), index=False, encoding="utf-8-sig")
    print("[manifest] rebuilt frozen manifest (same rule)", flush=True)

# ---- tracked pairs ---------------------------------------------------------
TRACK = {
 "MACCS_129 x geo:signed_dihedral_subst_c_rn1_rn2": (idx("MACCS_129"), idx("geo:signed_dihedral_subst_c_rn1_rn2")),
 "geo:signed_tetra_volume x geo:subst_to_ringplane_signed_dist": (idx("geo:signed_tetra_volume"), idx("geo:subst_to_ringplane_signed_dist")),
 "ECFP_1412 x PAS_5": (idx("ECFP_1412"), idx("PAS_5")),
}
iu = np.triu_indices(NF, k=1)
print(f"[setup] loaded X+models {time.time()-T0:.0f}s; seeds={SEEDS}", flush=True)

# ============================ per-seed streaming ============================
for seed in SEEDS:
    odir = os.path.join(OUT, f"seed_{seed}"); os.makedirs(odir, exist_ok=True)
    if os.path.exists(os.path.join(odir, f"meta_seed{seed}.json")):
        print(f"[seed{seed}] already done, skip", flush=True); continue
    sub = man[man.seed == seed].reset_index(drop=True); assert len(sub) == 128, len(sub)
    ts = time.time(); acc = np.zeros((NF, NF), dtype=np.float64)
    track_vals = {k: [] for k in TRACK}
    for j, row in sub.iterrows():
        sid = int(row.sample_id); k = int(row.fold)
        iv = fold_models[k].get_feature_importance(Pool(X[sid:sid+1]), type="ShapInteractionValues")
        acc += np.abs(iv[0, :NF, :NF])
        for name, (a, b) in TRACK.items(): track_vals[name].append(float(iv[0, a, b]))
        del iv
        if (j+1) % 32 == 0: print(f"  seed{seed} {j+1}/128 ({time.time()-ts:.0f}s)", flush=True)
    acc /= len(sub)
    vals = acc[iu]; order = np.argsort(vals)[::-1][:50]
    rows = [dict(rank=rk, feat_i=fname(iu[0][o]), feat_j=fname(iu[1][o]),
                 pair=f"{fname(iu[0][o])} x {fname(iu[1][o])}",
                 mean_abs_interaction=round(float(vals[o]), 6)) for rk, o in enumerate(order, 1)]
    pd.DataFrame(rows).to_csv(os.path.join(odir, f"top50_pairs_seed{seed}.csv"), index=False, encoding="utf-8-sig")
    allsorted = np.sort(vals)[::-1]; tvr = []
    for name, (a, b) in TRACK.items():
        v = np.array(track_vals[name]); pm = float(acc[a, b]); rank = int((allsorted > pm).sum() + 1)
        tvr.append(dict(pair=name, mean_abs=round(float(np.abs(v).mean()), 6),
                        mean_abs_from_matrix=round(pm, 6), mean_signed=round(float(v.mean()), 6),
                        n=len(v), rank_among_all_pairs=rank, in_top50=bool(rank <= 50)))
    pd.DataFrame(tvr).to_csv(os.path.join(odir, f"tracked_pairs_seed{seed}.csv"), index=False, encoding="utf-8-sig")
    meta = dict(seed=seed, n=128, runtime_s=round(time.time()-ts, 1),
                fold_dist=sub.fold.value_counts().sort_index().to_dict(),
                OR_minus=int((sub.true_label == 0).sum()), OR_plus=int((sub.true_label == 1).sum()),
                top_pair=rows[0]["pair"], top_pair_val=rows[0]["mean_abs_interaction"])
    json.dump(meta, open(os.path.join(odir, f"meta_seed{seed}.json"), "w"), indent=2, default=str)
    print(f"[seed{seed}] DONE {meta['runtime_s']}s top={meta['top_pair']} ({meta['top_pair_val']})  "
          f"[total {time.time()-T0:.0f}s]", flush=True)

# ============================ cross-seed aggregate ==========================
done = [s for s in SEEDS if os.path.exists(os.path.join(OUT, f"seed_{s}", f"meta_seed{s}.json"))]
if len(done) >= 2:
    from scipy.stats import spearmanr
    def norm(pk): return " x ".join(sorted(pk.split(" x ")))
    seed_pairs = {}
    for s in done:
        df = pd.read_csv(os.path.join(OUT, f"seed_{s}", f"top50_pairs_seed{s}.csv"))
        seed_pairs[s] = {norm(f"{r.feat_i} x {r.feat_j}"): (int(r["rank"]), float(r.mean_abs_interaction)) for _, r in df.iterrows()}
    # seed42 official reference (kept SEPARATE, not merged)
    seed42 = {}
    if os.path.exists(P("oof_interaction_pair_importance_v2.csv")):
        off = pd.read_csv(P("oof_interaction_pair_importance_v2.csv")).sort_values("mean_abs_interaction", ascending=False).reset_index(drop=True)
        seed42 = {norm(f"{a} x {b}"): (rk+1, float(v)) for rk, (a, b, v) in enumerate(zip(off.feat_i, off.feat_j, off.mean_abs_interaction))}
    allpk = sorted(set().union(*[set(d) for d in seed_pairs.values()]))
    agg = []
    for pk in allpk:
        ranks = [seed_pairs[s][pk][0] for s in done if pk in seed_pairs[s]]
        mags  = [seed_pairs[s][pk][1] for s in done if pk in seed_pairs[s]]
        t10 = sum(1 for s in done if pk in seed_pairs[s] and seed_pairs[s][pk][0] <= 10)
        t20 = sum(1 for s in done if pk in seed_pairs[s] and seed_pairs[s][pk][0] <= 20)
        agg.append(dict(pair=pk, top10_freq=f"{t10}/{len(done)}", top20_freq=f"{t20}/{len(done)}",
            appear_in_n_new_seeds=len(ranks),
            mean_rank=round(float(np.mean(ranks)), 2) if ranks else None,
            median_rank=int(np.median(ranks)) if ranks else None,
            rank_sd=round(float(np.std(ranks)), 2) if ranks else None,
            rank_iqr=round(float(np.percentile(ranks, 75)-np.percentile(ranks, 25)), 2) if ranks else None,
            mean_magnitude=round(float(np.mean(mags)), 6) if mags else None,
            magnitude_sd=round(float(np.std(mags)), 6) if mags else None,
            seed42_rank=seed42.get(pk, ("NA",))[0], seed42_mag=round(seed42[pk][1], 6) if pk in seed42 else "NA",
            _t10=t10, _t20=t20))
    aggdf = pd.DataFrame(agg).sort_values(["_t10", "_t20", "mean_rank"], ascending=[False, False, True]).drop(columns=["_t10", "_t20"])
    aggdf.to_csv(os.path.join(OUT, "interaction_stability_FULL_aggregate.csv"), index=False, encoding="utf-8-sig")
    # spearman between new seeds + vs seed42
    def rv(s): return [seed_pairs[s][pk][0] if pk in seed_pairs[s] else 60 for pk in allpk]
    cors = []
    for i in range(len(done)):
        for j in range(i+1, len(done)):
            rho, _ = spearmanr(rv(done[i]), rv(done[j])); cors.append(dict(seed_i=done[i], seed_j=done[j], spearman_rho=round(float(rho), 3)))
    if seed42:
        for s in done:
            v2 = [seed42[pk][0] if pk in seed42 else 60 for pk in allpk]
            rho, _ = spearmanr(rv(s), v2); cors.append(dict(seed_i=s, seed_j="seed42_ref", spearman_rho=round(float(rho), 3)))
    pd.DataFrame(cors).to_csv(os.path.join(OUT, "interaction_rank_correlations_FULL.csv"), index=False, encoding="utf-8-sig")
    # tracked pairs summary
    TR = ["MACCS_129 x geo:signed_dihedral_subst_c_rn1_rn2",
          "geo:signed_tetra_volume x geo:subst_to_ringplane_signed_dist", "ECFP_1412 x PAS_5"]
    trk = [aggdf[aggdf.pair.apply(lambda p: norm(p) == norm(tp))].iloc[0].to_dict()
           for tp in TR if len(aggdf[aggdf.pair.apply(lambda p: norm(p) == norm(tp))])]
    pd.DataFrame(trk).to_csv(os.path.join(OUT, "interaction_tracked_pairs_FULL.csv"), index=False, encoding="utf-8-sig")
    mean_spear = float(np.mean([c["spearman_rho"] for c in cors if c["seed_j"] != "seed42_ref"])) if len(done) >= 2 else None
    # wording-tier verdict for tracked pairs
    def tier(pk):
        r = aggdf[aggdf.pair.apply(lambda p: norm(p) == norm(pk))]
        if not len(r): return "C_prominent_in_seed42_only_exploratory"
        t10 = int(r.iloc[0].top10_freq.split("/")[0]); t20 = int(r.iloc[0].top20_freq.split("/")[0])
        if t10 >= 4: return "A_consistently_prominent"
        if t20 >= 3: return "B_recurrent"
        return "C_prominent_in_seed42_only_exploratory"
    summ = dict(new_seeds=done, mean_spearman_new_seeds=round(mean_spear, 3) if mean_spear is not None else None,
                n_pairs_top10_in_ge4_new_seeds=int(aggdf.top10_freq.isin([f"4/{len(done)}", f"5/{len(done)}"]).sum()),
                tracked_pair_tiers={tp: tier(tp) for tp in TR},
                top_stable_pairs=aggdf.head(10)[["pair", "top10_freq", "top20_freq", "mean_rank", "seed42_rank"]].to_dict("records"),
                note="seed42 official result kept SEPARATE as historical reference; NOT merged into a 6-seed average.")
    json.dump(summ, open(os.path.join(OUT, "_d3full_summary.json"), "w"), indent=2, default=str)
    print("[aggregate] done. mean Spearman(new seeds) =", summ["mean_spearman_new_seeds"], flush=True)
    print("[aggregate] tracked-pair tiers:", summ["tracked_pair_tiers"], flush=True)
print(f"ALL DONE total {time.time()-T0:.0f}s", flush=True)
