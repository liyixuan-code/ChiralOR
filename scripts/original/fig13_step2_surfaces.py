"""
SI Fig13 STEP 2 — compute fold-ensemble conditional response surfaces.
Reproduces Model B EXACTLY (same ART + StratifiedGroupKFold seed42 + group key
as _gate9_oof_shap.py). Trains 5 fold models + 1 final model on all 5204.

For each of the 6 frozen pairs (SI_Fig13_pair_selection.csv):
  grid = 1st-99th percentile of each continuous var (60x60)
  median-background: other 2281 features fixed at training median
  For every grid point:
     - 5 fold predict_proba(OR+)  -> mean / SD / min / max / per-fold
     - final Model B predict_proba -> consistency surface
  empirical-background (robustness, not main fig): for each grid point substitute
     the 2 vars into ALL 5204 real rows, keep other features real, average pred.
     (subsampled to 400 rows for tractability, seed42)
  support mask: 2D-bin density on real (x,y) observations -> SUPPORTED / LOW_SUPPORT

Exports per pair:
  SI_Fig13_pairXX_surface_values.npz   (grids, fold surfaces, mean/sd, final, empirical)
  SI_Fig13_pairXX_plot_data.csv        (long-form grid mean/sd/support)
  SI_Fig13_pairXX_support_mask.csv
  SI_Fig13_pairXX_fold_variability.csv
Records model SHA-256. simulated_data_used=False.
"""
import os, re, json, time, hashlib
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

t0 = time.time()
RNG = np.random.RandomState(42)
GRID_N = 60
EMP_SUB = 400            # rows for empirical-background averaging (seed42 subsample)

# ── real inputs (hard lock) ──────────────────────────────────────────────────
X      = np.load("X_C3_5204x2283.npy")
master = pd.read_csv("master_labels_v2.csv")
y      = master.OR_label.values.astype(int)
grp    = (master.canonical_smiles + "||" + master.solvent_group.astype(str)).values
feat   = pd.read_csv("feature_order_check.csv").feature_name.tolist()
pairs  = pd.read_csv("SI_Fig13_pair_selection.csv")

def sha(path,n=64): return hashlib.sha256(open(path,"rb").read()).hexdigest()[:n]
X_SHA = sha("X_C3_5204x2283.npy",16)

ART = dict(iterations=288, depth=8, learning_rate=0.1, l2_leaf_reg=1, random_seed=42,
    loss_function="Logloss", bootstrap_type="MVS", subsample=0.8, random_strength=1,
    border_count=254, nan_mode="Min", boost_from_average=False,
    leaf_estimation_method="Newton", leaf_estimation_iterations=10, min_data_in_leaf=1,
    grow_policy="SymmetricTree", feature_border_type="GreedyLogSum", rsm=1,
    auto_class_weights=None, verbose=0, thread_count=-1)

# ── train 5 fold models (exact Model B split) + final model ──────────────────
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
fold_models = []
fold_aucs = []
oof_pred = np.zeros(len(y))
for k,(tr,te) in enumerate(sgkf.split(X,y,groups=grp)):
    m = CatBoostClassifier(**ART); m.fit(X[tr], y[tr])
    fold_models.append(m)
    pr = m.predict_proba(X[te])[:,1]; oof_pred[te]=pr
    auc = roc_auc_score(y[te], pr); fold_aucs.append(auc)
    print(f"  fold{k}: n_tr={len(tr)} n_te={len(te)} AUC={auc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
oof_auc = roc_auc_score(y, oof_pred)
print(f"  OOF AUC={oof_auc:.4f} (expect ~0.928)")
assert abs(oof_auc-0.9278) < 0.01, f"OOF AUC {oof_auc} != frozen Model B 0.928 — refuse"

print("  training final Model B (all 5204)...", flush=True)
final_m = CatBoostClassifier(**ART); final_m.fit(X, y)

# model provenance SHA (save + hash each fold model)
os.makedirs("SI_Fig13_models", exist_ok=True)
model_sha = {}
for k,m in enumerate(fold_models):
    p=f"SI_Fig13_models/ModelB_fold{k}.cbm"; m.save_model(p); model_sha[f"fold{k}"]=sha(p,16)
final_m.save_model("SI_Fig13_models/ModelB_final.cbm"); model_sha["final"]=sha("SI_Fig13_models/ModelB_final.cbm",16)

bg = np.median(X, axis=0)            # median background (2283,)
emp_idx = RNG.choice(len(X), size=EMP_SUB, replace=False)   # empirical bg subsample

os.makedirs("SI_Fig13_STYLELOCK", exist_ok=True)

def clean(f): return re.sub(r"\W","", f.replace("geo:","geo_"))

variability_rows = []
for _,row in pairs.iterrows():
    pr_rank = int(row.pair_rank); fx, fy = row.feature_x, row.feature_y
    ci, cj = feat.index(fx), feat.index(fy)
    xv, yv = X[:,ci], X[:,cj]
    x1 = np.linspace(np.percentile(xv,1), np.percentile(xv,99), GRID_N)
    x2 = np.linspace(np.percentile(yv,1), np.percentile(yv,99), GRID_N)
    XX, YY = np.meshgrid(x1, x2)                 # (GRID_N,GRID_N)
    flat = XX.size

    # median-background grid design matrix
    grid = np.tile(bg, (flat,1)); grid[:,ci]=XX.ravel(); grid[:,cj]=YY.ravel()

    # 5 fold surfaces + final
    fold_surf = np.zeros((5, GRID_N, GRID_N))
    for k,m in enumerate(fold_models):
        fold_surf[k] = m.predict_proba(grid)[:,1].reshape(GRID_N,GRID_N)
    mean_surf = fold_surf.mean(0)
    sd_surf   = fold_surf.std(0)
    min_surf  = fold_surf.min(0)
    max_surf  = fold_surf.max(0)
    final_surf = final_m.predict_proba(grid)[:,1].reshape(GRID_N,GRID_N)

    # empirical-background averaged surface (robustness; subsample rows)
    Xsub = X[emp_idx].copy()
    emp_surf = np.zeros((GRID_N, GRID_N))
    for a in range(GRID_N):
        for b in range(GRID_N):
            tmp = Xsub.copy(); tmp[:,ci]=XX[a,b]; tmp[:,cj]=YY[a,b]
            emp_surf[a,b] = final_m.predict_proba(tmp)[:,1].mean()
    print(f"  pair{pr_rank} {fx} x {fy}: mean[{mean_surf.min():.2f},{mean_surf.max():.2f}] "
          f"SDmax={sd_surf.max():.3f}  ({time.time()-t0:.0f}s)", flush=True)

    # 2D support mask: bin real (x,y) into GRID_N-1 bins; cell SUPPORTED if the
    # local bin has >= MINCOUNT real observations (within 1-99 percentile box)
    MINCOUNT = 3
    xe = np.linspace(x1[0],x1[-1],GRID_N+1); ye = np.linspace(x2[0],x2[-1],GRID_N+1)
    Hc,_,_ = np.histogram2d(xv, yv, bins=[xe,ye])   # (GRID_N,GRID_N) counts, x=rows
    Hc = Hc.T                                        # -> (row=y,col=x) to match meshgrid
    supported = Hc >= MINCOUNT

    # save NPZ
    np.savez_compressed(f"SI_Fig13_STYLELOCK/SI_Fig13_pair{pr_rank:02d}_surface_values.npz",
        x1=x1, x2=x2, XX=XX, YY=YY,
        fold_surfaces=fold_surf, mean_surface=mean_surf, sd_surface=sd_surf,
        min_surface=min_surf, max_surface=max_surf, final_surface=final_surf,
        empirical_surface=emp_surf, support_counts=Hc, supported=supported,
        feature_x=fx, feature_y=fy)

    # long-form plot_data csv
    rows=[]
    for a in range(GRID_N):
        for b in range(GRID_N):
            rows.append(dict(ix=b, iy=a, x=XX[a,b], y=YY[a,b],
                mean_prob=mean_surf[a,b], sd_prob=sd_surf[a,b],
                min_prob=min_surf[a,b], max_prob=max_surf[a,b],
                final_prob=final_surf[a,b], empirical_prob=emp_surf[a,b],
                support_count=int(Hc[a,b]),
                support_flag="SUPPORTED" if supported[a,b] else "LOW_SUPPORT"))
    pd.DataFrame(rows).to_csv(f"SI_Fig13_STYLELOCK/SI_Fig13_pair{pr_rank:02d}_plot_data.csv", index=False)

    pd.DataFrame({"ix":np.tile(np.arange(GRID_N),GRID_N),
                  "iy":np.repeat(np.arange(GRID_N),GRID_N),
                  "support_count":Hc.ravel().astype(int),
                  "supported":supported.ravel()}).to_csv(
        f"SI_Fig13_STYLELOCK/SI_Fig13_pair{pr_rank:02d}_support_mask.csv", index=False)

    # fold variability summary
    variability_rows.append(dict(pair_rank=pr_rank, feature_x=fx, feature_y=fy,
        mean_prob_min=float(mean_surf.min()), mean_prob_max=float(mean_surf.max()),
        sd_mean=float(sd_surf.mean()), sd_max=float(sd_surf.max()),
        pct_supported=float(supported.mean()*100)))
    pd.DataFrame([dict(fold=k, auc=fold_aucs[k],
        surf_min=float(fold_surf[k].min()), surf_max=float(fold_surf[k].max()),
        surf_mean=float(fold_surf[k].mean())) for k in range(5)]).to_csv(
        f"SI_Fig13_STYLELOCK/SI_Fig13_pair{pr_rank:02d}_fold_variability.csv", index=False)

pd.DataFrame(variability_rows).to_csv("SI_Fig13_STYLELOCK/SI_Fig13_variability_summary.csv", index=False)

manifest = dict(figure="SI_Fig13", surface_type="fold-ensemble bivariate conditional response surface",
    main_surface="mean of 5 StratifiedGroupKFold Model B fold models predict_proba(OR+)",
    consistency_surface="final Model B (all 5204) predict_proba(OR+)",
    background="median of 2281 other features (median-background)",
    grid="1st-99th percentile, 60x60", n_pairs=int(len(pairs)),
    oof_auc=float(oof_auc), fold_aucs=[float(a) for a in fold_aucs],
    ART=ART, X_sha256_16=X_SHA, model_sha256_16=model_sha,
    NOT_oof_surface="grid points are hypothetical, not held-out observations",
    simulated_data_used=False)
json.dump(manifest, open("SI_Fig13_STYLELOCK/SI_Fig13_surface_manifest.json","w"),
          indent=2, default=float)
print(f"\nSTEP2 done ({time.time()-t0:.0f}s). models sha={model_sha}")
