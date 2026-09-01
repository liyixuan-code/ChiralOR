"""
Fig4 v2 — full lineage-traced recompute. Produces data for TWO figure versions:
  Version A (primary submission): Panel B = single full-feature algorithms only.
  Version B (backup): Panel B additionally shows a VERIFIED 6-base GBDT SuperLearner.

Historical Full8 SuperLearner (traced to run_v27_superlearner.py:298 GBDT_CONFIGS
+ v31_stage_b2_full8.py) = 8 base learners:
  CatBoost(-reference), CatBoost_v2-selection, LightGBM, XGBoost, HGBoosting,
  ExtraTrees, TabM, TabICL   + EnantiomerConstrainedStacker meta.
TabM (custom PyTorch arch) and TabICL (library absent) CANNOT be faithfully
reconstructed under the corrected grouped-CV protocol -> the ensemble here is a
VERIFIED 6-base GBDT reconstruction, NOT the historical Full8.

Unified protocol (identical for every model):
  X      = X_C3_5204x2283.npy           (5204 x 2283, no NaN)
  labels = master_labels_v2.csv          (OR_label; OR+ = positive)
  groups = canonical_smiles||solvent_group
  folds  = StratifiedGroupKFold(5, shuffle=True, random_state=42)  [ONE assignment]
  positive class = OR+ ; n = 5204 ; OR- 3141 / OR+ 2063
No old 0.954/0.9547, no simulated data, no Train/Test, no old Full8 predictions.
"""
import os, time, json, hashlib, pickle
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef,
    balanced_accuracy_score, brier_score_loss, confusion_matrix)
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier

t0 = time.time()
BASE = os.path.dirname(os.path.abspath(__file__))
def P(f): return os.path.join(BASE, f)
def sha16(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]

X      = np.load(P("X_C3_5204x2283.npy"))
master = pd.read_csv(P("master_labels_v2.csv"))
y      = master.OR_label.values.astype(int)
sid    = master.sample_id.values
grp    = (master.canonical_smiles + "||" + master.solvent_group.astype(str)).values
X_SHA  = sha16(P("X_C3_5204x2283.npy"))
GEO_IDX, BASE_IDX, ALL_IDX = list(range(2276, 2283)), list(range(2276)), list(range(2283))
assert X.shape == (5204, 2283) and (int((y == 0).sum()), int((y == 1).sum())) == (3141, 2063)

# ── released-artifact CatBoost = CatBoost_v2-release = Model B = deployed ──
ART = dict(iterations=288, depth=8, learning_rate=0.1, l2_leaf_reg=1, random_seed=42,
    loss_function="Logloss", bootstrap_type="MVS", subsample=0.8, random_strength=1,
    border_count=254, nan_mode="Min", boost_from_average=False,
    leaf_estimation_method="Newton", leaf_estimation_iterations=10, min_data_in_leaf=1,
    grow_policy="SymmetricTree", feature_border_type="GreedyLogSum", rsm=1,
    auto_class_weights=None, verbose=0, thread_count=-1)
# ── historical Full8 GBDT configs (run_v27_superlearner.py:298) ───────────
HIST_CATBOOST = dict(iterations=1500, learning_rate=0.048, depth=8, l2_leaf_reg=3.0,
    random_seed=42, verbose=0, eval_metric="AUC", scale_pos_weight=2.0)
HIST_CATBOOST_V2SEL = dict(iterations=1200, learning_rate=0.03, depth=7, l2_leaf_reg=7.0,
    random_seed=100, verbose=0, eval_metric="AUC", scale_pos_weight=1.8)
CFG_XGB = dict(n_estimators=500, max_depth=7, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, random_state=42, eval_metric="logloss", verbosity=0,
    n_jobs=-1, tree_method="hist")
CFG_LGB = dict(n_estimators=500, learning_rate=0.05, max_depth=-1, subsample=0.8,
    colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=-1)
CFG_HGB = dict(max_iter=500, learning_rate=0.05, max_depth=7, random_state=42)
CFG_ET  = dict(n_estimators=500, max_depth=None, n_jobs=-1, random_state=42)

def mk(kind):
    if kind == "logreg":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=42))])
    if kind == "rf":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("clf", RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42))])
    if kind == "et":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("clf", ExtraTreesClassifier(**CFG_ET))])
    if kind == "hgb":  return HistGradientBoostingClassifier(**CFG_HGB)
    if kind == "xgb":  return xgb.XGBClassifier(**CFG_XGB)
    if kind == "lgb":  return lgb.LGBMClassifier(**CFG_LGB)
    if kind == "cat_release":  return CatBoostClassifier(**ART)
    if kind == "cat_hist":     return CatBoostClassifier(**HIST_CATBOOST)
    if kind == "cat_v2sel":    return CatBoostClassifier(**HIST_CATBOOST_V2SEL)
    raise ValueError(kind)

# ── ONE fold assignment reused everywhere ─────────────────────────────────
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
FOLDS = list(sgkf.split(X, y, groups=grp))
fold_of = np.full(len(y), -1, int)
for k, (_, te) in enumerate(FOLDS): fold_of[te] = k
assert (fold_of >= 0).all()

def oof(kind, cols):
    Xc = X[:, cols]; o = np.zeros(len(y)); fa = []
    for k, (tr, te) in enumerate(FOLDS):
        m = mk(kind); m.fit(Xc[tr], y[tr])
        p = m.predict_proba(Xc[te])[:, 1]; o[te] = p
        fa.append(roc_auc_score(y[te], p))
    return o, fa

# ── single full-feature algorithm candidates (Panel B, both versions) ─────
# Panel B uses the FULL 2283 features for every algorithm. CatBoost here = the
# released-artifact/deployed config (Model B). No feature-ablation model here.
print("== Panel B single full-feature models (2283) ==", flush=True)
runs = {}   # display_name -> dict(oof, faucs, cols, kind, nfeat)
def add(name, kind, cols):
    o, fa = oof(kind, cols)
    runs[name] = dict(oof=o, faucs=fa, cols=cols, kind=kind, nfeat=len(cols))
    print(f"  {name}: OOF AUC={roc_auc_score(y,o):.4f}  ({time.time()-t0:.0f}s)", flush=True)

add("Logistic Regression",       "logreg",      ALL_IDX)
add("Random Forest",             "rf",          ALL_IDX)
add("ExtraTrees",                "et",          ALL_IDX)
add("HistGradientBoosting",      "hgb",         ALL_IDX)
add("XGBoost",                   "xgb",         ALL_IDX)
add("LightGBM",                  "lgb",         ALL_IDX)
add("CatBoost_v2 (Model B)",     "cat_release", ALL_IDX)

mb_auc = roc_auc_score(y, runs["CatBoost_v2 (Model B)"]["oof"])
assert abs(mb_auc - 0.9278) < 0.01, mb_auc
print(f"  Model B check OK ({mb_auc:.4f} == frozen 0.9278)", flush=True)

# ── extra historical GBDT base learners needed ONLY for the 6-base stack ──
# CatBoost_v2-selection and CatBoost(-reference) are historical Full8 bases,
# distinct configs from the deployed Model B. Reused as SuperLearner bases.
print("== extra historical GBDT bases for 6-base SuperLearner ==", flush=True)
add("CatBoost-reference (hist)",     "cat_hist",   ALL_IDX)
add("CatBoost_v2-selection (hist)",  "cat_v2sel",  ALL_IDX)

# ── ablation arms (Panel C only) ──────────────────────────────────────────
print("== Panel C ablation arms ==", flush=True)
add("CatBoost_v2 baseline (2276, no signed-geom.)", "cat_release", BASE_IDX)
add("CatBoost_v2 signed-geometry only (7)",         "cat_release", GEO_IDX)

# ── VERIFIED 6-base GBDT SuperLearner (cross-fitted meta) ─────────────────
# Bases = the 6 reproducible historical GBDTs (NOT TabM/TabICL):
#   CatBoost(-reference), CatBoost_v2-selection, LightGBM, XGBoost, HGBoosting, ExtraTrees
SL6_BASES = ["CatBoost-reference (hist)", "CatBoost_v2-selection (hist)", "LightGBM",
             "XGBoost", "HistGradientBoosting", "ExtraTrees"]
Z6 = np.column_stack([runs[b]["oof"] for b in SL6_BASES])
sl6 = np.zeros(len(y)); sl6_fa = []; sl6_coef = []
for k, (_, te) in enumerate(FOLDS):
    trm = fold_of != k
    meta = LogisticRegression(max_iter=2000, C=1.0, random_state=42).fit(Z6[trm], y[trm])
    p = meta.predict_proba(Z6[fold_of == k])[:, 1]; sl6[fold_of == k] = p
    sl6_fa.append(roc_auc_score(y[fold_of == k], p)); sl6_coef.append(meta.coef_[0])
runs["SuperLearner (6-base GBDT, verified)"] = dict(
    oof=sl6, faucs=sl6_fa, cols=ALL_IDX, kind="stack6", nfeat="6 GBDT bases")
sl6_auc = roc_auc_score(y, sl6)
print(f"  SuperLearner(6-base) OOF AUC={sl6_auc:.4f} coefs={np.round(np.mean(sl6_coef,0),3)}", flush=True)
assert sl6_auc < 0.95, f"6-base stack {sl6_auc} too high — cross-fit check"

# cache all OOF vectors
pred_df = pd.DataFrame({"sample_id": sid, "fold": fold_of, "y_true": y})
for name, d in runs.items(): pred_df[name] = d["oof"]
pred_df.to_csv(P("Fig4_v2_oof_predictions_all_models.csv"), index=False)
pickle.dump({"runs": {k: (v["oof"], v["faucs"], v["nfeat"], v["kind"]) for k, v in runs.items()},
             "fold_of": fold_of, "sl6_coef": np.mean(sl6_coef, 0).tolist(),
             "X_SHA": X_SHA, "mb_auc": mb_auc, "sl6_auc": sl6_auc},
            open(P("_fig4v2_runs.pkl"), "wb"))
print(f"cached. total {time.time()-t0:.0f}s", flush=True)

