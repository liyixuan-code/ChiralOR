import numpy as np, pandas as pd, os, time
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool
t0=time.time()

X = np.load("X_C3_5204x2283.npy")
master = pd.read_csv("master_labels_v2.csv")
y = master.OR_label.values.astype(int)
grp = (master.canonical_smiles + "||" + master.solvent_group.astype(str)).values
feat_names = pd.read_csv("feature_order_check.csv").feature_name.tolist()

ART=dict(iterations=288,depth=8,learning_rate=0.1,l2_leaf_reg=1,random_seed=42,
    loss_function="Logloss",bootstrap_type="MVS",subsample=0.8,random_strength=1,
    border_count=254,nan_mode="Min",boost_from_average=False,leaf_estimation_method="Newton",
    leaf_estimation_iterations=10,min_data_in_leaf=1,grow_policy="SymmetricTree",
    feature_border_type="GreedyLogSum",rsm=1,auto_class_weights=None,verbose=0,thread_count=-1)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_shap = np.zeros((len(y),2283))
oof_base = np.zeros(len(y))
oof_fold = np.zeros(len(y),dtype=int)
oof_pred = np.zeros(len(y))
oof_rawmargin_check = []

for k,(tr,te) in enumerate(sgkf.split(X,y,groups=grp)):
    m=CatBoostClassifier(**ART); m.fit(X[tr],y[tr])
    oof_pred[te]=m.predict_proba(X[te])[:,1]; oof_fold[te]=k
    # TreeSHAP on validation fold
    pool_te = Pool(X[te])
    sv = m.get_feature_importance(pool_te, type="ShapValues")  # (n_te, 2283+1)
    oof_shap[te] = sv[:,:-1]
    oof_base[te] = sv[:,-1]
    # additivity check: base+sum(shap) == RawFormulaVal
    raw = m.predict(X[te], prediction_type="RawFormulaVal")
    recon = sv[:,-1] + sv[:,:-1].sum(axis=1)
    max_err = np.abs(recon - raw).max()
    oof_rawmargin_check.append((k, len(te), roc_auc_score(y[te],oof_pred[te]), max_err))
    print(f"  fold{k}: n_te={len(te)} AUC={roc_auc_score(y[te],oof_pred[te]):.4f} additivity_max_err={max_err:.2e}", flush=True)

np.save("oof_shap_values_modelB.npy", oof_shap)
np.save("oof_shap_base_values_modelB.npy", oof_base)

# 验证 SHAP 尺度: base+sum 重构的是 RawFormulaVal (log-odds), 非 probability
print(f"\nSHAP尺度: base+sum(SHAP)=RawFormulaVal (log-odds/margin), additivity max_err<{max(e[3] for e in oof_rawmargin_check):.1e}")

# additivity check csv
pd.DataFrame(oof_rawmargin_check,columns=["fold","n_val","AUC","additivity_max_err"]).to_csv("shap_additivity_check_modelB.csv",index=False)

# global importance (full OOF)
mabs = np.abs(oof_shap).mean(axis=0)
gi = pd.DataFrame({"feature":feat_names,"mean_abs_shap":mabs,"mean_shap":oof_shap.mean(axis=0)})
gi["rank"]=gi.mean_abs_shap.rank(ascending=False).astype(int)
gi["percentage"]=gi.mean_abs_shap/gi.mean_abs_shap.sum()*100
gi=gi.sort_values("rank")
gi.to_csv("oof_shap_global_importance.csv",index=False)

# manifest
man=pd.DataFrame([{"n_samples":len(y),"n_features":2283,"unique_sample_id":master.sample_id.nunique(),
    "OOF_AUC":roc_auc_score(y,oof_pred),"scale":"RawFormulaVal (log-odds margin)",
    "splitter":"StratifiedGroupKFold(5,seed42,group=smiles+solvent)",
    "has_nan":bool(np.isnan(oof_shap).any()),"has_inf":bool(np.isinf(oof_shap).any())}])
man.to_csv("oof_shap_manifest_modelB.csv",index=False)

print(f"\n=== 完整OOF SHAP完成 ===")
print(f"  shape={oof_shap.shape}, OOF_AUC={roc_auc_score(y,oof_pred):.4f}")
print(f"  top5特征:")
print(gi.head(5)[["feature","mean_abs_shap","percentage"]].to_string(index=False))
print(f"耗时{time.time()-t0:.0f}s")
