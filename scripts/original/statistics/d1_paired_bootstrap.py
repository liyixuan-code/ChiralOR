# -*- coding: utf-8 -*-
"""D1 paired performance uncertainty via paired GROUP/CLUSTER bootstrap.
Bootstrap unit = grouping unit of the official grouped CV = canonical_smiles ||
solvent_group. Both models in a comparison use the SAME resampled groups per
replicate. Uses existing real OOF predictions only (no re-split/re-train)."""
import os, hashlib, csv
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
FR=r"C:\Users\lenovo\.claude\projects\PM\JCIM_MANUSCRIPT\output_revision\FINAL_RELEASE"
OUT=r"C:\Users\lenovo\.claude\projects\PM\JCIM_MANUSCRIPT\output_revision\_JCIM_REVISION\ROUND2_ADDITIONAL_ANALYSIS"

pred=pd.read_csv(os.path.join(FR,"01_DATA/oof_predictions/Fig4_v2_oof_predictions_all_models.csv"))
mst =pd.read_csv(os.path.join(FR,"01_DATA/source_of_truth/master_labels_v2.csv"))
# group key identical to official grouped CV
mst["grp"]=mst.canonical_smiles+"||"+mst.solvent_group.astype(str)
g=pred.merge(mst[["sample_id","grp"]],on="sample_id",how="left")
assert g.grp.notna().all() and len(g)==5204
y=g.y_true.values
groups=g.grp.values
uniq_groups=np.unique(groups)
# map group -> row indices
gidx={u:np.where(groups==u)[0] for u in uniq_groups}
print("n=",len(g),"unique groups=",len(uniq_groups))

COMparisons=[
 ("A_Full_vs_NoSigned","CatBoost_v2 (Model B)","CatBoost_v2 baseline (2276, no signed-geom.)"),
 ("B_CatBoost_vs_RF","CatBoost_v2 (Model B)","Random Forest"),
 ("C_CatBoost_vs_ExtraTrees","CatBoost_v2 (Model B)","ExtraTrees"),
]
NBOOT=5000; RNG=np.random.RandomState(12345)

def auc(col_vals,idx): return roc_auc_score(y[idx],col_vals[idx])

rows=[]
for cid,mA,mB in COMparisons:
    pA=g[mA].values; pB=g[mB].values
    aucA=roc_auc_score(y,pA); aucB=roc_auc_score(y,pB); dobs=aucA-aucB
    deltas=[]; nvalid=0
    for _ in range(NBOOT):
        # resample GROUPS with replacement; both models share identical resample
        samp=RNG.choice(len(uniq_groups),len(uniq_groups),replace=True)
        idx=np.concatenate([gidx[uniq_groups[s]] for s in samp])
        if len(np.unique(y[idx]))<2: continue
        try:
            dd=roc_auc_score(y[idx],pA[idx])-roc_auc_score(y[idx],pB[idx])
        except ValueError: continue
        deltas.append(dd); nvalid+=1
    deltas=np.array(deltas)
    rows.append(dict(comparison=cid, model_A=mA, model_B=mB,
        AUC_model_A=round(aucA,4), AUC_model_B=round(aucB,4), delta_AUC=round(dobs,4),
        bootstrap_mean_delta=round(float(deltas.mean()),4),
        CI95_lo=round(float(np.percentile(deltas,2.5)),4),
        CI95_hi=round(float(np.percentile(deltas,97.5)),4),
        P_delta_gt_0=round(float((deltas>0).mean()),4),
        n_valid_bootstrap=nvalid, n_requested=NBOOT))
    print(f"{cid}: AUC {aucA:.4f} vs {aucB:.4f} d={dobs:+.4f} CI[{np.percentile(deltas,2.5):+.4f},{np.percentile(deltas,97.5):+.4f}] P(d>0)={ (deltas>0).mean():.3f} nvalid={nvalid}")

pd.DataFrame(rows).to_csv(os.path.join(OUT,"paired_auc_bootstrap_results.csv"),index=False,encoding="utf-8-sig")

# per-fold descriptive
fold=g.fold.values
frows=[]
for cid,mA,mB in COMparisons:
    pA=g[mA].values; pB=g[mB].values
    for k in sorted(np.unique(fold)):
        m=fold==k
        frows.append(dict(comparison=cid,fold=int(k),
            AUC_A=round(roc_auc_score(y[m],pA[m]),4),
            AUC_B=round(roc_auc_score(y[m],pB[m]),4),
            delta=round(roc_auc_score(y[m],pA[m])-roc_auc_score(y[m],pB[m]),4)))
pd.DataFrame(frows).to_csv(os.path.join(OUT,"paired_auc_fold_descriptive.csv"),index=False,encoding="utf-8-sig")
print("D1 done.")
