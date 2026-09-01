# -*- coding: utf-8 -*-
"""D2 OOF probability calibration ASSESSMENT (no re-fit).
Uses final CatBoost_v2 grouped-OOF p(OR+) from oof_predictions_master_labels_v2.
Clipping only for logit; original probabilities unchanged. Group-aware bootstrap
CI. Prefixed 10-bin equal-frequency (primary) + 10-bin equal-width (sensitivity)."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
import statsmodels.api as sm
FR=r"C:\Users\lenovo\.claude\projects\PM\JCIM_MANUSCRIPT\output_revision\FINAL_RELEASE"
OUT=r"C:\Users\lenovo\.claude\projects\PM\JCIM_MANUSCRIPT\output_revision\_JCIM_REVISION\ROUND2_ADDITIONAL_ANALYSIS"

o=pd.read_csv(os.path.join(FR,"01_DATA/oof_predictions/oof_predictions_master_labels_v2.csv"))
m=pd.read_csv(os.path.join(FR,"01_DATA/source_of_truth/master_labels_v2.csv"))
m["grp"]=m.canonical_smiles+"||"+m.solvent_group.astype(str)
o=o.merge(m[["sample_id","grp"]],on="sample_id",how="left")
y=o.y_true.values.astype(float); p=o.oof_prob.values.astype(float); fold=o.fold.values
groups=o.grp.values; uniq=np.unique(groups); gidx={u:np.where(groups==u)[0] for u in uniq}
EPS=1e-6
def logit(x): x=np.clip(x,EPS,1-EPS); return np.log(x/(1-x))

def cal_slope_intercept(y,p):
    # logistic regression of y on logit(p): slope & intercept (diagnostic fit)
    X=sm.add_constant(logit(p))
    res=sm.GLM(y,X,family=sm.families.Binomial()).fit()
    return res.params[1], res.params[0]  # slope, intercept

def ece(y,p,nbin=10,strategy="quantile"):
    if strategy=="quantile":
        edges=np.quantile(p,np.linspace(0,1,nbin+1)); edges[0],edges[-1]=0,1
    else:
        edges=np.linspace(0,1,nbin+1)
    e=0.0; N=len(p)
    for i in range(nbin):
        lo,hi=edges[i],edges[i+1]
        mask=(p>=lo)&(p<hi) if i<nbin-1 else (p>=lo)&(p<=hi)
        if mask.sum()==0: continue
        e += mask.sum()/N*abs(y[mask].mean()-p[mask].mean())
    return e

brier=brier_score_loss(y,p)
slope,intercept=cal_slope_intercept(y,p)
ece_q=ece(y,p,10,"quantile"); ece_w=ece(y,p,10,"uniform")
print(f"Brier={brier:.4f} slope={slope:.4f} intercept={intercept:.4f} ECE_q={ece_q:.4f} ECE_w={ece_w:.4f}")

pd.DataFrame([dict(metric="Brier",value=round(brier,4)),
  dict(metric="calibration_slope",value=round(slope,4)),
  dict(metric="calibration_intercept",value=round(intercept,4)),
  dict(metric="ECE_10bin_equalfreq_PRIMARY",value=round(ece_q,4)),
  dict(metric="ECE_10bin_equalwidth_SENS",value=round(ece_w,4)),
  ]).to_csv(os.path.join(OUT,"calibration_metrics_oof.csv"),index=False,encoding="utf-8-sig")

# group-aware bootstrap CI
NB=2000; RNG=np.random.RandomState(2024); B=[];S=[];I=[];EQ=[]
for _ in range(NB):
    samp=RNG.choice(len(uniq),len(uniq),replace=True)
    idx=np.concatenate([gidx[uniq[s]] for s in samp])
    if len(np.unique(y[idx]))<2: continue
    try:
        sl,ic=cal_slope_intercept(y[idx],p[idx])
    except Exception: continue
    B.append(brier_score_loss(y[idx],p[idx])); S.append(sl); I.append(ic); EQ.append(ece(y[idx],p[idx],10,"quantile"))
def ci(a): a=np.array(a); return round(float(np.percentile(a,2.5)),4),round(float(np.percentile(a,97.5)),4)
rows=[]
for nm,arr,pt in [("Brier",B,brier),("calibration_slope",S,slope),("calibration_intercept",I,intercept),("ECE_10bin_equalfreq",EQ,ece_q)]:
    lo,hi=ci(arr); rows.append(dict(metric=nm,point=round(pt,4),CI95_lo=lo,CI95_hi=hi,n_valid=len(arr)))
pd.DataFrame(rows).to_csv(os.path.join(OUT,"calibration_bootstrap_ci.csv"),index=False,encoding="utf-8-sig")

# reliability data (10-bin equal-frequency) + per-fold
edges=np.quantile(p,np.linspace(0,1,11)); edges[0],edges[-1]=0,1
rel=[]
for i in range(10):
    lo,hi=edges[i],edges[i+1]
    mask=(p>=lo)&(p<hi) if i<9 else (p>=lo)&(p<=hi)
    if mask.sum()==0: continue
    rel.append(dict(bin=i+1,p_lo=round(lo,4),p_hi=round(hi,4),n=int(mask.sum()),
        mean_pred=round(float(p[mask].mean()),4),obs_freq=round(float(y[mask].mean()),4)))
pd.DataFrame(rel).to_csv(os.path.join(OUT,"calibration_reliability_data.csv"),index=False,encoding="utf-8-sig")

# reliability plot
fig,ax=plt.subplots(figsize=(5.2,5.2))
rd=pd.DataFrame(rel)
ax.plot([0,1],[0,1],'--',color='gray',lw=1,label='Perfect calibration')
ax.plot(rd.mean_pred,rd.obs_freq,'o-',color='#9F3E3F',label='OOF (10-bin equal-freq)')
ax.set_xlabel('Mean predicted p(OR+)');ax.set_ylabel('Observed OR+ frequency')
ax.set_title(f'OOF reliability (Brier={brier:.3f}, slope={slope:.2f}, ECE={ece_q:.3f})',fontsize=10)
ax.legend(frameon=False,fontsize=8);ax.set_xlim(0,1);ax.set_ylim(0,1)
fig.tight_layout()
for ext in ("png","pdf","svg"):
    fig.savefig(os.path.join(OUT,f"calibration_reliability_plot.{ext}"),dpi=300,bbox_inches="tight")
plt.close()
print("D2 done. slope/intercept = diagnostic fit on pooled OOF (NOT a re-calibrated deployment model).")
