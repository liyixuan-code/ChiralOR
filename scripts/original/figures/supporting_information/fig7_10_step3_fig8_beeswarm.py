"""
Fig8 REBUILD — OLD-JCIM beeswarm interaction-matrix form, REAL data (replaces v5 simulation).
NF x NF subplot grid (top-15 by pooled OOF global mean|SHAP|):
  lower triangle (row>col): colored cell = real mean|Phi_ij|, bold .3f text
  diagonal+upper (row<=col): deterministic beeswarm scatter (200-bin, no jitter)
     x = real T[:,i,i] (diagonal main) OR real 2*T[:,i,j] (upper, full pairwise)
     color = real feature value X[sub_ids,row], percentile5-95 normalized
     draw order: c ascending (green first, red on top)
  labels right/bottom, single colorbar, template SHAP_CMAP #48A597->white->#9C1A1C.
Real leak-free tensor v2 (sha 7622f7e9). simulated_data_used=False.
"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm

_av={f.name for f in fm.fontManager.ttflist}
SANS="Arial" if "Arial" in _av else "DejaVu Sans"
matplotlib.rcParams.update({"font.family":"sans-serif","font.sans-serif":[SANS],
    "svg.fonttype":"none","axes.linewidth":0.4})
SHAP_CMAP=mcolors.LinearSegmentedColormap.from_list("shap",["#48A597","#FFFFFF","#9C1A1C"])
OUT="SI_Fig7_10"; os.makedirs(OUT,exist_ok=True)

feat=pd.read_csv("feature_order_check.csv").feature_name.tolist()
gi=pd.read_csv("oof_shap_global_importance.csv")
sub=pd.read_csv("interaction_subset_manifest_seed42.csv"); sub_ids=sub.sample_id.values
X=np.load("X_C3_5204x2283.npy")
T=np.load("oof_shap_interaction_modelB_subset128_v2.npy",mmap_mode="r")
defn=json.load(open("interaction_definition_freeze_v2.json")); SHA=defn["tensor_sha16"]

NF=15
top=gi.head(NF).feature.tolist(); tidx=[feat.index(f) for f in top]
NMAP={"geo:signed_tetra_volume":"sgn tetra vol","ECFP_1412":"ECFP 1412",
 "geo:signed_dihedral_subst_c_rn1_rn2":"sgn dih subst","geo:baseline_pm_dihedral_sin":"PM dih sin",
 "dihedral":"baseline dih","geo:signed_dihedral_NS_path":"sgn dih NS","sPAS_0":"sPAS-0",
 "3D_Sphero":"3D sph","PAS_5":"PAS-5","solvent_code":"solvent","PAS_10":"PAS-10","sPAS_10":"sPAS-10",
 "PAS_15":"PAS-15","MACCS_129":"MACCS-129","ECFP_492":"ECFP 492"}
def lab(f): return NMAP.get(f,f.replace("geo:","")[:12])
FEAT=[lab(f) for f in top]

# real slices: interaction NFxNF over 128, and real feature values (percentile 5-95 norm)
Ti=np.zeros((128,NF,NF))
for a in range(NF):
    for b in range(NF):
        Ti[:,a,b]=np.asarray(T[:,tidx[a],tidx[b]])
Xv=X[np.ix_(sub_ids,tidx)]                      # (128,NF) real feature values
Xn=np.zeros_like(Xv)
for j in range(NF):
    lo,hi=np.percentile(Xv[:,j],5),np.percentile(Xv[:,j],95)
    Xn[:,j]=np.clip((Xv[:,j]-lo)/(hi-lo+1e-9),0,1)
mabs_inter=np.mean(np.abs(Ti),axis=0)           # (NF,NF) real mean|Phi_ij|
vals_lt=[mabs_inter[r,c] for r in range(NF) for c in range(NF) if r>c]
vmax_lt=np.percentile(vals_lt,97) if vals_lt else 1e-9

def beeswarm_y(x,c,nbins=200,ylim=0.45):
    counts,edges=np.histogram(x,bins=nbins)
    bi=np.clip(np.digitize(x,edges)-1,0,nbins-1); mc=max(counts.max(),1); y=np.zeros(len(x))
    for b in range(nbins):
        m=np.where(bi==b)[0]; n=len(m)
        if n<2: continue
        order=m[np.argsort(c[m])]; w=(n/mc)*ylim; y[order]=np.linspace(-w,w,n)
    return y

fig,axes=plt.subplots(NF,NF,figsize=(14,14))
plt.subplots_adjust(wspace=0.08,hspace=0.08,bottom=0.20,left=0.22,top=0.92)
for row in range(NF):
    for col in range(NF):
        ax=axes[row,col]; ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_linewidth(0.3); sp.set_edgecolor("#CCCCCC")
        if row>col:
            ax.set_xticks([])
            iv=mabs_inter[row,col]; nv=min(0.5+0.5*(iv/(vmax_lt+1e-12)),0.999)
            bg=SHAP_CMAP(nv); ax.set_facecolor(bg)
            lum=0.299*bg[0]+0.587*bg[1]+0.114*bg[2]
            ax.text(0.5,0.5,f"{iv:.3f}",ha="center",va="center",transform=ax.transAxes,
                fontsize=11,fontweight="bold",color="white" if lum<0.55 else "#222")
        else:
            ax.set_facecolor("white")
            ax.grid(True,color="lightgray",ls="-",lw=0.8,zorder=0)   # template grid lw0.8
            xvals=Ti[:,row,row] if row==col else Ti[:,row,col]*2   # real: main diag, full pair upper
            cvals=Xn[:,row]
            yy=beeswarm_y(xvals,cvals); order=np.argsort(cvals)
            ax.scatter(xvals[order],yy[order],c=cvals[order],cmap=SHAP_CMAP,vmin=0,vmax=1,
                s=8,alpha=0.7,edgecolors="none",zorder=3)     # template s=8 alpha=0.7
            ax.axhline(0,color="gray",lw=0.5,ls="-",alpha=0.3,zorder=1)
            ax.axvline(0,color="gray",lw=0.8,ls="--",zorder=1)
            ax.set_ylim(-0.6,0.6)
            if row==0:
                ax.xaxis.set_ticks_position("top")
                ax.tick_params(axis="x",top=True,labeltop=True,bottom=False,labelbottom=False,labelsize=4.5,pad=1.5)
                ax.locator_params(axis="x",nbins=3)
            elif row==NF-1:
                ax.xaxis.set_ticks_position("bottom")
                ax.tick_params(axis="x",top=False,labeltop=False,bottom=True,labelbottom=True,labelsize=4.5,pad=1.5)
                ax.locator_params(axis="x",nbins=3)
            else: ax.set_xticks([])
# template-exact labels: y on LEFT (col 0) rotation 0 ha right; x on bottom row rotation 90
# fig8_label_fontsize=16. LEFT placement keeps them clear of the right-side colorbar.
LFS=16
for i in range(NF):
    axes[i][0].set_ylabel(FEAT[i],fontsize=LFS,labelpad=15,rotation=0,ha="right",va="center")
    axes[i][0].yaxis.set_label_position("left")
for j in range(NF):
    axes[NF-1][j].set_xlabel(FEAT[j],fontsize=LFS,labelpad=10,rotation=90,ha="center",va="top")
cax=fig.add_axes([0.935,0.24,0.013,0.52])
sm=plt.cm.ScalarMappable(cmap=SHAP_CMAP,norm=mcolors.Normalize(0,1)); sm.set_array([])
cb=fig.colorbar(sm,cax=cax); cb.set_label("Raw feature value / Mean |Interaction|",rotation=270,labelpad=14,fontsize=8)
cb.set_ticks([0,1]); cb.set_ticklabels(["Low","High"]); cb.ax.tick_params(labelsize=13)
fig.suptitle(f"SHAP interaction matrix  (top {NF} of 2,283 features,  real OOF n=128, tensor {SHA})",
    fontsize=9.5,fontweight="bold",y=0.957)
for ext,dpi in [("png",600),("pdf",300),("svg",300)]:
    fig.savefig(f"{OUT}/SI_Fig8_Interaction_Matrix.{ext}",dpi=dpi,bbox_inches="tight",facecolor="white")
plt.close()
np.save(f"{OUT}/fig8_matrix_{NF}x{NF}_real.npy",mabs_inter)
print(f"Fig8 REBUILT (old-JCIM beeswarm form, {NF}x{NF}, REAL data). sha={SHA} sim=False")
print(f"  max mean|Phi_ij| lower-tri={max(vals_lt):.4f}  vmax97={vmax_lt:.4f}")
