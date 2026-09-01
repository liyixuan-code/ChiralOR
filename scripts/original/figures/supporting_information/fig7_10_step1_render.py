"""
Fig7/8/10 render — real leak-free interaction tensor v2. NO simulation.
Uses frozen definitions (interaction_definition_freeze_v2.json) + precomputed
fig7 data + interaction_pair_table_v2.csv. Top-20 by pooled OOF global mean|SHAP|.
Template style: Fig7 grouped bars (#71BCB1/#9F6566); Fig8 20x20 matrix SHAP_CMAP;
Fig10 circular network (nodes=full5204 mean|SHAP|, edges=subset mean|Phi_ij|).
simulated_data_used=False. signed-geometry marked via annotation only.
"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm

_av={f.name for f in fm.fontManager.ttflist}
TNR="Times New Roman" if "Times New Roman" in _av else "DejaVu Serif"
plt.rcParams.update({"font.family":"serif","font.serif":[TNR],"svg.fonttype":"none"})
MAIN_COL="#71BCB1"; INTER_COL="#9F6566"
SHAP_CMAP=mcolors.LinearSegmentedColormap.from_list("shap",["#48A597","#FFFFFF","#9C1A1C"])
OUT="SI_Fig7_10"; os.makedirs(OUT,exist_ok=True)

feat=pd.read_csv("feature_order_check.csv").feature_name.tolist()
gi=pd.read_csv("oof_shap_global_importance.csv")
top20=gi.head(20).feature.tolist(); top20_idx=[feat.index(f) for f in top20]
gmean={f:float(gi.set_index("feature").loc[f,"mean_abs_shap"]) for f in top20}
defn=json.load(open("interaction_definition_freeze_v2.json"))
SHA=defn["tensor_sha16"]

NMAP={"geo:signed_tetra_volume":"sgn tetra vol","ECFP_1412":"ECFP 1412",
 "geo:signed_dihedral_subst_c_rn1_rn2":"sgn dih subst","geo:baseline_pm_dihedral_sin":"P/M dih sin",
 "dihedral":"baseline dih","geo:signed_dihedral_NS_path":"sgn dih N-S","sPAS_0":"sPAS-0",
 "3D_Sphero":"3D sph","PAS_5":"PAS-5","solvent_code":"solvent","PAS_10":"PAS-10","sPAS_10":"sPAS-10",
 "PAS_15":"PAS-15","MACCS_129":"MACCS-129","ECFP_492":"ECFP 492","ECFP_1465":"ECFP 1465",
 "3D_RoG":"3D RoG","TPSA":"TPSA","geo:baseline_pm_dihedral_cos":"P/M dih cos",
 "geo:subst_to_ringplane_signed_dist":"subst-ring dist"}
def lab(f): return NMAP.get(f,f.replace("geo:","")[:14])
def is_geo(f): return f.startswith("geo:") or f=="dihedral"

# ═══ FIG 7 — Main vs Interaction (real) ═══════════════════════════════════════
d7=pd.read_csv("fig7_main_vs_interaction_data_v2.csv")
d7=d7.sort_values("total",ascending=False).reset_index(drop=True)
names=[lab(f) for f in d7.feature]; geoflag=[is_geo(f) for f in d7.feature]
x=np.arange(len(names)); w=0.4
fig,ax=plt.subplots(figsize=(max(10,len(names)*0.8),6))
ax.bar(x-w/2,d7.main_effect,w,label="Main effect  mean|$\\Phi_{ii}$|",color=MAIN_COL)
ax.bar(x+w/2,d7.interaction_all_partners,w,label="Total interaction  mean$\\sum_{j\\neq i}$|$\\Phi_{ij}$| (all 2282)",color=INTER_COL)
mx=max(d7.main_effect.max(),d7.interaction_all_partners.max()); ax.set_ylim(0,mx*1.16)
for i in range(len(names)):
    ax.annotate(f"{d7.main_effect[i]:.3f}",(x[i]-w/2,d7.main_effect[i]),(0,2),textcoords="offset points",ha="center",va="bottom",fontsize=7.5)
    ax.annotate(f"{d7.interaction_all_partners[i]:.3f}",(x[i]+w/2,d7.interaction_all_partners[i]),(0,2),textcoords="offset points",ha="center",va="bottom",fontsize=7.5)
ax.set_ylabel("Magnitude (mean |SHAP|, log-odds)",fontsize=14)
ax.set_title("All features: main vs total interaction attribution (real OOF interaction, n=128)",fontsize=14,fontweight="bold")
ax.set_xticks(x)
xt=ax.set_xticklabels(names,fontsize=12,rotation=90)
for t,g in zip(xt,geoflag):      # mark signed-geometry (annotation only, rule1)
    if g: t.set_color("#1a5e54"); t.set_fontweight("bold")
ax.legend(fontsize=12,frameon=False); ax.grid(True,axis="y",ls=":",alpha=0.6)
ax.text(0.99,0.97,f"* signed-geometry in dark teal\nsimulated_data_used=False  tensor {SHA}",
    transform=ax.transAxes,ha="right",va="top",fontsize=8,color="#666")
fig.tight_layout()
for e in ["png","pdf","svg"]: fig.savefig(f"{OUT}/SI_Fig7_Main_vs_Interaction.{e}",dpi=300,bbox_inches="tight",facecolor="white")
plt.close(); print("Fig7 done")

# ═══ FIG 8 — 20x20 interaction matrix (real, mean|Phi_ij| off-diag) ═══════════
T=np.load("oof_shap_interaction_modelB_subset128_v2.npy",mmap_mode="r")
M=np.zeros((20,20))
for a in range(20):
    for b in range(20):
        M[a,b]=float(np.abs(np.asarray(T[:,top20_idx[a],top20_idx[b]])).mean())
np.save(f"{OUT}/fig8_matrix_20x20.npy",M)
off=M.copy(); np.fill_diagonal(off,np.nan)
fig,ax=plt.subplots(figsize=(12,10))
im=ax.imshow(M,cmap="magma_r",aspect="equal")
ax.set_xticks(range(20)); ax.set_yticks(range(20))
xl=ax.set_xticklabels([lab(f) for f in top20],rotation=90,fontsize=9)
yl=ax.set_yticklabels([lab(f) for f in top20],fontsize=9)
for tks,fs in [(xl,top20),(yl,top20)]:
    for t,f in zip(tks,fs):
        if is_geo(f): t.set_color("#1a5e54"); t.set_fontweight("bold")
# annotate cell values (small)
for a in range(20):
    for b in range(20):
        v=M[a,b]; ax.text(b,a,f"{v:.2f}",ha="center",va="center",fontsize=5,
            color="white" if v>M.max()*0.5 else "black")
cbar=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.03); cbar.set_label("mean |$\\Phi_{ij}$| (log-odds)",fontsize=12)
ax.set_title("SHAP interaction matrix — top-20 features (real OOF, n=128; diagonal = main effect)",fontsize=13,fontweight="bold")
fig.tight_layout()
for e in ["png","pdf","svg"]: fig.savefig(f"{OUT}/SI_Fig8_Interaction_Matrix.{e}",dpi=300,bbox_inches="tight",facecolor="white")
plt.close(); print("Fig8 done (20x20)")

# ═══ FIG 10 — interaction network ════════════════════════════════════════════
try: import networkx as nx; HAVE_NX=True
except ImportError: HAVE_NX=False
pairs=pd.read_csv("interaction_pair_table_v2.csv")
fig,ax=plt.subplots(figsize=(11,10)); ax.axis("off")
import math
angs={f:2*math.pi*k/20 for k,f in enumerate(top20)}
pos={f:(math.cos(a),math.sin(a)) for f,a in angs.items()}
emax=pairs.mean_abs_half_interaction.max()
for _,r in pairs.iterrows():
    x1,y1=pos[r.feature_i]; x2,y2=pos[r.feature_j]
    wln=r.mean_abs_half_interaction/emax
    ax.plot([x1,x2],[y1,y2],color=SHAP_CMAP(0.5+0.5*wln),lw=0.3+4.5*wln,alpha=0.25+0.6*wln,zorder=1)
nmax=max(gmean.values())
for f in top20:
    x1,y1=pos[f]; s=300+2600*(gmean[f]/nmax)
    ax.scatter(x1,y1,s=s,c="#9C1A1C" if is_geo(f) else "#48A597",edgecolors="black",lw=0.8,zorder=3,alpha=0.9)
    ax.annotate(lab(f),(x1,y1),(x1*1.18,y1*1.18),fontsize=9.5,ha="center",va="center",
        fontweight="bold" if is_geo(f) else "normal",color="#1a5e54" if is_geo(f) else "black")
ax.set_xlim(-1.45,1.45); ax.set_ylim(-1.45,1.45)
ax.set_title("Feature interaction network — top-20\n(node size = full-5204 pooled OOF mean|SHAP|; edge = subset n=128 mean|$\\Phi_{ij}$|)",
    fontsize=12,fontweight="bold")
ax.text(0,-1.4,f"red nodes = signed-geometry family   simulated_data_used=False  tensor {SHA}",ha="center",fontsize=8,color="#666")
for e in ["png","pdf","svg"]: fig.savefig(f"{OUT}/SI_Fig10_Interaction_Network.{e}",dpi=300,bbox_inches="tight",facecolor="white")
plt.close(); print(f"Fig10 done (networkx={HAVE_NX}, circular layout)")

json.dump(dict(figures=["Fig7","Fig8","Fig10"],feature_scope="top20 pooled OOF global mean|SHAP|",
    fig8_size="20x20",fig7_interaction="sum over all 2282 partners, not x2",
    fig8_offdiag="mean|Phi_ij|",fig10_node="full5204 mean|SHAP|",fig10_edge="subset mean|Phi_ij|",
    tensor_sha16=SHA,oof_leak=0,simulated_data_used=False),
    open(f"{OUT}/SI_Fig7_8_10_manifest.json","w"),indent=2)
print("manifest written")
