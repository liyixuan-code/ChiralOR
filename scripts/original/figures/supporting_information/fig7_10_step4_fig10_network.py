"""
Fig10 REBUILD — template-exact Feature Impact & Interaction Network (Class_01_1 style).
Real data (rule 9):
  nodes = top-20 by FULL-5204 pooled OOF mean|SHAP|  (node weight = importance Vimp)
  edges = subset n=128 mean|Phi_ij|                  (edge weight = interaction Vint)
Template visual (XGBoost _plot_figure_10):
  nx.circular_layout; node_size=400+1500*node_norm; edge_width=1+7*edge_norm
  node_cmap #E0F2F1->#48A597 ; edge_cmap #FCE4EC->#9C1A1C ; edges alpha0.8, colored by weight
  node edgecolor gray lw1.5 ; labels offset above/below node by size
  two horizontal colorbars: 'Importance (Vimp)' + 'Interaction Intensity (Vint)'
  title 'Feature Impact & Interaction Network' ; figsize(10,10)
simulated_data_used=False ; tensor sha 7622f7e9.
"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import networkx as nx

_av={f.name for f in fm.fontManager.ttflist}
TNR="Times New Roman" if "Times New Roman" in _av else "DejaVu Serif"
plt.rcParams.update({"font.family":"serif","font.serif":[TNR],"svg.fonttype":"none"})
OUT="SI_Fig7_10"; os.makedirs(OUT,exist_ok=True)

LABEL_FS=14; TITLE_FS=16; TICK_FS=14
NODE_COLOR="#48A597"; EDGE_COLOR="#9C1A1C"

feat=pd.read_csv("feature_order_check.csv").feature_name.tolist()
gi=pd.read_csv("oof_shap_global_importance.csv")
pairs=pd.read_csv("interaction_pair_table_v2.csv")     # real subset mean|Phi_ij| (top20-internal)
defn=json.load(open("interaction_definition_freeze_v2.json")); SHA=defn["tensor_sha16"]

NF=20
top=gi.head(NF).feature.tolist()
node_w={f:float(gi.set_index("feature").loc[f,"mean_abs_shap"]) for f in top}   # full-5204 importance
edge_w={}                                                                        # subset interaction
for _,r in pairs.iterrows():
    edge_w[frozenset((r.feature_i,r.feature_j))]=float(r.mean_abs_half_interaction)

NMAP={"geo:signed_tetra_volume":"sgn tetra vol","ECFP_1412":"ECFP 1412",
 "geo:signed_dihedral_subst_c_rn1_rn2":"sgn dih subst","geo:baseline_pm_dihedral_sin":"PM dih sin",
 "dihedral":"baseline dih","geo:signed_dihedral_NS_path":"sgn dih NS","sPAS_0":"sPAS-0",
 "3D_Sphero":"3D sph","PAS_5":"PAS-5","solvent_code":"solvent","PAS_10":"PAS-10","sPAS_10":"sPAS-10",
 "PAS_15":"PAS-15","MACCS_129":"MACCS-129","ECFP_492":"ECFP 492","ECFP_1465":"ECFP 1465",
 "3D_RoG":"3D RoG","TPSA":"TPSA","geo:baseline_pm_dihedral_cos":"PM dih cos",
 "geo:subst_to_ringplane_signed_dist":"subst-ring dist"}
def lab(f): return NMAP.get(f,f.replace("geo:","")[:14])

# build graph (template logic)
G=nx.Graph()
for f in top: G.add_node(lab(f), weight=node_w[f])
for a in range(NF):
    for b in range(a+1,NF):
        w=edge_w.get(frozenset((top[a],top[b])),0.0)
        G.add_edge(lab(top[a]),lab(top[b]),weight=w)

pos=nx.circular_layout(G)
node_weights=np.array([G.nodes[n]["weight"] for n in G.nodes])
edge_weights=np.array([G[u][v]["weight"] for u,v in G.edges])
node_norm=(node_weights-node_weights.min())/(node_weights.max()-node_weights.min()+1e-8)
edge_norm=(edge_weights-edge_weights.min())/(edge_weights.max()-edge_weights.min()+1e-8)
node_sizes=400+1500*node_norm
edge_widths=1+7*edge_norm
node_cmap=mcolors.LinearSegmentedColormap.from_list("nc",["#E0F2F1",NODE_COLOR])
edge_cmap=mcolors.LinearSegmentedColormap.from_list("ec",["#FCE4EC",EDGE_COLOR])
node_list=list(G.nodes)

fig=plt.figure(figsize=(10,10))
ax=fig.add_axes([0.1,0.15,0.8,0.8])
nx.draw_networkx_edges(G,pos,width=edge_widths,edge_color=edge_weights,edge_cmap=edge_cmap,alpha=0.8,ax=ax)
nx.draw_networkx_nodes(G,pos,node_size=node_sizes,node_color=node_weights,cmap=node_cmap,
    edgecolors="gray",linewidths=1.5,ax=ax)
# place each label RADIALLY OUTSIDE its node (along the ring radius), so labels
# ring the circle and never cross the interior interaction lines.
import math as _m
for node,(x,y) in pos.items():
    r=_m.hypot(x,y) or 1.0
    ux,uy=x/r,y/r                      # unit radial direction (outward)
    gap=0.14+(node_sizes[node_list.index(node)]/40000.0)
    lx,ly=x+ux*gap, y+uy*gap           # push label outward past the node marker
    # horizontal alignment by x-direction, vertical by y-direction -> text sits outside
    ha = "left" if ux>0.15 else ("right" if ux<-0.15 else "center")
    va = "bottom" if uy>0.15 else ("top" if uy<-0.15 else "center")
    ax.text(lx,ly,node,ha=ha,va=va,fontsize=TICK_FS,zorder=6)
ax.set_title("Feature Impact & Interaction Network",fontsize=TITLE_FS)
ax.axis("off")
ax.set_xlim(-1.45,1.45); ax.set_ylim(-1.45,1.45)   # room for outer labels

cax1=fig.add_axes([0.15,0.10,0.3,0.015])
sm1=plt.cm.ScalarMappable(cmap=node_cmap,norm=plt.Normalize(node_weights.min(),node_weights.max())); sm1.set_array([])
cb1=fig.colorbar(sm1,cax=cax1,orientation="horizontal"); cb1.set_label("Importance (mean|SHAP|, full n=5204)",fontsize=LABEL_FS-2); cb1.ax.tick_params(labelsize=TICK_FS-3)
cax2=fig.add_axes([0.55,0.10,0.3,0.015])
sm2=plt.cm.ScalarMappable(cmap=edge_cmap,norm=plt.Normalize(edge_weights.min(),edge_weights.max())); sm2.set_array([])
cb2=fig.colorbar(sm2,cax=cax2,orientation="horizontal"); cb2.set_label("Interaction intensity (mean|$\\Phi_{ij}$|, subset n=128)",fontsize=LABEL_FS-2); cb2.ax.tick_params(labelsize=TICK_FS-3)

for e in ["png","pdf","svg"]:
    fig.savefig(f"{OUT}/SI_Fig10_Interaction_Network.{e}",dpi=300,bbox_inches="tight",facecolor="white")
plt.close()
print(f"Fig10 REBUILT (template Class_01_1 style, networkx circular). sha={SHA} sim=False")
print(f"  nodes={NF} (full-5204 mean|SHAP|)  edges={len(edge_weights)} (subset mean|Phi_ij|)")
print(f"  node_w[{node_weights.min():.3f},{node_weights.max():.3f}]  edge_w[{edge_weights.min():.4f},{edge_weights.max():.4f}]")
