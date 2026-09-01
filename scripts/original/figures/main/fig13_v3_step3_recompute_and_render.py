"""
Fig13 REVIEW V3 — STEP 3: recompute ALL statistics from S4 (primary) + render 2x3 figure.
Runs AFTER step1 (seed-averaged S4 primary surfaces in consistency/pairNN_S4primary_meanoverseeds.npy).
S4 = empirical-background averaged fold-ensemble bivariate conditional model-response surface.
  item 1: recompute supported Q1/median/Q3/range, P=0.5 contour, shared color range FROM S4
  item 2: NO Max/Min markers on submission
  item 4: single 2x3 panel Fig S13 (a..f); panel letter + short pair title only;
          one shared legend + one shared colorbar; NO per-panel long term / [MANIFOLD...] text
  item 5: TwoSlopeNorm(vcenter=0.5, vmin=global supported S4 min, vmax=global supported S4 max)
  item K: subtle empirical-support boundary only
simulated_data_used=False.
"""
import os, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors

ROOT="SI_Fig13_STYLELOCK/plot_data"; CONS="SI_Fig13_REVIEW_V3/consistency"
SUBF="SI_Fig13_REVIEW_V3/submission_final"; STAT="SI_Fig13_REVIEW_V3/statistics"
os.makedirs(SUBF,exist_ok=True); os.makedirs(STAT,exist_ok=True)
feat=pd.read_csv("feature_order_check.csv").feature_name.tolist()
pairs=pd.read_csv("SI_Fig13_pair_selection.csv")
SHAP_CMAP=mcolors.LinearSegmentedColormap.from_list("shap",["#48A597","#FFFFFF","#9C1A1C"])
SUPP_COL="#8A8A8A"
SHORT={"geo:signed_tetra_volume":"sgn tetra vol","geo:subst_to_ringplane_signed_dist":"subst-ring dist",
 "geo:baseline_pm_dihedral_sin":"P/M dih (sin)","geo:signed_dihedral_subst_c_rn1_rn2":"sgn dih subst",
 "3D_Sphero":"3D spherocity","PAS_5":"PAS-5","PAS_15":"PAS-15"}
def sh(f): return SHORT.get(f,f.replace("geo:","")[:14])

# load S4 primary per pair + supported mask/grids
D={}
gmin,gmax=1.0,0.0
for _,pr in pairs.iterrows():
    p=int(pr.pair_rank)
    npz=np.load(f"{ROOT}/pair{p:02d}_surface_values.npz",allow_pickle=True)
    # prefer FULL-background deterministic primary; fall back to seed-averaged
    _full=f"{CONS}/pair{p:02d}_S4primary_fullbg.npy"
    _seed=f"{CONS}/pair{p:02d}_S4primary_meanoverseeds.npy"
    S4=np.load(_full) if os.path.exists(_full) else np.load(_seed)
    sup=npz["supported"].astype(bool)
    D[p]=dict(XX=npz["XX"],YY=npz["YY"],S4=S4,sup=sup,fx=str(npz["feature_x"]),fy=str(npz["feature_y"]))
    sv=S4[sup]
    if sv.size: gmin=min(gmin,float(sv.min())); gmax=max(gmax,float(sv.max()))
# item 5: TwoSlopeNorm centered at 0.5. vmin<vcenter<vmax is required by matplotlib,
# so if the data lie entirely on one side of 0.5 (here full-bg gmax<0.5), extend the
# empty side minimally to keep 0.5 as the white centre (does NOT alter surface values).
vlo=min(gmin,0.5-1e-3); vhi=max(gmax,0.5+1e-3)
NORM=mcolors.TwoSlopeNorm(vcenter=0.5,vmin=vlo,vmax=vhi)
VMIN,VMAX=vlo,vhi
print(f"S4 shared supported range [{gmin:.3f},{gmax:.3f}] -> TwoSlopeNorm(0.5,{vlo:.3f},{vhi:.3f})"
      + ("  [data entirely <0.5: teal side]" if gmax<0.5 else ""))

# item 1: recompute supported statistics FROM S4
stat=[]
for p in sorted(D):
    d=D[p]; sv=d["S4"][d["sup"]]
    stat.append(dict(pair_id=p,feature_x=d["fx"],feature_y=d["fy"],surface="S4_empbg_foldens_primary",
        n_grid_supported=int(d["sup"].sum()),
        supported_min=round(float(sv.min()),4),supported_q1=round(float(np.percentile(sv,25)),4),
        supported_median=round(float(np.percentile(sv,50)),4),supported_q3=round(float(np.percentile(sv,75)),4),
        supported_max=round(float(sv.max()),4)))
pd.DataFrame(stat).to_csv(f"{STAT}/SI_Fig13_S4_supported_region_statistics.csv",index=False)
statmap={s["pair_id"]:s for s in stat}

levels=np.linspace(VMIN,VMAX,61)
letters=["(a)","(b)","(c)","(d)","(e)","(f)"]
p50_note={}   # pair -> whether P=0.5 drawn

def draw_panel(ax,p,template_labels=True,letter=None):
    """One panel in template Fig13 visual style. Contours drawn on SUPPORTED region only:
    surface is masked to supported cells before contouring so unsupported area yields no
    Q1/med/Q3/P=0.5 contour. Full rectangular surface kept as background; unsupported cells
    lightly desaturated (white veil, no hatch/dots)."""
    d=D[p]; XX,YY,S4,sup=d["XX"],d["YY"],d["S4"],d["sup"]
    cf=ax.contourf(XX,YY,S4,levels=levels,cmap=SHAP_CMAP,norm=NORM,extend="both",zorder=1)
    # unsupported region: light desaturation only (item: no hatch/dots)
    if sup.any() and not sup.all():
        ax.contourf(XX,YY,np.ma.masked_where(sup,np.ones_like(S4)),levels=[0,2],
                    colors=[(1,1,1,0.35)],zorder=2)
        ax.contour(XX,YY,sup.astype(float),levels=[0.5],colors=[SUPP_COL],linewidths=0.7,alpha=0.45,zorder=3)
    # contours on SUPPORTED-ONLY masked surface
    Smask=np.where(sup,S4,np.nan)
    s=statmap[p]
    for lv,c,ls in [(s["supported_q1"],"#2A6F97","--"),(s["supported_median"],"#48A597","-"),(s["supported_q3"],"#9C1A1C","--")]:
        sv=S4[sup]
        if sv.min()<lv<sv.max(): ax.contour(XX,YY,Smask,levels=[lv],colors=[c],linestyles=[ls],linewidths=1.2,alpha=0.85,zorder=4)
    # P=0.5 only if a crossing truly exists inside supported region
    sv=S4[sup]; has50=bool(sv.min()<0.5<sv.max()); p50_note[p]=has50
    if has50: ax.contour(XX,YY,Smask,levels=[0.5],colors=["black"],linestyles=[":"],linewidths=1.6,zorder=5)
    ax.set_xlabel(sh(d["fx"]),fontsize=13); ax.set_ylabel(sh(d["fy"]),fontsize=13)
    ttl=(f"{letter} " if letter else "")+f"{sh(d['fx'])} × {sh(d['fy'])}"
    ax.set_title(ttl,fontsize=12,fontweight="bold"); ax.tick_params(labelsize=11)
    return cf

def legend_handles(with_p50):
    h=[mlines.Line2D([],[],color="#2A6F97",ls="--",label="supported Q1"),
       mlines.Line2D([],[],color="#48A597",ls="-",label="supported median"),
       mlines.Line2D([],[],color="#9C1A1C",ls="--",label="supported Q3"),
       mlines.Line2D([],[],color=SUPP_COL,lw=0.7,alpha=0.6,label="empirical-support boundary")]
    if with_p50: h.insert(3,mlines.Line2D([],[],color="black",ls=":",label="P(OR+) = 0.50 (only if supported crossing)"))
    return h

# ── (A) six single-panel template-style figures (7x6, own colorbar+legend) ──
for k,p in enumerate(sorted(D)):
    fig,ax=plt.subplots(figsize=(7,6))
    cf=draw_panel(ax,p,letter=None)
    ax.legend(handles=legend_handles(p50_note[p]),loc="lower right",bbox_to_anchor=(1.0,1.02),
              ncol=2,fontsize=8.5,frameon=True,facecolor=(1,1,1,0.85),edgecolor="#cccccc")
    cb=fig.colorbar(cf,ax=ax,pad=0.03,ticks=sorted({round(gmin,3),0.5,round(gmax,3)}))
    cb.set_label("Fold-ensemble P(OR+)  (empirical background)",fontsize=12)
    d=D[p]
    stem=f"{SUBF}/SI_FigS13_{p:02d}_{d['fx'].replace('geo:','geo_').replace(':','_')}_vs_{d['fy'].replace('geo:','geo_').replace(':','_')}"
    for e in ["png","pdf","svg"]: fig.savefig(f"{stem}.{e}",dpi=300,bbox_inches="tight",facecolor="white")
    plt.close()

# ── (B) 2x3 combined (shared legend + shared colorbar) ──────────────────────
fig,axes=plt.subplots(2,3,figsize=(16,10)); axes=axes.flatten(); cf=None
for k,p in enumerate(sorted(D)):
    cf=draw_panel(axes[k],p,letter=letters[k])
any50=any(p50_note.values())
fig.legend(handles=legend_handles(any50),loc="upper center",ncol=5,fontsize=11,frameon=True,
           facecolor=(1,1,1,0.9),edgecolor="#cccccc",bbox_to_anchor=(0.5,1.0))
fig.subplots_adjust(left=0.05,right=0.90,top=0.90,bottom=0.07,wspace=0.28,hspace=0.30)
cax=fig.add_axes([0.92,0.12,0.017,0.72])
cb=fig.colorbar(cf,cax=cax,ticks=sorted({round(gmin,3),0.5,round(gmax,3)}))
cb.set_label("Fold-ensemble P(OR+)  (empirical background)",fontsize=12)
for e in ["png","pdf","svg"]:
    fig.savefig(f"{SUBF}/SI_FigS13_2x3_empbg_foldensemble.{e}",dpi=300,bbox_inches="tight",facecolor="white")
plt.close()
pd.DataFrame([dict(pair_id=p,P50_line_drawn=p50_note[p]) for p in sorted(D)]).to_csv(
    f"{STAT}/SI_Fig13_P50_contour_presence.csv",index=False)
print("Rendered 6 single-panel (template style) + 1 combined 2x3. P=0.50 drawn per pair:",
      {p:p50_note[p] for p in sorted(D)})
for s in stat:
    print(f"  pair{s['pair_id']}: SUPP[{s['supported_min']:.2f},{s['supported_max']:.2f}] "
          f"Q1/med/Q3={s['supported_q1']:.2f}/{s['supported_median']:.2f}/{s['supported_q3']:.2f}")
