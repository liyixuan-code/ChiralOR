"""
Fig13 V3 STEP 7 — batch Candidate A (strict) to ALL 6 pairs + regenerate 2x3 combined.
Approved style: supported-only Q1/Med/Q3 fragments (clipped, NOT joined), prominent halo
Max/Min from real supported extrema, 5-item legend OUTSIDE axes, shared normalization,
faint support boundary (not in legend), no P=0.5 (no supported crossing), no title.
Data frozen: surface/grid/support/quartiles/extrema/normalization unchanged.
Outputs -> SI_Fig13_REVIEW_V3/submission_final/  (single panels + 2x3)
         -> SI_Fig13_REVIEW_V3/statistics/SI_Fig13_all_extrema_audit.csv
"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.colors as mcolors, matplotlib.lines as mlines
import matplotlib.font_manager as fm

ROOT="SI_Fig13_STYLELOCK/plot_data"; CONS="SI_Fig13_REVIEW_V3/consistency"
STAT="SI_Fig13_REVIEW_V3/statistics"; SUBF="SI_Fig13_REVIEW_V3/submission_final"
os.makedirs(SUBF,exist_ok=True)
TNR="Times New Roman" if "Times New Roman" in {f.name for f in fm.fontManager.ttflist} else "DejaVu Serif"
plt.rcParams.update({"font.family":"serif","font.serif":[TNR],"svg.fonttype":"none"})
Q1C,MEDC,Q3C="#2A6F97","#48A597","#9C1A1C"; SUPP="#8A8A8A"; LFS=14; TFS=14
SHAP_CMAP=mcolors.LinearSegmentedColormap.from_list("shap",["#48A597","#FFFFFF","#9C1A1C"])

pairs=pd.read_csv("SI_Fig13_pair_selection.csv")
stat=pd.read_csv(f"{STAT}/SI_Fig13_S4_supported_region_statistics.csv").set_index("pair_id")

NAME={"geo:signed_tetra_volume":"Signed tetrahedral volume",
 "geo:subst_to_ringplane_signed_dist":"Signed substituent-to-ring-plane distance",
 "geo:baseline_pm_dihedral_sin":"P/M dihedral sine","geo:signed_dihedral_subst_c_rn1_rn2":"Signed dihedral (subst–ring)",
 "3D_Sphero":"3D sphericity","PAS_5":"PAS-5","PAS_15":"PAS-15"}
SHORT={"geo:signed_tetra_volume":"Signed tetra volume","geo:subst_to_ringplane_signed_dist":"Subst-to-ring dist",
 "geo:baseline_pm_dihedral_sin":"P/M dihedral sine","geo:signed_dihedral_subst_c_rn1_rn2":"Signed dih (subst–ring)",
 "3D_Sphero":"3D sphericity","PAS_5":"PAS-5","PAS_15":"PAS-15"}
def nm(f): return NAME.get(f,f.replace("geo:","").replace("_"," "))
def sh(f): return SHORT.get(f,nm(f))
def clean(f): return f.replace("geo:","geo_").replace(":","_")

# load surfaces + shared global normalization
D={}; gmin,gmax=1.0,0.0
for _,pr in pairs.iterrows():
    p=int(pr.pair_rank); nz=np.load(f"{ROOT}/pair{p:02d}_surface_values.npz",allow_pickle=True)
    S=np.load(f"{CONS}/pair{p:02d}_S4primary_fullbg.npy"); sup=nz["supported"].astype(bool)
    D[p]=dict(XX=nz["XX"],YY=nz["YY"],S=S,sup=sup,fx=str(nz["feature_x"]),fy=str(nz["feature_y"]))
    sv=S[sup]; gmin=min(gmin,sv.min()); gmax=max(gmax,sv.max())
VMIN,VMAX=float(gmin),float(gmax)
NORM=mcolors.TwoSlopeNorm(0.5,min(VMIN,0.5-1e-3),max(VMAX,0.5+1e-3))
LEVELS=np.linspace(VMIN,VMAX,60); CBAR_TICKS=np.round(np.linspace(VMIN,VMAX,6),3)
print(f"shared norm supported [{VMIN:.3f},{VMAX:.3f}]")

def clip_paths_for(p):
    d=D[p]; f0,a0=plt.subplots(); cc=a0.contourf(d["XX"],d["YY"],d["sup"].astype(float),levels=[0.5,1.5],colors="none")
    cps=[pp for pp in cc.get_paths() if len(pp.vertices)>2]; plt.close(f0); return cps

extrema_rows=[]
def draw_core(ax,p,label_font=LFS,short=False,for_grid=False):
    """Draw one panel's surface+contours+extrema. Returns (Pmax,Pmin,mx,mn,handles)."""
    d=D[p]; XX,YY,S,sup=d["XX"],d["YY"],d["S"],d["sup"]; s=stat.loc[p]
    q1,med,q3=float(s.supported_q1),float(s.supported_median),float(s.supported_q3); sv=S[sup]
    cf=ax.contourf(XX,YY,S,levels=LEVELS,cmap=SHAP_CMAP,norm=NORM,alpha=0.9,zorder=1)
    if sup.any() and not sup.all():
        ax.contourf(XX,YY,np.ma.masked_where(sup,np.ones_like(S)),levels=[0,2],colors=[(1,1,1,0.06)],zorder=2)
        ax.contour(XX,YY,sup.astype(float),levels=[0.5],colors=[SUPP],linewidths=0.6,alpha=0.22,zorder=3)
    cps=clip_paths_for(p)
    for lv,c,ls,lw in [(q1,Q1C,"--",1.6),(med,MEDC,"-",2.2),(q3,Q3C,"--",1.6)]:
        if sv.min()<lv<sv.max():
            cs=ax.contour(XX,YY,S,levels=[lv],colors=[c],linestyles=[ls],linewidths=lw,alpha=0.95,zorder=6)
            if cps:
                try: cs.set_clip_path(cps[0],ax.transData)
                except Exception:
                    for coll in getattr(cs,"collections",[]): coll.set_clip_path(cps[0],ax.transData)
    ms=np.where(sup,S,np.nan); mx=np.unravel_index(np.nanargmax(ms),ms.shape); mn=np.unravel_index(np.nanargmin(ms),ms.shape)
    Pmax,Pmin=float(S[mx]),float(S[mn])
    hs=520 if not for_grid else 360; ho=340 if not for_grid else 230
    ax.scatter([XX[mx]],[YY[mx]],marker="*",s=hs,color="white",zorder=18,linewidths=0,clip_on=False)
    ax.scatter([XX[mx]],[YY[mx]],marker="*",s=ho,facecolor="#FF8F00",edgecolor="black",linewidth=1.2,zorder=20,clip_on=False)
    ax.scatter([XX[mn]],[YY[mn]],marker="o",s=(300 if not for_grid else 210),color="white",zorder=18,linewidths=0,clip_on=False)
    ax.scatter([XX[mn]],[YY[mn]],marker="o",s=(150 if not for_grid else 100),facecolor="#00BCD4",edgecolor="black",linewidth=1.4,zorder=20,clip_on=False)
    handles=[mlines.Line2D([],[],color=Q1C,ls="--",lw=1.6,label=f"Q1: {q1:.2f}"),
             mlines.Line2D([],[],color=MEDC,ls="-",lw=2.2,label=f"Median: {med:.2f}"),
             mlines.Line2D([],[],color=Q3C,ls="--",lw=1.6,label=f"Q3: {q3:.2f}"),
             mlines.Line2D([],[],color="none",marker="*",markerfacecolor="#FF8F00",markeredgecolor="black",markersize=14,label=f"Max: {Pmax:.2f}"),
             mlines.Line2D([],[],color="none",marker="o",markerfacecolor="#00BCD4",markeredgecolor="black",markersize=9,label=f"Min: {Pmin:.2f}")]
    xlo,xhi=XX.min(),XX.max(); ylo,yhi=YY.min(),YY.max()
    ax.set_xlim(xlo-0.03*(xhi-xlo),xhi+0.03*(xhi-xlo)); ax.set_ylim(ylo-0.03*(yhi-ylo),yhi+0.03*(yhi-ylo))
    ax.set_xlabel(sh(d["fx"]) if short else nm(d["fx"]),fontsize=label_font-1)
    ax.set_ylabel(sh(d["fy"]) if short else nm(d["fy"]),fontsize=label_font-1)
    ax.tick_params(labelsize=(TFS-3 if for_grid else TFS),direction="out",length=4,width=0.8)
    for spn in ax.spines.values(): spn.set_linewidth(0.8)
    return cf,Pmax,Pmin,mx,mn,handles

# ── single panels (outside legend) ──────────────────────────────────────────
for _,pr in pairs.iterrows():
    p=int(pr.pair_rank); d=D[p]
    fig,ax=plt.subplots(figsize=(9.2,6))
    cf,Pmax,Pmin,mx,mn,handles=draw_core(ax,p)
    ax.legend(handles=handles,loc="upper left",bbox_to_anchor=(1.30,1.0),fontsize=LFS-3,
              framealpha=0.95,edgecolor="0.7",facecolor="white",borderpad=0.7,handlelength=1.9,labelspacing=0.5).set_zorder(30)
    fig.subplots_adjust(left=0.10,right=0.68,top=0.94,bottom=0.11)
    cb=fig.colorbar(cf,ax=ax,pad=0.02,fraction=0.046,ticks=CBAR_TICKS)
    cb.set_label("Fold-ensemble P(OR+)",fontsize=LFS); cb.ax.tick_params(labelsize=TFS-2); cb.outline.set_linewidth(0.8)
    stem=f"{SUBF}/SI_FigS13_{p:02d}_{clean(d['fx'])}_vs_{clean(d['fy'])}"
    for e in ["png","pdf","svg"]: fig.savefig(f"{stem}.{e}",dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    # extrema audit
    XX,YY,S,sup=d["XX"],d["YY"],d["S"],d["sup"]
    Hc=np.load(f"{ROOT}/pair{p:02d}_surface_values.npz",allow_pickle=True)["support_counts"]
    sd=np.load(f"{ROOT}/pair{p:02d}_surface_values.npz",allow_pickle=True)["sd_surface"]
    for kind,idx,P in [("highest_supported",mx,Pmax),("lowest_supported",mn,Pmin)]:
        extrema_rows.append(dict(pair_id=p,extremum_type=kind,feature_x=d["fx"],feature_y=d["fy"],
            probability=round(P,4),support_count=int(Hc[idx]),fold_SD=round(float(sd[idx]),4),
            supported_cell=bool(sup[idx]),marker_x=round(float(XX[idx]),4),marker_y=round(float(YY[idx]),4),
            manual_coordinate_change=False,old_template_value_used=False))
    print(f"  pair{p} single-panel: Max {Pmax:.2f} Min {Pmin:.2f}")

pd.DataFrame(extrema_rows).to_csv(f"{STAT}/SI_Fig13_all_extrema_audit.csv",index=False)
assert all(r["supported_cell"] for r in extrema_rows), "extremum outside supported cell!"

# ── 2x3 combined (shared legend + shared colorbar) ──────────────────────────
letters=["(a)","(b)","(c)","(d)","(e)","(f)"]
fig,axes=plt.subplots(2,3,figsize=(17,10)); axes=axes.flatten(); cf=None
for k,p in enumerate(sorted(D)):
    cf,Pmax,Pmin,mx,mn,_=draw_core(axes[k],p,label_font=12,short=True,for_grid=True)
    axes[k].set_title(f"{letters[k]} {sh(D[p]['fx'])} × {sh(D[p]['fy'])}",fontsize=11,fontweight="bold")
sharedh=[mlines.Line2D([],[],color=Q1C,ls="--",lw=1.6,label="supported Q1"),
         mlines.Line2D([],[],color=MEDC,ls="-",lw=2.2,label="supported Median"),
         mlines.Line2D([],[],color=Q3C,ls="--",lw=1.6,label="supported Q3"),
         mlines.Line2D([],[],color="none",marker="*",markerfacecolor="#FF8F00",markeredgecolor="black",markersize=13,label="Max (supported)"),
         mlines.Line2D([],[],color="none",marker="o",markerfacecolor="#00BCD4",markeredgecolor="black",markersize=9,label="Min (supported)"),
         mlines.Line2D([],[],color=SUPP,lw=0.6,alpha=0.5,label="empirical-support boundary")]
fig.legend(handles=sharedh,loc="upper center",ncol=6,fontsize=10.5,frameon=True,facecolor=(1,1,1,0.9),edgecolor="#cccccc",bbox_to_anchor=(0.5,1.0))
fig.subplots_adjust(left=0.05,right=0.90,top=0.90,bottom=0.07,wspace=0.30,hspace=0.30)
cax=fig.add_axes([0.92,0.12,0.017,0.72]); cb=fig.colorbar(cf,cax=cax,ticks=CBAR_TICKS)
cb.set_label("Fold-ensemble P(OR+)",fontsize=12)
for e in ["png","pdf","svg"]: fig.savefig(f"{SUBF}/SI_FigS13_2x3_empbg_foldensemble.{e}",dpi=300,bbox_inches="tight",facecolor="white")
plt.close(fig)
print("2x3 combined regenerated. extrema audit rows:",len(extrema_rows))
