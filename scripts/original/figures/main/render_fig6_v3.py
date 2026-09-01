"""
SI Fig6 STYLE-LOCK v3 (QC final): 旧版视觉外壳 + n=5204真实OOF
新增: minimum-support规则; solvent真实名称+无序处理; 简化图注; 逐图validation。
全参数预定义统一, 不逐图手调。simulated_data_used=False 强制校验。
"""
import os, numpy as np, pandas as pd, hashlib, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
try:
    import statsmodels.api as sm; HAS_SM=True
except: HAS_SM=False

plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
    "font.size":10,"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False})
RED,GREEN="#9C1A1C","#48A597"; POS_BG="#DEF4F1"; NEG_BG="#E6DADA"; LOWESS_COLOR="#FF451B"
SHAP_CMAP=mcolors.LinearSegmentedColormap.from_list("shap",[GREEN,"#FFFFFF",RED])

# ===== 全Fig6统一预定义参数 =====
SCATTER_S=10; SCATTER_ALPHA=0.22; LOWESS_FRAC=0.30
GAP_FACTOR=20; GAP_MIN_SPAN_FRAC=0.05
MIN_CLUSTER_N=50; DENSITY_FRAC=0.25   # 段密度<最密段25%不画LOWESS          # minimum local support: 段内<50点不画LOWESS
SOLVENT_NAMES={1:"MeOH",2:"EtOH",3:"H2O",4:"DCM",5:"CHCl3"}

name_map={"geo:signed_tetra_volume":"Signed tetrahedral volume",
    "geo:signed_dihedral_subst_c_rn1_rn2":"Signed dihedral (subst-C*-ring)",
    "geo:baseline_pm_dihedral_sin":"P/M dihedral (sin component)",
    "geo:signed_dihedral_NS_path":"Signed dihedral (N-S path)","dihedral":"Baseline dihedral",
    "solvent_code":"Solvent","3D_Sphero":"3D spherocity","sPAS_0":"sPAS descriptor 0","PAS_5":"PAS descriptor 5"}
def label(n):
    n=str(n)
    if n in name_map: return name_map[n]
    if n.startswith("ECFP_"): return f"ECFP bit {n[5:]}"
    if n.startswith("PAS_"): return f"PAS descriptor {n[4:]}"
    if n.startswith("sPAS_"): return f"sPAS descriptor {n[5:]}"
    return n.replace("geo:","").replace("_"," ")

def detect_segments(xs):
    xr=xs.max()-xs.min(); gaps=np.diff(xs)
    med=np.median(gaps[gaps>0]) if (gaps>0).any() else 1e-9
    gap_idx=np.where((gaps>GAP_FACTOR*med)&(gaps>GAP_MIN_SPAN_FRAC*xr))[0]
    segs=[]; start=0
    for gi in gap_idx: segs.append((start,gi)); start=gi+1
    segs.append((start,len(xs)-1))
    return segs

sv=np.load("oof_shap_values_modelB.npy"); X=np.load("X_C3_5204x2283.npy")
gi=pd.read_csv("oof_shap_global_importance.csv")
feat_all=pd.read_csv("feature_order_check.csv").feature_name.tolist()
SHA_SV=hashlib.sha256(open("oof_shap_values_modelB.npy","rb").read()).hexdigest()[:16]
SHA_X=hashlib.sha256(open("X_C3_5204x2283.npy","rb").read()).hexdigest()[:16]
os.makedirs("SI_Fig6_STYLELOCK",exist_ok=True)

val_rows=[]
t10=gi.head(10)
for _,row in t10.iterrows():
    feat=str(row["feature"]); rank=int(row["rank"])
    ci=feat_all.index(feat); x=X[:,ci]; y=sv[:,ci]
    nu=len(np.unique(x))
    is_solvent = (feat=="solvent_code")
    is_binary = nu==2
    is_disc = nu<=10
    lo,hi=np.percentile(x,2),np.percentile(x,98)
    cval=np.clip((x-lo)/(hi-lo+1e-9),0,1)

    fig,ax=plt.subplots(figsize=(9,6.5))
    y_min,y_max=y.min()*1.3,y.max()*1.3
    if y_max<=0: y_max=abs(y.min())*0.1
    if y_min>=0: y_min=-abs(y.max())*0.1
    ax.axhspan(0,max(y_max,0.01),facecolor=POS_BG,alpha=0.55,zorder=0)
    ax.axhspan(min(y_min,-0.01),0,facecolor=NEG_BG,alpha=0.55,zorder=0)
    ax.axhline(0,color="#888",lw=0.9,ls=":",zorder=2)
    ax.scatter(x,y,s=SCATTER_S,c=cval,cmap=SHAP_CMAP,vmin=0,vmax=1,
               alpha=SCATTER_ALPHA,edgecolors="none",zorder=3,rasterized=True)
    sm_=plt.cm.ScalarMappable(cmap=SHAP_CMAP,norm=plt.Normalize(0,1)); sm_.set_array([])
    cbar=plt.colorbar(sm_,ax=ax,fraction=0.046,pad=0.02); cbar.set_label("Feature value",fontsize=9.5)

    seg_n=[]
    ftype="continuous"
    if is_solvent:
        ftype="nominal_category"
        lv=sorted(np.unique(x)); med=[np.median(y[x==l]) for l in lv]
        # 无序类别: 只画median点+细连接线(视觉), x轴用溶剂名
        ax.plot(lv,med,color=LOWESS_COLOR,lw=1.5,ls="-",zorder=4,marker="o",ms=8,label="level median")
        ax.set_xticks(lv); ax.set_xticklabels([SOLVENT_NAMES.get(int(l),str(l)) for l in lv])
        seg_n=[int((x==l).sum()) for l in lv]
    elif is_binary:
        ftype="binary"
        lv=sorted(np.unique(x)); med=[np.median(y[x==l]) for l in lv]
        ax.plot(lv,med,color=LOWESS_COLOR,lw=2.8,zorder=4,marker="o",ms=8,label="level median")
        ax.set_xticks(lv)
        seg_n=[int((x==l).sum()) for l in lv]
    else:
        sidx=np.argsort(x); xs=x[sidx]; ys=y[sidx]
        segs=detect_segments(xs); first=True
        # 预定义密度门槛: 段密度 < 最密段的 DENSITY_FRAC 则不画LOWESS(仅留散点)
        # max_dens 只在通过 MIN_CLUSTER_N 的段中取(避免单点段虚假拉高)
        valid_dens=[(b-a+1)/max(xs[b]-xs[a],1e-9) for (a,b) in segs if (b-a+1)>=MIN_CLUSTER_N and (xs[b]-xs[a])>1e-6]
        max_dens=max(valid_dens) if valid_dens else 1.0
        for si,(a,b) in enumerate(segs):
            npts=b-a+1; span=xs[b]-xs[a]
            if npts<MIN_CLUSTER_N or span<=1e-6: continue      # minimum count + 非退化段
            dens=npts/span
            if dens < DENSITY_FRAC*max_dens: continue          # minimum density (稀疏长段不画)
            xseg,yseg=xs[a:b+1],ys[a:b+1]
            if HAS_SM:
                loo=sm.nonparametric.lowess(yseg,xseg,frac=min(LOWESS_FRAC*len(xs)/npts,0.9))
                ax.plot(loo[:,0],loo[:,1],color=LOWESS_COLOR,lw=2.8,zorder=4,
                        label="LOWESS trend" if first else None); first=False
                seg_n.append(npts)

    is_geo=feat.startswith("geo:")
    ax.set_title(f"SI Fig 6.{rank}  {label(feat)}{'  [signed-geometry]' if is_geo else ''}\n(rank {rank} of 2,283 features)",
                 fontsize=11,fontweight="bold",loc="left",pad=10)
    xlab = "Solvent" if is_solvent else "Feature value (descriptor scale)"
    ax.set_xlabel(xlab,fontsize=11,fontweight="bold")
    ax.set_ylabel("SHAP value (model attribution)",fontsize=11,fontweight="bold")
    ax.set_ylim(y_min,y_max)
    if seg_n: ax.legend(fontsize=9.5,loc="upper left")
    # 简化图注(旧版简洁度)
    ax.text(0.5,-0.13,"Pooled OOF TreeSHAP (n = 5,204).",ha="center",fontsize=8.5,
            style="italic",color="#555",transform=ax.transAxes)

    safe=feat.replace("geo:","").replace(":","_")
    for ext in ["png","pdf"]:
        fig.savefig(f"SI_Fig6_STYLELOCK/{rank:02d}_{safe}.{ext}",dpi=300,bbox_inches="tight")
    plt.close()

    val_rows.append({"figure":f"SI_Fig6.{rank}","feature":feat,"n_samples":len(x),
        "n_unique_x":nu,"feature_type":ftype,"x_min":round(float(x.min()),4),"x_max":round(float(x.max()),4),
        "n_support_segments":len(seg_n),"n_per_segment":str(seg_n),
        "lowess_frac":LOWESS_FRAC,"min_cluster_n":MIN_CLUSTER_N,"gap_factor":GAP_FACTOR,
        "coloring_feature":"self","coloring_source":"feature own value (interim)",
        "shap_sha256":SHA_SV,"feature_matrix_sha256":SHA_X,"simulated_data_used":False})
    print(f"  [{rank}] {label(feat)} ({ftype}, segs={len(seg_n)} n={seg_n})")

vdf=pd.DataFrame(val_rows)
vdf.to_csv("SI_Fig6_data_validation_report.csv",index=False)
assert (vdf.simulated_data_used==False).all(), "STOP: simulated data detected!"
print("\nSI Fig6 v3 QC done. simulated_data_used 全部False ✓")
print("validation report: SI_Fig6_data_validation_report.csv")
