"""SI Fig5 STYLE-LOCK: 1:1 复现旧版 render_si_fig5_global_shap_v2.py 视觉,只换真实OOF数据"""
import numpy as np, pandas as pd, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ===== 旧版静态样式参数 (从 render_si_fig5_global_shap_v2.py 提取) =====
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 14, "svg.fonttype": "none",
})
shap_cmap = mcolors.LinearSegmentedColormap.from_list("shap", ["#48A597","#FFFFFF","#9C1A1C"])
BAR_COL = "#C3D3F2"   # 旧版统一浅蓝,不分类高亮

# ===== 真实数据 (数值source-of-truth) =====
gi = pd.read_csv("oof_shap_global_importance.csv")   # 真实OOF排序
sv = np.load("oof_shap_values_modelB.npy")           # (5204,2283) 真实OOF
X  = np.load("X_C3_5204x2283.npy")
feat_all = pd.read_csv("feature_order_check.csv").feature_name.tolist()

# 人类可读特征名 (旧版写法)
NMAP={"geo:signed_tetra_volume":"signed tetra volume","ECFP_1412":"ECFP bit 1412",
 "geo:signed_dihedral_subst_c_rn1_rn2":"signed dihedral (subst-ring)","geo:baseline_pm_dihedral_sin":"P/M dihedral (sin)",
 "dihedral":"baseline dihedral","geo:signed_dihedral_NS_path":"signed dihedral (NS)","sPAS_0":"sPAS-0",
 "3D_Sphero":"3D spherocity","PAS_5":"PAS-5","solvent_code":"solvent code","PAS_10":"PAS-10","sPAS_10":"sPAS-10",
 "PAS_15":"PAS-15","MACCS_129":"MACCS key 129","ECFP_492":"ECFP bit 492","geo:baseline_pm_dihedral_cos":"P/M dihedral (cos)",
 "geo:subst_to_ringplane_signed_dist":"subst-to-ringplane","geo:ringnormal_dot_substvec":"ringnormal.substvec",
 "TPSA":"TPSA","3D_ISF":"3D ISF","3D_NPR1":"3D NPR1","MACCS_62":"MACCS key 62"}
def lab(f): return NMAP.get(f, f.replace("geo:","").replace("_"," "))

# top20 (真实OOF排序), 升序排列使最重要在顶部(旧版 iloc[::-1])
top20 = gi.head(20).iloc[::-1].reset_index(drop=True)
feat_names = [lab(f) for f in top20.feature]
mean_shap  = top20.mean_abs_shap.values
total_shap = gi.mean_abs_shap.sum()   # 全2283特征之和
y_pos = np.arange(len(top20))
top20_col_idx = [feat_all.index(f) for f in top20.feature]

# 真实 per-sample OOF SHAP + 特征值(用于beeswarm着色), 逐特征percentile归一化
N = sv.shape[0]  # 5204
np.random.seed(42)

# ===== 旧版布局: 单图 figsize=(10,8), twiny =====
fig, ax1 = plt.subplots(figsize=(10, 8))
ax2 = ax1.twiny()

# ax2(top x): barh 单色浅蓝 (旧版参数)
ax2.barh(y_pos, mean_shap, color=BAR_COL, align="center", alpha=0.80, height=0.60, zorder=2)

# ax1(bottom x): 真实 beeswarm (旧版 s=12 alpha=0.80)
for i, ci in enumerate(top20_col_idx):
    row_shap = sv[:, ci]                       # 真实OOF SHAP
    xv = X[:, ci]
    lo,hi = np.percentile(xv,5), np.percentile(xv,95)
    row_feat = np.clip((xv-lo)/(hi-lo+1e-9),0,1)  # 特征值归一化着色
    jitter = np.random.normal(0, 0.10, size=N)     # 仅防重叠,不改横坐标
    ax1.scatter(row_shap, np.repeat(i,N)+jitter, c=row_feat, cmap=shap_cmap,
                s=12, alpha=0.80, edgecolors="none", zorder=4)

# 百分比标注(真实值, 旧版样式: ax2 transData, 白底)
max_mean = mean_shap.max()
ax2.set_xlim(0, max_mean*1.22)
for i, v in enumerate(mean_shap):
    pct = v/total_shap*100
    ax1.text(v + max_mean*0.012, i, f"{pct:.1f}%", va="center", ha="left", fontsize=12,
             transform=ax2.transData, zorder=10,
             bbox=dict(facecolor="white", alpha=0.80, edgecolor="none", pad=1))

ax1.set_zorder(ax2.get_zorder()+1); ax1.patch.set_visible(False)
ax1.set_yticks(y_pos); ax1.set_yticklabels(feat_names, fontsize=13)
ax1.set_xlabel("SHAP value (impact on model output)", fontsize=14)
ax2.set_xlabel("Mean absolute SHAP value", fontsize=14)
ax1.axvline(0, color="#888", lw=0.8, ls="--", alpha=0.5, zorder=1)
ax1.grid(True, axis="x", linestyle="--", alpha=0.35, zorder=0)

# colorbar(旧版: 右侧 3%, High顶Low底, Feature value竖排)
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="3%", pad=0.12)
sm = plt.cm.ScalarMappable(cmap=shap_cmap, norm=plt.Normalize(0,1)); sm.set_array([])
cbar = fig.colorbar(sm, cax=cax); cbar.set_ticks([0,1]); cbar.set_ticklabels(["Low","High"])
cbar.set_label("Feature value", rotation=270, labelpad=18, fontsize=13)

fig.tight_layout()
for ext in ["svg","pdf","png"]:
    fig.savefig(f"SI_Fig5_OOF_STYLELOCK_V2.{ext}", dpi=300, bbox_inches="tight")
# tiff
fig.savefig("SI_Fig5_OOF_STYLELOCK_V2.tiff", dpi=300, bbox_inches="tight", pil_kwargs={"compression":"tiff_lzw"})
plt.close()

# plot_data.csv
pd.DataFrame({"rank_top20":range(1,21),"feature":top20.feature.values[::-1],
    "label":feat_names[::-1],"mean_abs_shap":mean_shap[::-1],
    "percentage":(mean_shap[::-1]/total_shap*100)}).to_csv("SI_Fig5_plot_data.csv",index=False)
print("SI_Fig5_OOF_STYLELOCK_V2 saved (svg/pdf/png/tiff) + plot_data")
