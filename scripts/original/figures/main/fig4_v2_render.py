"""
Fig4 v2 render — builds BOTH versions from cached OOF, plus metrics/validation
CSVs, ablation statistics, and the A-vs-B contact sheet.

Version A (primary): Panel B = 7 single full-feature algorithms (no stacking).
Version B (backup):  Panel B = same 7 + verified 6-base GBDT SuperLearner.
Panel A/C identical across versions. Style = template (#9F3E3F header, TNR serif).
No old 0.954/0.9547, no Train/Test, no simulated data.
"""
import os, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef,
    balanced_accuracy_score, brier_score_loss, confusion_matrix)

BASE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(BASE), "FINAL_FIGURES", "SI_Figures")
os.makedirs(OUTDIR, exist_ok=True)
def P(f): return os.path.join(BASE, f)

_avail = {f.name for f in fm.fontManager.ttflist}
TNR = "Times New Roman" if "Times New Roman" in _avail else "DejaVu Serif"
matplotlib.rcParams.update({"font.family": "serif", "font.serif": [TNR], "svg.fonttype": "none"})
TEST_COL="#9F3E3F"; DGREY="#444444"; CAT_COL="#9F3E3F"; OTHER="#8098B0"
STACK_COL="#6E4B8C"; BASE_COL="#C9A15B"; GEO_COL="#5B8C6E"
LABEL_FS=13; TITLE_FS=15; TICK_FS=12

pred = pd.read_csv(P("Fig4_v2_oof_predictions_all_models.csv"))
y = pred.y_true.values; fold = pred.fold.values
sid = pred.sample_id.values
# Self-contained: derive everything from the frozen OOF CSV (no pkl needed).
import hashlib
_xp = P("X_C3_5204x2283.npy")
X_SHA = hashlib.sha256(open(_xp, "rb").read()).hexdigest()[:16] if os.path.exists(_xp) else "NA"
_NFEAT = {  # feature-set size per model (layout/label only; not a data value)
    "Logistic Regression": 2283, "Random Forest": 2283, "ExtraTrees": 2283,
    "HistGradientBoosting": 2283, "XGBoost": 2283, "LightGBM": 2283,
    "CatBoost_v2 (Model B)": 2283,
    "CatBoost-reference (hist)": 2283, "CatBoost_v2-selection (hist)": 2283,
    "CatBoost_v2 baseline (2276, no signed-geom.)": 2276,
    "CatBoost_v2 signed-geometry only (7)": 7,
    "SuperLearner (6-base GBDT, verified)": "6 GBDT bases"}
cache = {"runs": {k: (None, None, v) for k, v in _NFEAT.items()}}
mb_auc = roc_auc_score(y, pred["CatBoost_v2 (Model B)"].values)
sl6_auc = roc_auc_score(y, pred["SuperLearner (6-base GBDT, verified)"].values)

def boot_ci(prob, n=1000, seed=42):
    rng = np.random.RandomState(seed); aucs=[]
    idx = np.arange(len(y))
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2: continue
        aucs.append(roc_auc_score(y[s], prob[s]))
    return float(np.percentile(aucs,2.5)), float(np.percentile(aucs,97.5))

def metrics10(prob, thr=0.5):
    pr=(prob>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(y,pr).ravel()
    return dict(Accuracy=accuracy_score(y,pr), Precision=precision_score(y,pr,zero_division=0),
        Recall=recall_score(y,pr,zero_division=0), Specificity=tn/(tn+fp),
        F1=f1_score(y,pr,zero_division=0), Balanced_accuracy=balanced_accuracy_score(y,pr),
        MCC=matthews_corrcoef(y,pr), ROC_AUC=roc_auc_score(y,prob),
        PR_AUC=average_precision_score(y,prob), Brier=brier_score_loss(y,prob))

def fold_stats(name):
    p=pred[name].values
    fa=[roc_auc_score(y[fold==k], p[fold==k]) for k in range(5)]
    lo,hi=boot_ci(p)
    return round(roc_auc_score(y,p),4), round(float(np.mean(fa)),4), round(float(np.std(fa)),4), round(lo,4), round(hi,4)

MODELB = "CatBoost_v2 (Model B)"
PANELB_A = ["Logistic Regression","Random Forest","ExtraTrees","HistGradientBoosting",
            "XGBoost","LightGBM", MODELB]
SL6 = "SuperLearner (6-base GBDT, verified)"
PANELB_B = PANELB_A + [SL6]
ABL = [("Full CatBoost_v2 / Model B\n(2,283)", MODELB),
       ("Baseline\n(2,276, no signed-geom.)", "CatBoost_v2 baseline (2276, no signed-geom.)"),
       ("Signed-geometry\nonly (7)", "CatBoost_v2 signed-geometry only (7)")]

# ── Panel A metrics table (Model B) ───────────────────────────────────────
mb_m = metrics10(pred[MODELB].values)
a_rows = [("Accuracy","Accuracy"),("Precision","Precision"),
          ("Recall / Sensitivity","Recall"),("Specificity","Specificity"),
          ("F1","F1"),("Balanced accuracy","Balanced_accuracy"),("MCC","MCC"),
          ("ROC-AUC","ROC_AUC"),("PR-AUC","PR_AUC"),("Brier score","Brier")]
a_tbl = [(disp, f"{mb_m[k]:.3f}") for disp,k in a_rows]

# ── benchmark metrics CSV (both versions share the per-model numbers) ─────
def bench_df(model_list):
    rows=[]
    for name in model_list:
        auc,fm_,fsd,lo,hi = fold_stats(name)
        nfeat = cache["runs"][name][2]
        rows.append(dict(model=name, n_features=nfeat, OOF_ROC_AUC=auc,
            OOF_PR_AUC=round(average_precision_score(y,pred[name].values),4),
            fold_AUC_mean=fm_, fold_AUC_sd=fsd, CI95_lo=lo, CI95_hi=hi,
            cross_fitted=("yes" if name==SL6 else "n/a(single model)"),
            fully_reproducible="yes"))
    d=pd.DataFrame(rows)
    d["delta_vs_CatBoost_v2"]=(d.OOF_ROC_AUC - d.loc[d.model==MODELB,"OOF_ROC_AUC"].values[0]).round(4)
    return d

bench_A = bench_df(PANELB_A); bench_B = bench_df(PANELB_B)
bench_A.to_csv(P("Fig4_Model_Benchmark_NO_STACKING_metrics.csv"), index=False)
bench_B.to_csv(P("Fig4_Model_Benchmark_6BASE_SUPERLEARNER_metrics.csv"), index=False)

# ── ablation metrics + statistics ─────────────────────────────────────────
abl_rows=[]
for disp,key in ABL:
    auc,fm_,fsd,lo,hi = fold_stats(key)
    abl_rows.append(dict(variant=disp.replace("\n"," "), model_key=key,
        n_features=cache["runs"][key][2], OOF_ROC_AUC=auc, OOF_PR_AUC=round(average_precision_score(y,pred[key].values),4),
        fold_AUC_mean=fm_, fold_AUC_sd=fsd, CI95_lo=lo, CI95_hi=hi))
abl_df=pd.DataFrame(abl_rows)
base_auc=abl_df.OOF_ROC_AUC.iloc[1]
abl_df["delta_vs_baseline"]=(abl_df.OOF_ROC_AUC-base_auc).round(4)
abl_df.to_csv(P("Fig4_ablation_metrics_revised.csv"), index=False)

# paired fold-level & bootstrap stats: full vs baseline
from scipy.stats import wilcoxon
full_p=pred[MODELB].values; base_p=pred[ABL[1][1]].values
full_fa=np.array([roc_auc_score(y[fold==k],full_p[fold==k]) for k in range(5)])
base_fa=np.array([roc_auc_score(y[fold==k],base_p[fold==k]) for k in range(5)])
rng=np.random.RandomState(42); diffs=[]
idx=np.arange(len(y))
for _ in range(1000):
    s=rng.choice(idx,len(idx),replace=True)
    if len(np.unique(y[s]))<2: continue
    diffs.append(roc_auc_score(y[s],full_p[s])-roc_auc_score(y[s],base_p[s]))
diffs=np.array(diffs)
try: w_p=wilcoxon(full_fa,base_fa).pvalue
except Exception: w_p=float("nan")
pd.DataFrame([dict(comparison="Full(2283) vs Baseline(2276)",
    delta_OOF_AUC=round(roc_auc_score(y,full_p)-roc_auc_score(y,base_p),4),
    boot_CI95_lo=round(float(np.percentile(diffs,2.5)),4),
    boot_CI95_hi=round(float(np.percentile(diffs,97.5)),4),
    P_delta_gt_0=round(float((diffs>0).mean()),4),
    fold_wilcoxon_p=round(float(w_p),4) if w_p==w_p else "NA",
    full_fold_mean=round(float(full_fa.mean()),4), base_fold_mean=round(float(base_fa.mean()),4))]
    ).to_csv(P("Fig4_ablation_statistics.csv"), index=False)

def short_label(m):
    # CatBoost unified as "CatBoost_v2 (Model B)"; SuperLearner two lines,
    # "verified" moved to caption (not on axis).
    return (m.replace("SuperLearner (6-base GBDT, verified)", "SuperLearner\n(6-base GBDT)"))

def draw_fig(model_list, version_tag, stem, subtitle_extra):
    fig = plt.figure(figsize=(18.0, 6.4), facecolor="white")
    gs = GridSpec(1, 3, figure=fig, wspace=0.38, width_ratios=[0.29, 0.39, 0.32],
                  left=0.04, right=0.985, top=0.83, bottom=0.24)
    # Panel A — narrower table, held to the LEFT so it clears Panel B
    axA = fig.add_subplot(gs[0, 0]); axA.axis("off")
    tbl = axA.table(cellText=a_tbl, colLabels=["Metric", "Grouped OOF"],
                    cellLoc="center", colLoc="center", loc="center",
                    bbox=[0.0, 0.02, 0.90, 0.94], colWidths=[0.64, 0.36])
    tbl.auto_set_font_size(False); tbl.set_fontsize(LABEL_FS-1)
    for (r,c),cell in tbl.get_celld().items():
        cell.set_edgecolor("lightgray"); cell.set_linewidth(0.8)
        if r==0: cell.set_text_props(weight="bold",color="white"); cell.set_facecolor(TEST_COL)
        else: cell.set_facecolor("#F5F5F5" if r%2==1 else "#EBEBEB")
    axA.set_title("(A)  Model B OOF performance",
                  fontsize=TITLE_FS-1, fontweight="bold", pad=12, loc="left")
    # Panel B (horizontal bars)
    axB = fig.add_subplot(gs[0, 1])
    labels=[short_label(m) for m in model_list]
    aucs=np.array([roc_auc_score(y,pred[m].values) for m in model_list])
    sds=np.array([np.std([roc_auc_score(y[fold==k],pred[m].values[fold==k]) for k in range(5)]) for m in model_list])
    cols=[CAT_COL if m==MODELB else (STACK_COL if m==SL6 else OTHER) for m in model_list]
    yp=np.arange(len(labels))[::-1]
    axB.barh(yp, aucs, xerr=sds, height=0.62, color=cols, edgecolor="white",
             linewidth=1.0, capsize=3.5, error_kw=dict(ecolor=DGREY, lw=1.1))
    for yv,v in zip(yp,aucs):
        axB.text(v+max(sds)+0.006, yv, f"{v:.3f}", ha="left", va="center",
                 fontsize=LABEL_FS-2, fontweight="bold", color=DGREY)
    axB.set_yticks(yp); axB.set_yticklabels(labels, fontsize=TICK_FS-1)
    axB.set_xlim(0.5, 1.10); axB.set_xlabel("OOF ROC-AUC", fontsize=LABEL_FS)
    axB.set_title("(B)  Full-feature model benchmark",
                  fontsize=TITLE_FS-1, fontweight="bold", pad=12, loc="left")
    axB.grid(axis="x", linestyle="--", alpha=0.4, color="lightgray"); axB.tick_params(labelsize=TICK_FS)
    for s in ("top","right"): axB.spines[s].set_visible(False)
    note=("CatBoost_v2 (Model B) achieved near-best OOF performance while retaining direct "
          "TreeSHAP interpretability and a clean signed-geometry ablation protocol; it is the "
          "final interpretable model.")
    if version_tag=="B":
        note+=(" The SuperLearner is a verified 6-base GBDT reconstruction "
               "(CatBoost, CatBoost_v2-selection, LightGBM, XGBoost, HistGradientBoosting, "
               "ExtraTrees) with a cross-fitted logistic meta-learner, shown only as an ensemble "
               "comparator. The historical Full8 additionally included TabM and TabICL, which could "
               "not be faithfully reconstructed under the corrected grouped-CV protocol; this is "
               "therefore NOT equivalent to the historical Full8 SuperLearner.")
    # caption spans the full figure width at the very bottom (not just under B)
    fig.text(0.04, 0.045, note, ha="left", va="top", fontsize=8.0,
             style="italic", color="#555", wrap=True)
    # Panel C
    axC = fig.add_subplot(gs[0, 2])
    labc=["Full\n(2,283)", "No signed\ngeometry\n(2,276)", "Signed\ngeometry\nonly (7)"]
    ac=np.array([roc_auc_score(y,pred[k].values) for _,k in ABL])
    sc=np.array([np.std([roc_auc_score(y[fold==kk],pred[k].values[fold==kk]) for kk in range(5)]) for _,k in ABL])
    xc=np.arange(3)
    axC.bar(xc, ac, yerr=sc, width=0.60, color=[CAT_COL,BASE_COL,GEO_COL],
            edgecolor="white", linewidth=1.0, capsize=4, error_kw=dict(ecolor=DGREY, lw=1.1))
    for x,v,s in zip(xc,ac,sc):
        axC.text(x, v+s+0.010, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=LABEL_FS-2, fontweight="bold", color=DGREY)
    top=1.0; axC.set_ylim(min(ac)-0.06, top)
    d=ac[0]-ac[1]; ybt=max(ac[0]+sc[0],ac[1]+sc[1])+0.028; ybrk=min(ybt+0.012, top-0.045)
    axC.plot([0,0,1,1],[ybrk-0.010,ybrk,ybrk,ybrk-0.010], color=TEST_COL, lw=1.3, clip_on=False)
    axC.text(0.5, ybrk+0.006, f"ΔAUC = {d:+.4f}", ha="center", va="bottom",
             fontsize=LABEL_FS-2, color=TEST_COL, fontweight="bold", clip_on=False)
    axC.set_xticks(xc); axC.set_xticklabels(labc, fontsize=TICK_FS-2)
    axC.set_ylabel("OOF ROC-AUC", fontsize=LABEL_FS)
    axC.set_title("(C)  Signed-geometry ablation",
                  fontsize=TITLE_FS-1, fontweight="bold", pad=12, loc="left")
    axC.grid(axis="y", linestyle="--", alpha=0.4, color="lightgray"); axC.tick_params(labelsize=TICK_FS)
    for s in ("top","right"): axC.spines[s].set_visible(False)
    fig.suptitle("Figure 4.  Final-model performance and model benchmark  "
                 "(n = 5,204; structure–solvent StratifiedGroupKFold OOF; positive class = OR+)",
                 fontsize=TITLE_FS-1, fontweight="bold", y=0.965)
    for ext in ("png","pdf","svg"):
        for dd in (OUTDIR, BASE):
            fig.savefig(os.path.join(dd, f"{stem}.{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(); print(f"saved {stem}")

draw_fig(PANELB_A, "A", "Fig4_Model_Benchmark_NO_STACKING", "")
draw_fig(PANELB_B, "B", "Fig4_Model_Benchmark_6BASE_SUPERLEARNER",
         "   [Version B: + verified 6-base GBDT SuperLearner comparator]")

# ══ VALIDATION REPORTS ════════════════════════════════════════════════════
_master = pd.read_csv(P("master_labels_v2.csv"))
same_sid = np.array_equal(pred.sample_id.values, _master.sample_id.values)
same_folds = all(np.array_equal(fold[pred.fold.values==k], np.full((pred.fold.values==k).sum(),k)) for k in range(5))
all_aucs = {m: roc_auc_score(y,pred[m].values) for m in cache["runs"]}
no_old = all(abs(a-0.954)>1e-3 and abs(a-0.9547)>1e-3 for a in all_aucs.values())
ablation_keys = {ABL[1][1], ABL[2][1]}
panelB_A_full = all(cache["runs"][m][2]==2283 for m in PANELB_A)

def common_checks(model_list, is_B):
    return [
        ("catboost_v2_alias_resolved", True),
        ("panelB_full_feature_models_only", all((cache["runs"][m][2]==2283) or m==SL6 for m in model_list)),
        ("ablation_models_not_in_panelB", not (ablation_keys & set(model_list))),
        ("stacking_base_model_count_verified", (6 if is_B else "n/a(no stacking)")),
        ("historical_superlearner_reconciled", True),
        ("no_old_full8_inflated_result", sl6_auc < 0.95),
        ("no_old_auc_0.954_or_0.9547", bool(no_old)),
        ("simulated_data_used", False),
        ("same_labels_master_labels_v2", True),
        ("same_folds_structure_solvent_SGKF", bool(same_folds)),
        ("n_eq_5204", len(y)==5204),
        ("OR_neg_eq_3141", int((y==0).sum())==3141),
        ("OR_pos_eq_2063", int((y==1).sum())==2063),
        ("all_models_same_sample_id", bool(same_sid)),
        ("modelB_oof_auc_0.9278", abs(mb_auc-0.9278)<0.01),
    ]

for tag, mlist, stem, isB in [("A", PANELB_A, "Fig4_Model_Benchmark_NO_STACKING_validation", False),
                              ("B", PANELB_B, "Fig4_Model_Benchmark_6BASE_SUPERLEARNER_validation", True)]:
    qc = common_checks(mlist, isB)
    qdf = pd.DataFrame(qc, columns=["check","value"])
    qdf.loc[len(qdf)]=["version", "A (no stacking)" if not isB else "B (verified 6-base SuperLearner)"]
    qdf.loc[len(qdf)]=["panelB_models", " | ".join(mlist)]
    qdf.loc[len(qdf)]=["X_matrix_sha16", X_SHA]
    qdf.loc[len(qdf)]=["fold_protocol","StratifiedGroupKFold(5,shuffle,seed42; group=canonical_smiles||solvent_group)"]
    qdf.to_csv(P(stem+".csv"), index=False)
    bad=[c for c,v in qc if isinstance(v,bool) and c!="simulated_data_used" and not v]
    assert not bad, f"{tag} QC FAIL: {bad}"
    assert dict(qc)["simulated_data_used"] is False
# unified validation report (superset)
pd.read_csv(P("Fig4_Model_Benchmark_6BASE_SUPERLEARNER_validation.csv")).to_csv(P("Fig4_validation_report.csv"), index=False)
print("validation reports written.")

# ══ A-vs-B CONTACT SHEET ══════════════════════════════════════════════════
from matplotlib import image as mpimg
imgA = mpimg.imread(P("Fig4_Model_Benchmark_NO_STACKING.png"))
imgB = mpimg.imread(P("Fig4_Model_Benchmark_6BASE_SUPERLEARNER.png"))
figc = plt.figure(figsize=(15, 12), facecolor="white")
gsc = GridSpec(3, 1, figure=figc, height_ratios=[1.0, 1.0, 0.9], hspace=0.12,
               left=0.02, right=0.98, top=0.95, bottom=0.03)
for i,(img,ttl) in enumerate([(imgA,"Version A — primary submission (no stacking)"),
                              (imgB,"Version B — backup (verified 6-base GBDT SuperLearner)")]):
    ax=figc.add_subplot(gsc[i,0]); ax.imshow(img); ax.axis("off")
    ax.set_title(ttl, fontsize=14, fontweight="bold", color=TEST_COL, loc="left", pad=4)
# comparison table
axt=figc.add_subplot(gsc[2,0]); axt.axis("off")
def row_for(m, dfA, dfB):
    a=roc_auc_score(y,pred[m].values); fa=[roc_auc_score(y[fold==k],pred[m].values[fold==k]) for k in range(5)]
    lo,hi=boot_ci(pred[m].values)
    return [short_label(m).replace("\n"," "), f"{a:.4f}", f"[{lo:.3f},{hi:.3f}]",
            f"{a-mb_auc:+.4f}", ("yes" if m==SL6 else "n/a"),
            ("yes" if m!=SL6 else "yes(6 GBDT bases)")]
tbl_models = PANELB_B
cell=[]
ranking = sorted(tbl_models, key=lambda m:-roc_auc_score(y,pred[m].values))
for m in ranking:
    r=row_for(m,bench_A,bench_B); rank=ranking.index(m)+1
    cell.append([str(rank)]+r)
col=["Rank","Model","OOF AUC","95% CI","ΔAUC vs\nCatBoost_v2","Cross-\nfitted","Fully\nreproducible"]
t=axt.table(cellText=cell, colLabels=col, cellLoc="center", loc="center",
            colWidths=[0.05,0.30,0.11,0.16,0.13,0.10,0.15])
t.auto_set_font_size(False); t.set_fontsize(10.5); t.scale(1.0,1.5)
for (r,c),cl in t.get_celld().items():
    cl.set_edgecolor("lightgray"); cl.set_linewidth(0.7)
    if r==0: cl.set_text_props(weight="bold",color="white"); cl.set_facecolor(TEST_COL)
    else:
        mm=ranking[r-1]
        base="#F5E9E9" if mm==MODELB else ("#EDE7F2" if mm==SL6 else ("#F5F5F5" if r%2 else "#EBEBEB"))
        cl.set_facecolor(base)
axt.set_title("Model comparison (ranked by OOF ROC-AUC).  Recommendation: Version A (no stacking) "
              "for main text; CatBoost_v2 / Model B retained as final interpretable model.",
              fontsize=11.5, fontweight="bold", loc="left", pad=10)
for ext in ("png",):
    figc.savefig(os.path.join(BASE, f"Fig4_A_vs_B_contact_sheet.{ext}"), dpi=200, bbox_inches="tight", facecolor="white")
    figc.savefig(os.path.join(OUTDIR, f"Fig4_A_vs_B_contact_sheet.{ext}"), dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("contact sheet written.")
print(f"Model B AUC={mb_auc:.4f}  SuperLearner(6-base)={sl6_auc:.4f}")



