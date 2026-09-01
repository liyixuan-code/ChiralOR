"""
SI Figs 1-4 — REBUILT on Model B (real leak-free OOF, no simulation).

Replaces the stale Aug-7 figures that used the DEPRECATED artifact-matched
parquet (oof_prob_C3, AUC 0.9537) and a FABRICATED np.random "Train" series.

Authoritative data source (frozen, Gate 1-5 PASS, retraining summary):
  output_revision/oof_predictions_master_labels_v2.csv
    - Model B = StratifiedGroupKFold(group = canonical_smiles + solvent)
    - n=5204, y_true (master_labels_v2): OR- 3141 / OR+ 2063
    - OOF ROC-AUC = 0.9278  (CI 0.9182-0.9367)
    - confusion TN/FP/FN/TP = 2826/315/414/1649
  master_labels_v2 SHA-256 = 8a9614165ee6544878416dda7edf657cbf12a10554b87c632f8f2b6b26976ad8

Visual style is byte-for-byte the SAME template as render_figs1_4_template_style.py:
  figsize Fig1/2/3=(7,6), Fig4=(7,4.8); colours test/OOF=#9F3E3F, class0=#9E9E9E;
  conf cmap white->#9F3E3F; Times New Roman serif; grid lightgray dashed;
  Fig3 bins=50 alpha=0.50 kde_lw=1.8; Fig4 header #9F3E3F, alt rows #F5F5F5/#EBEBEB.

Decisions (user-confirmed this session):
  - Fig1: 5 per-fold OOF ROC curves + overall OOF (NO fabricated train series).
  - Fig4: single real OOF metrics column (NO fabricated train column).
  - Fig2/Fig3 titles: "OOF (5-fold)"; Fig2 cells add row-percentage.

simulated_data_used = False.
"""
import os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
)

BASE = r"C:\Users\lenovo\.claude\projects\PM\JCIM_MANUSCRIPT"
CSV  = os.path.join(BASE, "output_revision", "oof_predictions_master_labels_v2.csv")
OUT  = os.path.join(BASE, "FINAL_FIGURES", "SI_Figures")
os.makedirs(OUT, exist_ok=True)

# ── font ────────────────────────────────────────────────────────────────
_avail = {f.name for f in fm.fontManager.ttflist}
TNR    = "Times New Roman" if "Times New Roman" in _avail else "DejaVu Serif"
matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": [TNR],
    "svg.fonttype": "none",
})
LABEL_FS = 14; TITLE_FS = 16; TICK_FS = 14; ANNOT_FS = 16

# ── template colours (exact from CONFIG) ─────────────────────────────────
TEST_COL   = "#9F3E3F"   # OOF curve / class1 fill / header
CLASS0_COL = "#9E9E9E"   # class0 fill
CONF_CMAP  = mcolors.LinearSegmentedColormap.from_list("conf", ["#FFFFFF", TEST_COL])

# ── load frozen Model B OOF predictions ──────────────────────────────────
df     = pd.read_csv(CSV)
y_true = df["y_true"].values
p_oof  = df["oof_prob"].values
folds  = sorted(df["fold"].unique())

# ── assertions against frozen Model B state ──────────────────────────────
auc_oof = roc_auc_score(y_true, p_oof)
assert len(df) == 5204, len(df)
assert int((y_true == 0).sum()) == 3141 and int((y_true == 1).sum()) == 2063
assert abs(auc_oof - 0.9278) < 1e-3, auc_oof
# NOTE: no np.random / synthetic anywhere below — every series is real Model B OOF.
print(f"Model B overall OOF AUC = {auc_oof:.4f}  (n={len(df)}, folds={folds})")

THRESHOLD = 0.50
y_pred    = (p_oof >= THRESHOLD).astype(int)


def get_metrics(y, prob, pred):
    return {
        "Accuracy":  accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall":    recall_score(y, pred, zero_division=0),
        "F1":        f1_score(y, pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y, prob),
    }

oof_m = get_metrics(y_true, p_oof, y_pred)

# ══════════════════════════════════════════════════════════════════════════
# FIG 1 — ROC Curve: 5 per-fold OOF + overall OOF  (no fabricated train)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))
fold_colors = [cm.tab10(i) for i in range(len(folds))]

for i, fold in enumerate(folds):
    mask = df["fold"].values == fold
    fpr, tpr, _ = roc_curve(y_true[mask], p_oof[mask])
    auc_f = roc_auc_score(y_true[mask], p_oof[mask])
    ax.plot(fpr, tpr, color=fold_colors[i], lw=1.6, alpha=0.85,
            label=f"Fold {fold} (AUC = {auc_f:.3f})")

fpr_all, tpr_all, _ = roc_curve(y_true, p_oof)
ax.plot(fpr_all, tpr_all, color=TEST_COL, lw=2.8,
        label=f"Overall OOF (AUC = {auc_oof:.3f})")
ax.plot([0, 1], [0, 1], color="black", linestyle="--", lw=1.2, label="Random")

ax.set_xlabel("False Positive Rate", fontsize=LABEL_FS)
ax.set_ylabel("True Positive Rate",  fontsize=LABEL_FS)
ax.set_title("ROC Curve (5-fold OOF)", fontsize=TITLE_FS, fontweight="bold")
ax.tick_params(labelsize=TICK_FS)
ax.grid(True, linestyle="--", alpha=0.50, color="lightgray")
ax.legend(loc="lower right", frameon=False, fontsize=LABEL_FS - 3)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"Fig1_ROC_Curve.{ext}"),
                dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Fig1 saved.")

# ══════════════════════════════════════════════════════════════════════════
# FIG 2 — OOF Confusion Matrix (counts + row %)
# ══════════════════════════════════════════════════════════════════════════
cm_mat = confusion_matrix(y_true, y_pred)
row_sums = cm_mat.sum(axis=1, keepdims=True)
annot = np.empty_like(cm_mat, dtype=object)
for r in range(cm_mat.shape[0]):
    for c in range(cm_mat.shape[1]):
        pct = 100.0 * cm_mat[r, c] / row_sums[r, 0]
        annot[r, c] = f"{cm_mat[r, c]:d}\n({pct:.1f}%)"

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm_mat, annot=annot, fmt="",
    cmap=CONF_CMAP, cbar=False, square=True,
    linewidths=0.8, linecolor="#FFFFFF", ax=ax,
    xticklabels=["0: OR-", "1: OR+"],
    yticklabels=["0: OR-", "1: OR+"],
    annot_kws={"fontsize": ANNOT_FS},
)
ax.set_xlabel("Predicted Class", fontsize=LABEL_FS)
ax.set_ylabel("True Class",      fontsize=LABEL_FS)
ax.set_title(f"OOF Confusion Matrix (5-fold, threshold={THRESHOLD:.2f})",
             fontsize=TITLE_FS, fontweight="bold")
ax.tick_params(axis="x", labelsize=TICK_FS - 2, rotation=0)
ax.tick_params(axis="y", labelsize=TICK_FS - 2, rotation=0)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"Fig2_Test_Confusion_Matrix.{ext}"),
                dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Fig2 saved.")

# ══════════════════════════════════════════════════════════════════════════
# FIG 3 — OOF Probability Distribution
# ══════════════════════════════════════════════════════════════════════════
prob_neg = p_oof[y_true == 0]
prob_pos = p_oof[y_true == 1]

fig, ax = plt.subplots(figsize=(7, 6))
sns.histplot(prob_neg, bins=50, stat="density", element="bars",
             fill=True, multiple="layer", color=CLASS0_COL,
             edgecolor="#FFFFFF", linewidth=1.15, alpha=0.50,
             label="True 0: OR-", ax=ax)
sns.histplot(prob_pos, bins=50, stat="density", element="bars",
             fill=True, multiple="layer", color=TEST_COL,
             edgecolor="#FFFFFF", linewidth=1.15, alpha=0.50,
             label="True 1: OR+", ax=ax)
if len(np.unique(prob_neg)) > 1:
    sns.kdeplot(prob_neg, color=CLASS0_COL, linewidth=1.8, cut=0, ax=ax)
if len(np.unique(prob_pos)) > 1:
    sns.kdeplot(prob_pos, color=TEST_COL, linewidth=1.8, cut=0, ax=ax)
ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.5,
           label=f"Threshold = {THRESHOLD:.2f}")
ax.set_xlim(0, 1)
ax.set_xlabel("Predicted Probability of Positive Class (OR+)", fontsize=LABEL_FS)
ax.set_ylabel("Density", fontsize=LABEL_FS)
ax.set_title("OOF Probability Distribution (5-fold)",
             fontsize=TITLE_FS, fontweight="bold")
ax.tick_params(labelsize=TICK_FS)
ax.grid(True, linestyle="--", alpha=0.35, color="lightgray")
ax.legend(frameon=False, fontsize=LABEL_FS - 2)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"Fig3_Test_Probability_Distribution.{ext}"),
                dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Fig3 saved.")

# ══════════════════════════════════════════════════════════════════════════
# FIG 4 — Classification Metrics Table (single OOF column)
# ══════════════════════════════════════════════════════════════════════════
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
table_data = [[m, f"{oof_m[m]:.3f}"] for m in METRIC_NAMES]

figure_height = max(4.8, 1.0 + 0.62 * len(METRIC_NAMES))
fig, ax = plt.subplots(figsize=(7, figure_height))
ax.axis("off")
tbl = ax.table(
    cellText=table_data,
    colLabels=["Metric", "OOF (5-fold)"],
    cellLoc="center", colLoc="center", loc="center",
    colWidths=[0.5, 0.4],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(LABEL_FS)
tbl.scale(1.0, 2.2)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("lightgray")
    cell.set_linewidth(0.8)
    if row == 0:
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor(TEST_COL)
    else:
        cell.set_facecolor("#F5F5F5" if row % 2 == 1 else "#EBEBEB")
ax.set_title("Classification Metrics", fontsize=TITLE_FS, fontweight="bold", pad=15)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"Fig4_Classification_Metrics.{ext}"),
                dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Fig4 saved.")

print("\nModel B OOF metrics:", {k: round(v, 4) for k, v in oof_m.items()})
print(f"All 4 figures rebuilt (simulated_data_used=False) to:\n  {OUT}")
