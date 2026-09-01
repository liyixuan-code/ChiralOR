"""
SI Fig12 — native shap.force_plot + template stagger  (real Model-B OOF SHAP)
=============================================================================
Architecture EXACTLY matches the template & old JCIM script
(render_si_fig12_shap_native.py + XGBoost template _plot_figure_12):

  1. shap.force_plot(base, FULL 2283-dim OOF SHAP, features, matplotlib=True)
     -> native arrow/chevron band, native labels, native connector lines,
        native f(x)/base value/higher-lower markers.  Geometry 100% real.
  2. contribution_threshold auto-calibrated per sample so ~10 LOCAL top
     contributors are labelled (rest stay in the band, no "other features").
  3. Rename native labels to human-readable names + 3-decimal values IN PLACE.
  4. recolour()  -> #9C1A1C / #48A597   (template exact)
  5. refont()    -> Times New Roman
  6. stagger()   -> template 6-level anti-overlap (moves text AND connector end)
  7. fix_base_value_line() + fix_ylim()  (template exact)

Additivity: base + Σ(all 2283 SHAP) = real OOF RawFormulaVal (exact).
simulated_data_used = False.  Old numeric data NOT used (style template only).

Usage:  python render_fig12_native_stylelock.py fn1     # pilot
        python render_fig12_native_stylelock.py all     # all 10 panels
"""
import os, sys, re, json, hashlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import shap

WD = "C:/Users/lenovo/.claude/projects/PM/JCIM_MANUSCRIPT/output_revision"
os.chdir(WD)
OUTDIR = "SI_Fig12_NATIVE_STYLELOCK"
os.makedirs(OUTDIR, exist_ok=True)

_avail = {f.name for f in fm.fontManager.ttflist}
SERIF  = "Times New Roman" if "Times New Roman" in _avail else "DejaVu Serif"

# ── colour replacement (template exact) ───────────────────────────────────────
POS_TARGET = mcolors.to_rgb("#FF0051"); NEG_TARGET = mcolors.to_rgb("#008BFB")
POS_CUSTOM = "#9C1A1C";                 NEG_CUSTOM = "#48A597"

def _match(c):
    if c is None: return None
    try:
        rgb = mcolors.to_rgb(c)
        if sum((a-b)**2 for a,b in zip(rgb, POS_TARGET)) < 0.05: return POS_CUSTOM
        if sum((a-b)**2 for a,b in zip(rgb, NEG_TARGET)) < 0.05: return NEG_CUSTOM
    except: pass
    return None

def recolour(ax):
    for obj in ax.findobj():
        for getter, setter in [("get_color","set_color"),
                               ("get_facecolor","set_facecolor"),
                               ("get_edgecolor","set_edgecolor")]:
            if hasattr(obj, getter) and hasattr(obj, setter):
                c = getattr(obj, getter)()
                if isinstance(c, np.ndarray) and c.size >= 3:
                    c = c[0] if c.ndim == 2 else c
                nc = _match(c)
                if nc: getattr(obj, setter)(nc)

def refont(ax):
    # NOTE: do NOT touch mathtext (the native ← → arrows use $...$); forcing a
    # font family on them breaks the glyph rendering (arrows disappear).
    for t in ax.findobj(plt.Text):
        s = t.get_text()
        if "$" in s:            # mathtext (arrows) — leave native font
            continue
        t.set_fontfamily(SERIF)

def suppress_native_marker_line(ax):
    """
    Native shap.force_plot draws a thin red (#FF0D57) vertical line at the f(x)
    location dropping BELOW the band (y≈[0,-0.18]). The template/old-JCIM look
    does not show this stray line — remove it so it is not mistaken for a
    feature connector. Identify by: red-ish color AND y-range extending below 0.
    """
    for ln in list(ax.get_lines()):
        xd = ln.get_xdata(); yd = ln.get_ydata()
        if len(xd) == 2 and abs(float(xd[0]) - float(xd[1])) < 1e-6:
            y_lo = float(min(yd)); y_hi = float(max(yd))
            col = str(ln.get_color()).upper()
            is_reddish = ("FF0D57" in col) or ("FF0051" in col)
            if is_reddish and y_lo < -0.02 and y_hi <= 0.02:
                ln.remove()

def stagger(ax):
    """
    Template-EXACT stagger (render_si_fig12_shap_native.py / XGBoost template).
    Moves each feature-label text down to one of 6 levels AND moves the matching
    native connector-line endpoint to follow, so connectors never pierce text.
    Top annotations (f(x)/base value/higher-lower) are left in their NATIVE
    positions above the axis — unchanged.
    """
    texts = [t for t in ax.texts if "=" in t.get_text()]
    texts.sort(key=lambda t: t.get_position()[0])
    lines   = ax.get_lines()
    x_span  = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_span  = ax.get_ylim()[1] - ax.get_ylim()[0]
    step    = y_span * 0.12
    x_thr   = x_span * 0.15
    levels  = [0.0, -step, -step*2, -step*3, -step*4, -step*5]
    last_x  = {lvl: -float("inf") for lvl in levels}
    min_y   = ax.get_ylim()[0]
    for txt in texts:
        x, y = txt.get_position()
        chosen = levels[0]
        for lvl in levels:
            if x - last_x[lvl] > x_thr: chosen = lvl; break
        last_x[chosen] = x
        new_y = y + chosen; min_y = min(min_y, new_y)
        if chosen != 0.0:
            txt.set_position((x, new_y))
            for ln in lines:
                xd, yd = ln.get_xdata(), ln.get_ydata()
                if len(xd)==2 and abs(float(xd[0])-x)<1e-3 and abs(float(xd[1])-x)<1e-3:
                    yd = list(yd)
                    if abs(yd[0]-y) < abs(yd[1]-y): yd[0] = new_y
                    else:                            yd[1] = new_y
                    ln.set_ydata(yd)
    ax.set_ylim(bottom=min_y - step)

    # ── f(x) / base value stay on the SAME native row (y=0.330), side by side:
    #        f(x)              base value
    #    Native shap already lays them out this way with a real horizontal gap
    #    (verified: text bounding boxes do NOT overlap even at the boundary,
    #    where f(x)≈base). So we DO NOT move base value onto another row.
    #    Only if the RENDERED text boxes genuinely overlap do we nudge base
    #    value horizontally away from f(x) (never vertically) to keep the row.
    fx_txt = bv_txt = None
    for t in ax.texts:
        if t.get_text() == "f(x)":       fx_txt = t
        elif t.get_text() == "base value": bv_txt = t
    if fx_txt is not None and bv_txt is not None:
        try:
            fig = ax.figure; fig.canvas.draw()
            r = fig.canvas.get_renderer()
            fb = fx_txt.get_window_extent(r); bb = bv_txt.get_window_extent(r)
            overlap_px = min(fb.x1, bb.x1) - max(fb.x0, bb.x0)
            if overlap_px > 0 and abs(fb.y0 - bb.y0) < (fb.y1 - fb.y0):
                # real overlap on same row -> push base value sideways in data x
                inv = ax.transData.inverted()
                pad_px = overlap_px + 10
                (x0d, _), (x1d, _) = inv.transform((0, 0)), inv.transform((pad_px, 0))
                shift = abs(x1d - x0d)
                bvx, bvy = bv_txt.get_position()
                fxx, _ = fx_txt.get_position()
                bv_txt.set_position((bvx + (-shift if bvx < fxx else shift), bvy))
        except Exception:
            pass

# ── human-readable feature names ──────────────────────────────────────────────
NMAP = {
    "geo:signed_tetra_volume":              "signed tetra vol",
    "geo:signed_dihedral_subst_c_rn1_rn2": "signed dih (subst)",
    "geo:baseline_pm_dihedral_sin":         "P/M dih (sin)",
    "geo:baseline_pm_dihedral_cos":         "P/M dih (cos)",
    "geo:signed_dihedral_NS_path":          "signed dih (N-S)",
    "geo:subst_to_ringplane_signed_dist":   "subst-ring dist",
    "geo:ringnormal_dot_substvec":          "ring-normal-subst",
    "dihedral":                             "baseline dih",
    "solvent_code":                         "solvent",
    "pm_code":                              "P/M code",
    "TPSA":                                 "TPSA",
    "3D_Sphero":                            "3D sphero",
    "3D_RoG":                               "3D RoG",
}
SOLVENT_MAP = {0:"MeOH",1:"EtOH",2:"CHCl3",3:"DCM",4:"H2O",
               5:"hexane",6:"dioxane",7:"acetone",8:"DMSO"}

def disp_name(f):
    if f in NMAP: return NMAP[f]
    if f.startswith("ECFP_"):  return "ECFP " + f[5:]
    if f.startswith("MACCS_"): return "MACCS-" + f[6:]
    if f.startswith("PAS_"):   return "PAS-" + f[4:]
    if f.startswith("sPAS_"):  return "sPAS-" + f[5:]
    return f.replace("geo:", "").replace("_", " ")[:18]

def fmt_val(v, fname):
    if fname == "solvent_code": return SOLVENT_MAP.get(int(round(v)), str(int(round(v))))
    if fname == "pm_code":      return "P" if v > 0 else "M"
    if float(v) == int(v) and abs(v) <= 1.0: return str(int(v))
    return f"{v:.3f}"

# ── load real data ────────────────────────────────────────────────────────────
sv    = np.load("oof_shap_values_modelB.npy")       # (5204,2283)
base  = np.load("oof_shap_base_values_modelB.npy")  # (5204,)
X     = np.load("X_C3_5204x2283.npy")              # (5204,2283)
master= pd.read_csv("master_labels_v2.csv")
oof   = pd.read_csv("oof_predictions_master_labels_v2.csv").sort_values("sample_id").reset_index(drop=True)
feat_all = pd.read_csv("feature_order_check.csv").feature_name.tolist()

SHA_SV = hashlib.sha256(open("oof_shap_values_modelB.npy","rb").read()).hexdigest()[:16]
assert SHA_SV == "63aa33a08f295491", f"SHAP sha mismatch: {SHA_SV}"

p     = oof.oof_prob.values
y     = master.OR_label.values
foldv = oof.fold.values
pred  = (p >= 0.5).astype(int)
compound = master.compound_id.values if "compound_id" in master.columns else np.array([f"cmpd_{i}" for i in range(len(y))])
sid_all = np.arange(len(y))

# ── deterministic, audited sample selection ───────────────────────────────────
FN = sid_all[(y==1)&(pred==0)]; FP = sid_all[(y==0)&(pred==1)]
TP = sid_all[(y==1)&(pred==1)]; TN = sid_all[(y==0)&(pred==0)]
tp_med = np.median(p[TP]); tn_med = np.median(p[TN])
PICKS = [
 ("FN1_high_confidence","High-confidence false negative", int(FN[np.argmin(p[FN])])),
 ("FN2_borderline",      "Borderline false negative",     int(FN[np.argmax(p[FN])])),
 ("FP1_high_confidence","High-confidence false positive", int(FP[np.argmax(p[FP])])),
 ("FP2_borderline",      "Borderline false positive",     int(FP[np.argmin(p[FP])])),
 ("TP_borderline",       "Borderline true positive",      int(TP[np.argmin(np.abs(p[TP]-0.5))])),
 ("TN_borderline",       "Borderline true negative",      int(TN[np.argmin(np.abs(p[TN]-0.5))])),
 ("TP_high_confidence", "High-confidence true positive",  int(TP[np.argmax(p[TP])])),
 ("TN_high_confidence", "High-confidence true negative",  int(TN[np.argmin(p[TN])])),
 ("TP_typical",          "Typical true positive",         int(TP[np.argmin(np.abs(p[TP]-tp_med))])),
 ("TN_typical",          "Typical true negative",         int(TN[np.argmin(np.abs(p[TN]-tn_med))])),
]

# ── per-sample threshold calibration for ~N labels ────────────────────────────
def n_native_labels(sid, thr, feat_series):
    plt.close("all")
    fig = shap.force_plot(float(base[sid]), sv[sid], feat_series,
                          matplotlib=True, show=False, contribution_threshold=thr)
    if fig is None: fig = plt.gcf()
    n = len([t for t in fig.gca().texts if "=" in t.get_text()])
    plt.close(fig)
    return n

def calibrate(sid, feat_series, target=10):
    lo, hi = 0.001, 0.15; best = (0.02, 999)
    for _ in range(22):
        mid = (lo+hi)/2
        n = n_native_labels(sid, mid, feat_series)
        if abs(n-target) < abs(best[1]-target): best = (mid, n)
        if n > target: lo = mid
        else:          hi = mid
    return best

def rename_labels(ax, sid):
    """Rewrite native '<raw_name> = <rawvalue>' texts to human-readable + 3dp."""
    for t in ax.texts:
        s = t.get_text()
        if "=" not in s: continue
        raw = s.rsplit("=", 1)[0].strip()
        # raw is the feature name as passed in the Series index
        if raw in feat_all:
            fi = feat_all.index(raw)
            dn = disp_name(raw)
            fv = fmt_val(float(X[sid, fi]), raw)
            t.set_text(f"{dn} = {fv}")

def render(sid, title, tag, prefix, audit=False):
    sv_full = sv[sid]; bval = float(base[sid])
    raw_fx  = bval + float(sv_full.sum())
    prob    = float(p[sid]); prob_rec = 1/(1+np.exp(-raw_fx))
    assert abs(prob_rec - prob) < 1e-6, f"additivity fail sid={sid}"

    # features passed with RAW names (so rename can map back); values rounded 3dp
    feat_series = pd.Series([round(float(v),3) for v in X[sid]], index=feat_all)

    thr, nlab = calibrate(sid, feat_series, target=10)

    plt.close("all")
    fig = shap.force_plot(bval, sv_full, feat_series, matplotlib=True,
                          show=False, contribution_threshold=thr)
    if fig is None: fig = plt.gcf()
    fig.set_size_inches(16, 4)
    ax = fig.gca()

    rename_labels(ax, sid)              # human-readable names + 3dp values
    recolour(ax)                        # #FF0051->#9C1A1C, #008BFB->#48A597
    refont(ax)                          # Times New Roman
    suppress_native_marker_line(ax)     # remove stray red f(x) drop line
    stagger(ax)                         # template-exact stagger + top de-overlap
    # NOTE: template adds NO fix_ylim / no top extension — native ylim preserved.

    # The native "higher/lower" + f(x)/base text sit ABOVE the axes box
    # (data-y > ylim-top). Find the highest native text in AXES-fraction coords
    # so the title can be placed clearly ABOVE it (no overlap with axis region).
    fig.canvas.draw()
    inv = ax.transAxes.inverted()
    top_frac = 1.0
    for t in ax.texts:
        if t.get_text().strip() == "":
            continue
        # text anchor in display coords -> axes fraction
        x_disp, y_disp = ax.transData.transform(t.get_position())
        _, y_ax = inv.transform((x_disp, y_disp))
        top_frac = max(top_frac, y_ax)
    title_y = top_frac + 0.06            # title sits above the highest native text
    ax.text(0.0, title_y, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=13, fontweight="bold", color="#1a1a1a", fontfamily=SERIF)
    ax.text(1.0, title_y, f"OOF p(OR+) = {prob:.3f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10, color="#666",
            fontstyle="italic", fontfamily=SERIF)

    for ext in ["png","pdf","svg"]:
        fig.savefig(f"{OUTDIR}/{prefix}.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # audit CSV of local top-10
    if audit:
        a = np.abs(sv_full); top = np.argsort(a)[::-1][:10]
        rows=[]
        for r,fi in enumerate(top,1):
            rows.append(dict(local_rank=r, feature_name=feat_all[fi],
                display_name=disp_name(feat_all[fi]),
                feature_value_full_precision=float(X[sid,fi]),
                feature_value_display=fmt_val(float(X[sid,fi]), feat_all[fi]),
                shap_value=float(sv_full[fi]), abs_shap=float(a[fi]),
                direction="positive" if sv_full[fi]>0 else "negative"))
        pd.DataFrame(rows).to_csv(f"{OUTDIR}/SI_Fig12_{tag}_local_top10.csv", index=False)

    return dict(tag=tag, display_title=title, sample_id=int(sid),
        fold_id=int(foldv[sid]), y_true=int(y[sid]), predicted_label=int(pred[sid]),
        oof_probability=float(prob), base_logodds=float(bval), raw_fx=float(raw_fx),
        sum_shap_2283=float(sv_full.sum()), abs_diff_prob=float(abs(prob_rec-prob)),
        contribution_threshold=float(thr), n_labels=int(nlab),
        additivity_pass=bool(abs(prob_rec-prob)<1e-6), simulated_data_used=False)

MODE = sys.argv[1] if len(sys.argv)>1 else "fn1"
if MODE == "fn1":
    tag, disp, sid = PICKS[0]
    r = render(sid, disp, tag, "SI_Fig12_FN1_NATIVE", audit=True)
    json.dump(r, open(f"{OUTDIR}/SI_Fig12_FN1_validation.json","w"), indent=2)
    print(json.dumps(r, indent=2))
elif MODE == "all":
    res=[]
    for tag, disp, sid in PICKS:
        r = render(sid, disp, tag, f"SI_Fig12_{tag}_NATIVE", audit=True)
        res.append(r); print(f"  {tag}: sid={sid} nlab={r['n_labels']} thr={r['contribution_threshold']:.4f} add={r['additivity_pass']}")
    pd.DataFrame(res).to_csv(f"{OUTDIR}/SI_Fig12_all_validation.csv", index=False)
    print(f"\nAll 10 saved. shap_sha={SHA_SV} simulated_data_used=False")
