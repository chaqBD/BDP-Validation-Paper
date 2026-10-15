"""Generates every figure used in the paper, at publication quality (300 DPI),
from the canonical CSVs `analyze_release_review.py` and the supplementary
scripts produce. No figure here introduces a number that is not already in one
of those CSVs; this script only visualises them.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR

IN_DIR = RESULTS_DIR / "release_review"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 10.5,
    "font.family": "DejaVu Sans", "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "axes.axisbelow": True,
})

C_DATACO = "#2166ac"
C_OLIST = "#d6604d"
C_BLOCK = "#b2182b"
C_REVIEW = "#f4a582"
C_PASS = "#4393c3"
C_CORROB = "#1a9850"
C_METHOD = "#b2182b"


def fig1_detection_heatmap():
    """Figure 1: defect-specific detection heatmap, all six approaches kept
    separate (stock, extended and temporal reported individually, never
    combined into one boolean, per the primary endpoint in
    `primary_defect_check_detection.csv` / `detection_by_defect_type.csv`)."""
    prim = pd.read_csv(IN_DIR / "primary_defect_check_detection.csv")
    agg = pd.read_csv(IN_DIR / "detection_by_defect_type.csv")

    defects = ["leaky_feature", "exact_duplicate", "invalid_schema", "group_overlap"]
    datasets = ["dataco", "olist"]
    rows = [(d, t) for d in datasets for t in defects]
    cols = ["AUROC-only", "Metrics-only", "Automated checklist",
            "Stock gate\n(named check)", "Extension\n(group id)", "Temporal\nintegrity"]

    mat = np.full((len(rows), len(cols)), np.nan)
    for i, (ds, dt) in enumerate(rows):
        a = agg[(agg.dataset == ds) & (agg.defect_type == dt)]
        mat[i, 0] = a[a.approach == "AUROC-only"]["blocked_rate"].mean()
        mat[i, 1] = a[a.approach == "Metrics-only"]["blocked_rate"].mean()
        mat[i, 2] = a[a.approach == "Automated checklist"]["blocked_rate"].mean()
        p = prim[(prim.dataset == ds) & (prim.defect_type == dt)]
        mat[i, 3] = p["stock_named_check_fired"].mean()
        mat[i, 4] = p["extension_fired"].mean()
        mat[i, 5] = p["temporal_integrity_fired"].mean()

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=9.5)
    row_labels = [f"{'DataCo' if ds=='dataco' else 'Olist'} – {dt.replace('_', ' ')}"
                  for ds, dt in rows]
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(row_labels, fontsize=9.5)
    ax.hlines(3.5, -0.5, len(cols) - 0.5, color="black", linewidth=1.4)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = mat[i, j]
            if np.isnan(v):
                ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, hatch="///",
                                                 fill=False, edgecolor="#999999", linewidth=0))
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="#777777")
            else:
                txt_color = "white" if (v > 0.75 or v < 0.25) else "black"
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9, color=txt_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Detection rate (fraction of model families)", fontsize=9)
    ax.set_title("Figure 1. Defect-specific detection across all six approaches\n"
                 "(stock gate's named check, the extension and the temporal check reported separately, never combined)",
                 fontsize=10.5, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_detection_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_detection_heatmap.png")


def fig2_sampling_sensitivity():
    """Figure 2: overlap recovery vs. sampling cap and injected duplication
    severity (left), and the capped-vs-full understatement at this battery's
    own 20% severity (right), both datasets, with the gate's own 1%/5%
    thresholds marked on the left panel."""
    sev = pd.read_csv(IN_DIR / "heldout_severity_sweep.csv")
    full = pd.read_csv(IN_DIR / "sampling_sensitivity.csv")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.2, 5.0),
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    for ds, color, marker in [("dataco", C_DATACO, "o"), ("olist", C_OLIST, "s")]:
        d = sev[sev.dataset == ds].copy()
        d["pct"] = d["split_overlap_fraction"] * 100
        means = d.groupby("severity_frac")["pct"].mean()
        ax.scatter(d["severity_frac"] * 100, d["pct"], color=color, marker=marker,
                   alpha=0.55, s=34, zorder=3,
                   label=f"{'DataCo' if ds=='dataco' else 'Olist'} (capped 4,000/side, 3 seeds/severity)")
        ax.plot(means.index * 100, means.values, color=color, linewidth=1.6, zorder=2)

    ax.axhline(1.0, color="#555555", linewidth=1.0, linestyle="--")
    ax.text(0.6, 1.1, "NEEDS_REVIEW threshold (1%)", fontsize=8, color="#555555")
    ax.set_xlim(0, 32)
    ax.set_ylim(-0.15, 2.4)
    ax.set_xlabel("Injected duplication severity (% of test rows also in training)")
    ax.set_ylabel("split_overlap check's own\nmeasured overlap, capped sample (%)")
    ax.set_title("Capped measurement vs. severity\n(zoomed; both datasets' full-data rate is ~20% at every severity)", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=8, frameon=True)

    # Right panel: capped-at-20%-severity vs full, this battery's own instance.
    bar_labels, capped_vals, full_vals, ratios = [], [], [], []
    for ds in ["dataco", "olist"]:
        full_row = full[(full.case == "defect_exact_duplicate") & (full.dataset == ds)
                        & (full["sample"].str.startswith("full"))]
        capped_row = full[(full.case == "defect_exact_duplicate") & (full.dataset == ds)
                          & (full["sample"].str.startswith("capped"))]
        if len(full_row) and len(capped_row):
            f = float(full_row["overlap_with_training_fraction"].iloc[0]) * 100
            c = float(capped_row["overlap_with_training_fraction"].iloc[0]) * 100
            bar_labels.append("DataCo" if ds == "dataco" else "Olist")
            full_vals.append(f)
            capped_vals.append(c)
            ratios.append(f / c if c else np.nan)

    x = np.arange(len(bar_labels))
    w = 0.32
    ax2.bar(x - w / 2, full_vals, width=w, color="#999999", label="Full data (uncapped)")
    ax2.bar(x + w / 2, capped_vals, width=w, color=[C_DATACO, C_OLIST], label="Capped 4,000/side")
    for i, r in enumerate(ratios):
        ax2.annotate(f"{r:.1f}x\nunderstated", xy=(x[i] + w / 2, capped_vals[i]),
                    xytext=(x[i] + w / 2, capped_vals[i] + 2.2), ha="center", fontsize=8.5,
                    arrowprops=dict(arrowstyle="-", color="#555555", lw=0.7))
    ax2.axhline(1.0, color="#555555", linewidth=0.9, linestyle="--")
    ax2.axhline(5.0, color="#555555", linewidth=0.9, linestyle="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(bar_labels)
    ax2.set_ylabel("Measured overlap (%)")
    ax2.set_title("This battery's own case (20% severity)\nfull vs. capped", fontsize=9.5)
    handles = [mpatches.Patch(color="#999999", label="Full data (uncapped)"),
               mpatches.Patch(color="#bbbbbb", label="Capped 4,000/side")]
    ax2.legend(handles=handles, loc="upper right", fontsize=7.8, frameon=True)

    fig.suptitle("Figure 2. The stock gate's duplication check against its own sampling cap", fontsize=11.5, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_sampling_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_sampling_sensitivity.png")


def fig3_verdict_transition():
    """Figure 3: stock-to-extended verdict transition slopegraph, every
    case-model row in the main battery."""
    battery = pd.read_csv(IN_DIR / "release_review_battery.csv")
    order = {"PASS": 0, "NEEDS_REVIEW": 1, "BLOCKED": 2}
    ypos = {0: 0, 1: 1, 2: 2}

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    rng = np.random.default_rng(0)
    counts = {}
    for _, r in battery.iterrows():
        y0 = order[r["verdict_stock_gate"]]
        y1 = order[r["verdict_extended_gate"]]
        counts[(y0, y1)] = counts.get((y0, y1), 0) + 1

    for (y0, y1), n in counts.items():
        changed = y0 != y1
        color = C_BLOCK if changed else "#999999"
        lw = 0.8 + 2.6 * (n / battery.shape[0])
        jitter = rng.uniform(-0.05, 0.05)
        ax.plot([0, 1], [y0 + jitter, y1 + jitter], color=color, linewidth=lw,
                alpha=0.85 if changed else 0.45, zorder=3 if changed else 2,
                solid_capstyle="round")
        ax.annotate(str(n), xy=(0.5, (y0 + y1) / 2 + jitter), fontsize=8,
                    ha="center", va="center", color=color,
                    bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec=color, lw=0.8))

    for x, label in [(0, "Stock gate"), (1, "Extended gate")]:
        for level, name in [(0, "PASS"), (1, "NEEDS_\nREVIEW"), (2, "BLOCKED")]:
            ax.scatter([x], [level], s=1400, color="white", edgecolor="#333333",
                       zorder=4, linewidth=1.2)
            ax.text(x, level, name, ha="center", va="center", fontsize=8, fontweight="bold",
                    zorder=5)
        ax.text(x, 2.65, label, ha="center", fontsize=11, fontweight="bold")

    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.4, 3.0)
    ax.axis("off")
    ax.set_title("Figure 3. Verdict transitions: stock gate vs. extended gate\n"
                 "every case x model row (n=42); red = the verdict category itself changes; "
                 "circled numbers count rows on that path",
                 fontsize=10.5)
    legend_elems = [Line2D([0], [0], color=C_BLOCK, lw=2.2, label="Verdict changes (stock -> extended)"),
                    Line2D([0], [0], color="#999999", lw=2.2, label="No change")]
    ax.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.08),
              ncol=2, fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_verdict_transition.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_verdict_transition.png")


def fig4_review_burden_vs_corroboration():
    """Figure 4: review burden decomposed into metrics-corroborated vs.
    methodology-specific flags, on non-defect cases, by approach, plus overall
    review burden across the whole battery as a reference line."""
    fa = pd.read_csv(IN_DIR / "flagrate_by_nondefect_case.csv")
    agg = fa.groupby("approach")[["metrics_corroborated_flag_rate",
                                   "methodology_specific_flag_rate"]].mean()
    approach_order = ["AUROC-only", "Metrics-only", "Automated checklist",
                       "Stock gate", "Extended gate", "Temporal integrity"]
    agg = agg.reindex(approach_order)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    x = np.arange(len(approach_order))
    bar1 = ax.bar(x, agg["metrics_corroborated_flag_rate"], color=C_CORROB, width=0.6,
                  label="Metrics-corroborated (same-period metrics rule also flags it)")
    bar2 = ax.bar(x, agg["methodology_specific_flag_rate"], width=0.6,
                  bottom=agg["metrics_corroborated_flag_rate"], color=C_METHOD,
                  label="Methodology-specific (flagged, but metrics rule does not corroborate)")
    ax.set_xticks(x)
    ax.set_xticklabels(approach_order, rotation=20, ha="right", fontsize=9.5)
    ax.set_ylabel("Flag rate on clean / scenario (non-defect) cases")
    ax.set_ylim(0, 1.05)
    for i, (c, m) in enumerate(zip(agg["metrics_corroborated_flag_rate"],
                                     agg["methodology_specific_flag_rate"])):
        total = c + m
        if total > 0.01:
            ax.text(i, total + 0.02, f"{total:.0%}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_title("Figure 4. Flags on legitimate (non-defect) cases: how much is corroborated?\n"
                 "clean reference + both scenario cases, both datasets, all model families",
                 fontsize=10.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, fontsize=8.5, frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_review_burden_corroboration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig4_review_burden_corroboration.png")


def fig5_subsequent_period_validity():
    """Figure 5: precision/recall of each approach's flag for predicting a
    genuinely subsequent, independently fitted period failing a common
    acceptance criterion (scripts/run_subsequent_period_check.py)."""
    res = pd.read_csv(IN_DIR / "subsequent_period_check_summary.csv")
    labels = {"verdict_auroc_only": "AUROC-only", "verdict_metrics_only": "Metrics-only",
              "verdict_checklist": "Automated\nchecklist", "verdict_stock_gate": "Stock gate",
              "verdict_extended_gate": "Extended gate", "verdict_temporal_integrity": "Temporal\nintegrity"}
    res["label"] = res["approach"].map(labels)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    colors = [C_DATACO, C_DATACO, C_DATACO, C_OLIST, C_OLIST, "#777777"]
    # Jitter text offsets for points sharing identical (recall, precision), so
    # labels never overlap even when two approaches flag the exact same set of
    # origins (metrics-only/checklist; stock/extended gate both do, here).
    seen_coords = {}
    for i, row in res.iterrows():
        if pd.isna(row["precision"]):
            continue
        key = (round(row["recall"], 3), round(row["precision"], 3))
        n_at_key = seen_coords.get(key, 0)
        seen_coords[key] = n_at_key + 1
        ax.scatter(row["recall"], row["precision"], s=220, color=colors[i % len(colors)],
                  edgecolor="black", linewidth=0.8, zorder=3)
        xoff, yoff = 10, 8 - 26 * n_at_key
        ax.annotate(row["label"].replace("\n", " "), (row["recall"], row["precision"]),
                    xytext=(xoff, yoff), textcoords="offset points", fontsize=9)
    ax.annotate("Temporal integrity: 0 defects detected here by design\n"
                "(it targets group/temporal dependence, not generic\n"
                "performance drift -- not a comparable failure mode)",
                xy=(0.02, 0.05), fontsize=7.8, color="#555555", style="italic")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Recall (share of genuine next-period acceptance failures flagged)")
    ax.set_ylabel("Precision (share of flags whose next period actually failed)")
    ax.set_title("Figure 5. Does a flag predict the NEXT, independently fitted period's\n"
                 "own model failing a common 0.65 AUROC acceptance criterion?\n"
                 "(18 rolling-origin pairs pooled across both datasets; descriptive, not a significance test)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_subsequent_period_validity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig5_subsequent_period_validity.png")


if __name__ == "__main__":
    fig1_detection_heatmap()
    fig2_sampling_sensitivity()
    fig3_verdict_transition()
    fig4_review_burden_vs_corroboration()
    fig5_subsequent_period_validity()
    print("\nAll figures saved to", FIG_DIR)
