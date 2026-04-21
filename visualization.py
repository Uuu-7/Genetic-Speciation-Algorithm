"""Publication-quality visualization for GA / GEA / GSA comparison.

Generates six figure sets targeting paper-report level:
  Fig 1  — Convergence curves (mean ± std band), all 3 instances side-by-side
  Fig 2  — Grouped bar chart with BKS reference, all 3 instances
  Fig 3  — Box plots: result distribution across runs
  Fig 4  — Scalability: GSA advantage vs problem size
  Fig 5  — Species dynamics (count + diversity dual-axis), one run
  Fig 6  — Species size evolution (stacked area, clean palette)
"""

import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats

matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})

# ── Palette ──────────────────────────────────────────────────────────────
C = {
    "GA":  "#4878CF",   # steel blue
    "GEA": "#E87722",   # orange
    "GSA": "#2CA02C",   # green
    "bks": "#CC2222",   # red dashed  (best known solution)
    "div": "#9467BD",   # purple      (diversity)
}

RESULT_PATH = "results/experiment_results.json"
FIG_ROOT    = "results/figures_pub"

# Best Known Solutions (Augerat A-instances)
BKS = {
    "A-n32-k5.vrp":  784,
    "A-n60-k9.vrp":  1354,
    "A-n80-k10.vrp": 1763,
}

INSTANCE_LABELS = {
    "A-n32-k5.vrp":  "A-n32-k5\n(31 customers, 5 vehicles)",
    "A-n60-k9.vrp":  "A-n60-k9\n(59 customers, 9 vehicles)",
    "A-n80-k10.vrp": "A-n80-k10\n(79 customers, 10 vehicles)",
}

ALGOS = ["GA", "GEA", "GSA"]


def load_results():
    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# Fig 1 — Convergence: mean ± std band, all instances in one figure
# ══════════════════════════════════════════════════════════════════════════
def fig1_convergence(results):
    instances = list(results.keys())
    n = len(instances)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, inst in zip(axes, instances):
        data = results[inst]
        bks  = BKS.get(inst)
        for algo in ALGOS:
            arr = np.array(data["histories"][algo], dtype=float)
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            x    = np.arange(len(mean))
            ax.plot(x, mean, color=C[algo], lw=1.8, label=algo)
            ax.fill_between(x, mean - std, mean + std,
                            color=C[algo], alpha=0.13)
        if bks:
            ax.axhline(bks, color=C["bks"], lw=1.2,
                       linestyle="--", label=f"BKS={bks}")
        ax.set_title(INSTANCE_LABELS[inst], pad=8)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness" if ax is axes[0] else "")
        ax.legend(loc="upper right", framealpha=0.85)

    fig.suptitle("Mean Convergence Curves (± 1 Std. Dev.)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(FIG_ROOT, "fig1_convergence_all.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# Fig 2 — Grouped bar chart with BKS reference line
# ══════════════════════════════════════════════════════════════════════════
def fig2_grouped_bar(results):
    instances = list(results.keys())
    n_inst  = len(instances)
    n_algo  = len(ALGOS)
    x       = np.arange(n_inst)
    width   = 0.22

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, algo in enumerate(ALGOS):
        means = [results[inst]["summary"][algo]["mean"] for inst in instances]
        stds  = [results[inst]["summary"][algo]["std"]  for inst in instances]
        bars  = ax.bar(x + (i - 1) * width, means, width,
                       label=algo, color=C[algo], alpha=0.85,
                       yerr=stds, capsize=4, error_kw={"lw": 1.2})

    # BKS markers
    for j, inst in enumerate(instances):
        bks = BKS.get(inst)
        if bks:
            ax.plot([x[j] - 1.5 * width, x[j] + 1.5 * width],
                    [bks, bks], color=C["bks"], lw=2.0,
                    linestyle="--", zorder=5)

    # Percentage improvement labels (GSA vs GEA)
    for j, inst in enumerate(instances):
        gea_m = results[inst]["summary"]["GEA"]["mean"]
        gsa_m = results[inst]["summary"]["GSA"]["mean"]
        imp   = (gea_m - gsa_m) / gea_m * 100
        ax.annotate(f"↓{imp:.1f}%",
                    xy=(x[j] + width, gsa_m),
                    xytext=(x[j] + width + 0.02, gsa_m * 1.01),
                    fontsize=9, color=C["GSA"], fontweight="bold")

    # BKS legend entry
    bks_line = Line2D([0], [0], color=C["bks"], lw=2, linestyle="--",
                      label="Best Known Solution (BKS)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [bks_line], labels + ["BKS"],
              loc="upper left", framealpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels([INSTANCE_LABELS[i] for i in instances])
    ax.set_ylabel("Mean Objective Value")
    ax.set_title("Algorithm Comparison: Mean Objective (± Std. Dev.)",
                 fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIG_ROOT, "fig2_grouped_bar.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# Fig 3 — Box plots: result distribution across runs
# ══════════════════════════════════════════════════════════════════════════
def fig3_boxplot(results):
    instances = list(results.keys())
    n = len(instances)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, inst in zip(axes, instances):
        data_by_algo = [results[inst]["results"][a] for a in ALGOS]
        bp = ax.boxplot(data_by_algo, patch_artist=True,
                        widths=0.45, notch=False,
                        medianprops={"color": "black", "lw": 2})
        for patch, algo in zip(bp["boxes"], ALGOS):
            patch.set_facecolor(C[algo])
            patch.set_alpha(0.75)

        bks = BKS.get(inst)
        if bks:
            ax.axhline(bks, color=C["bks"], lw=1.5, linestyle="--",
                       label=f"BKS = {bks}")
            ax.legend(fontsize=9)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(ALGOS)
        ax.set_title(INSTANCE_LABELS[inst], pad=8)
        ax.set_ylabel("Objective Value" if ax is axes[0] else "")

    fig.suptitle("Result Distribution Across Independent Runs",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(FIG_ROOT, "fig3_boxplot.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# Fig 4 — Scalability: mean & gap-to-BKS vs problem size
# ══════════════════════════════════════════════════════════════════════════
def fig4_scalability(results):
    sizes     = [31, 59, 79]           # number of customers
    instances = list(results.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: mean objective value vs size
    for algo in ALGOS:
        means = [results[inst]["summary"][algo]["mean"] for inst in instances]
        stds  = [results[inst]["summary"][algo]["std"]  for inst in instances]
        ax1.plot(sizes, means, "o-", color=C[algo], lw=2,
                 markersize=7, label=algo)
        ax1.fill_between(sizes,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         color=C[algo], alpha=0.12)

    bks_vals = [BKS[inst] for inst in instances]
    ax1.plot(sizes, bks_vals, "s--", color=C["bks"], lw=1.5,
             markersize=6, label="BKS")
    ax1.set_xlabel("Number of Customers")
    ax1.set_ylabel("Mean Objective Value")
    ax1.set_title("Scalability: Solution Quality", fontweight="bold")
    ax1.set_xticks(sizes)
    ax1.legend()

    # Right: gap to BKS (%)
    for algo in ALGOS:
        gaps = [(results[inst]["summary"][algo]["mean"] - BKS[inst])
                / BKS[inst] * 100 for inst in instances]
        ax2.plot(sizes, gaps, "o-", color=C[algo], lw=2,
                 markersize=7, label=algo)

    ax2.axhline(0, color=C["bks"], lw=1.2, linestyle="--")
    ax2.set_xlabel("Number of Customers")
    ax2.set_ylabel("Gap to BKS (%)")
    ax2.set_title("Scalability: Gap to Best Known Solution", fontweight="bold")
    ax2.set_xticks(sizes)
    ax2.legend()

    fig.tight_layout()
    path = os.path.join(FIG_ROOT, "fig4_scalability.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# Fig 5 — Species dynamics: count + diversity on dual-axis (averaged over runs)
# ══════════════════════════════════════════════════════════════════════════
def fig5_species_dynamics(results):
    instances = list(results.keys())
    n = len(instances)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, inst in zip(axes, instances):
        stats_all = results[inst]["gsa_species_stats"]
        if not stats_all:
            continue

        # Average species count and diversity across all runs
        max_len = max(len(s["species_count"]) for s in stats_all)

        count_mat = np.full((len(stats_all), max_len), np.nan)
        div_mat   = np.full((len(stats_all), max_len), np.nan)
        for r, s in enumerate(stats_all):
            c = np.array(s["species_count"], dtype=float)
            d = np.array([np.mean(x) if x else 0.0
                          for x in s["species_diversity"]], dtype=float)
            count_mat[r, :len(c)] = c
            div_mat[r, :len(d)]   = d

        mean_count = np.nanmean(count_mat, axis=0)
        mean_div   = np.nanmean(div_mat,   axis=0)
        x = np.arange(max_len)

        # Species count (left y-axis)
        ax.step(x, mean_count, color=C["GEA"], lw=1.8,
                where="post", label="Avg Species Count")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Number of Species", color=C["GEA"])
        ax.tick_params(axis="y", labelcolor=C["GEA"])
        ax.set_ylim(0, 7)

        # Diversity (right y-axis)
        ax2 = ax.twinx()
        ax2.plot(x, mean_div, color=C["div"], lw=1.4,
                 alpha=0.85, label="Avg Intra-Species Diversity")
        ax2.set_ylabel("Avg Jaccard Distance", color=C["div"])
        ax2.tick_params(axis="y", labelcolor=C["div"])
        ax2.set_ylim(0, 1.05)
        ax2.spines["right"].set_visible(True)

        # Combined legend
        h1 = Line2D([0], [0], color=C["GEA"],  lw=1.8, label="Species Count")
        h2 = Line2D([0], [0], color=C["div"],  lw=1.4, label="Intra-Species Diversity")
        ax.legend(handles=[h1, h2], loc="upper right", fontsize=9, framealpha=0.85)
        ax.set_title(INSTANCE_LABELS[inst], pad=8)

    fig.suptitle("GSA Species Dynamics (Averaged over All Runs)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(FIG_ROOT, "fig5_species_dynamics.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# Fig 6 — Species size evolution (stacked area, clean palette)
# ══════════════════════════════════════════════════════════════════════════
def fig6_species_sizes(results):
    instances = list(results.keys())
    n = len(instances)
    # Use run_idx=0 (first run) for illustration
    PALETTE = ["#4878CF", "#E87722", "#2CA02C", "#D62728", "#9467BD"]

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.0))
    if n == 1:
        axes = [axes]

    for ax, inst in zip(axes, instances):
        stats_all = results[inst]["gsa_species_stats"]
        if not stats_all:
            continue
        sizes = stats_all[0]["species_sizes"]
        if not sizes:
            continue

        max_sp = max(len(x) for x in sizes)
        padded = np.array([x + [0] * (max_sp - len(x)) for x in sizes],
                          dtype=float).T      # shape (max_sp, T)

        x = np.arange(padded.shape[1])
        ax.stackplot(x, padded,
                     colors=PALETTE[:max_sp], alpha=0.82)

        patches = [mpatches.Patch(color=PALETTE[i], label=f"Species {i+1}")
                   for i in range(max_sp)]
        ax.legend(handles=patches, loc="upper right",
                  fontsize=8.5, framealpha=0.85)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Population Size" if ax is axes[0] else "")
        ax.set_ylim(0, 210)
        ax.set_title(INSTANCE_LABELS[inst], pad=8)

    fig.suptitle("GSA Species Size Evolution (Run 1)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(FIG_ROOT, "fig6_species_sizes.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# Fig 7 — Summary statistics table as figure (for direct insertion into paper)
# ══════════════════════════════════════════════════════════════════════════
def fig7_table(results):
    instances = list(results.keys())
    rows = []
    for inst in instances:
        bks = BKS.get(inst, None)
        for algo in ALGOS:
            s = results[inst]["summary"][algo]
            gap = (s["mean"] - bks) / bks * 100 if bks else float("nan")
            rows.append([
                inst.replace(".vrp", ""),
                algo,
                f"{s['best']:.1f}",
                f"{s['mean']:.1f}",
                f"{s['std']:.2f}",
                f"{gap:.1f}%",
                f"{s['avg_s']:.1f}s",
            ])

    col_labels = ["Instance", "Algorithm", "Best", "Mean",
                  "Std. Dev.", "Gap to BKS", "Avg. Time"]

    fig, ax = plt.subplots(figsize=(12, 0.55 * len(rows) + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.55)

    # Style header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight GSA rows and best values
    algo_col = 1
    row_idx = 1
    for inst in instances:
        for algo in ALGOS:
            bg = "#EAF4EA" if algo == "GSA" else (
                "#FFF3E0" if algo == "GEA" else "#EAF0FB")
            for j in range(len(col_labels)):
                tbl[row_idx, j].set_facecolor(bg)
            row_idx += 1

    ax.set_title("Experimental Results Summary",
                 fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    path = os.path.join(FIG_ROOT, "fig7_results_table.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    if not os.path.exists(RESULT_PATH):
        raise FileNotFoundError(f"Result file not found: {RESULT_PATH}")
    results = load_results()
    ensure_dir(FIG_ROOT)

    print("Generating publication-quality figures...")
    fig1_convergence(results)
    fig2_grouped_bar(results)
    fig3_boxplot(results)
    fig4_scalability(results)
    fig5_species_dynamics(results)
    fig6_species_sizes(results)
    fig7_table(results)
    print(f"\nAll figures saved to: {FIG_ROOT}/")


if __name__ == "__main__":
    main()