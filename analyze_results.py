import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

try:
    from scipy.stats import wilcoxon
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =========================================================
# Input / output paths
# =========================================================
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
STATS_DIR = RESULTS_DIR / "stats"

RUN_RESULTS_CSV = RESULTS_DIR / "experiment_run_results.csv"
SUMMARY_CSV = RESULTS_DIR / "experiment_summary.csv"
CONVERGENCE_CSV = RESULTS_DIR / "convergence_history.csv"
SPECIES_CSV = RESULTS_DIR / "species_history.csv"

WILCOXON_CSV = STATS_DIR / "wilcoxon_results.csv"


# =========================================================
# Plot
# =========================================================
SELECTED_ALGOS = ["GA", "GSA"]
ALGORITHM_ORDER = ["GA", "GSA"]
MAIN_CURVE_ALGOS = ["GA", "GSA"]
SPECIES_CURVE_ALGOS = ["GSA"]


# =========================================================
# BKS
# =========================================================
BKS = {
    "A-n32-k5.vrp": 784,
    "A-n60-k9.vrp": 1408,
    "A-n80-k10.vrp": 1763,
}


# =========================================================
# Utility
# =========================================================
def ensure_dirs():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)


def read_data():
    run_df = pd.read_csv(RUN_RESULTS_CSV)
    summary_df = pd.read_csv(SUMMARY_CSV)
    conv_df = pd.read_csv(CONVERGENCE_CSV)
    species_df = pd.read_csv(SPECIES_CSV)
    return run_df, summary_df, conv_df, species_df


def filter_algorithms(df: pd.DataFrame) -> pd.DataFrame:
    if "algorithm" not in df.columns:
        return df.copy()
    return df[df["algorithm"].isin(SELECTED_ALGOS)].copy()


def ordered_algorithms(df: pd.DataFrame) -> List[str]:
    existing = list(df["algorithm"].dropna().unique())
    ordered = [a for a in ALGORITHM_ORDER if a in existing]
    leftovers = [a for a in existing if a not in ordered]
    return ordered + sorted(leftovers)


# =========================================================
# Plot 1: convergence curves
# =========================================================
def plot_convergence_curves(conv_df: pd.DataFrame):
    conv_df = filter_algorithms(conv_df)
    algorithms = [a for a in MAIN_CURVE_ALGOS if a in conv_df["algorithm"].unique()]

    for instance in sorted(conv_df["instance"].unique()):
        sub = conv_df[conv_df["instance"] == instance].copy()
        sub = sub[sub["algorithm"].isin(algorithms)]

        if sub.empty:
            continue

        plt.figure(figsize=(8, 5))

        for algo in algorithms:
            algo_df = sub[sub["algorithm"] == algo]
            if algo_df.empty:
                continue

            grp = (
                algo_df.groupby("generation", as_index=False)["running_best_fitness"]
                .mean()
                .sort_values("generation")
            )

            plt.plot(
                grp["generation"],
                grp["running_best_fitness"],
                label=algo,
            )

        if instance in BKS:
            plt.axhline(
                y=BKS[instance],
                linestyle="--",
                linewidth=2,
                label="BKS",
            )

        plt.xlabel("Generation")
        plt.ylabel("Mean running best fitness")
        plt.title(f"Convergence curves - {instance}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"convergence_{instance}.png", dpi=200)
        plt.close()


# =========================================================
# Plot 2: boxplots
# =========================================================
def plot_boxplots(run_df: pd.DataFrame):
    run_df = filter_algorithms(run_df)
    algos = ordered_algorithms(run_df)

    for instance in sorted(run_df["instance"].unique()):
        sub = run_df[run_df["instance"] == instance].copy()

        data = []
        labels = []

        for algo in algos:
            vals = sub[sub["algorithm"] == algo]["best_fitness"].tolist()
            if vals:
                data.append(vals)
                labels.append(algo)

        if not data:
            continue

        plt.figure(figsize=(8, 5))
        plt.boxplot(data, labels=labels, showmeans=True)

        if instance in BKS:
            plt.axhline(
                y=BKS[instance],
                linestyle="--",
                linewidth=2,
                label="BKS",
            )

        plt.xlabel("Algorithm")
        plt.ylabel("Best fitness across runs")
        plt.title(f"Boxplot of fitness - {instance}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"boxplot_{instance}.png", dpi=200)
        plt.close()


# =========================================================
# Plot 3: species count curves
# =========================================================
def plot_species_curves(species_df: pd.DataFrame):
    species_df = filter_algorithms(species_df)
    sub = species_df[species_df["algorithm"].isin(SPECIES_CURVE_ALGOS)].copy()
    if sub.empty:
        return

    sub = sub.dropna(subset=["generation", "species_count"])
    sub = sub[
        ["algorithm", "instance", "seed", "generation", "species_count"]
    ].drop_duplicates()

    for instance in sorted(sub["instance"].unique()):
        inst_df = sub[sub["instance"] == instance]
        if inst_df.empty:
            continue

        plt.figure(figsize=(8, 5))

        for algo in [a for a in SPECIES_CURVE_ALGOS if a in inst_df["algorithm"].unique()]:
            algo_df = inst_df[inst_df["algorithm"] == algo]
            grp = (
                algo_df.groupby("generation", as_index=False)["species_count"]
                .mean()
                .sort_values("generation")
            )

            plt.plot(
                grp["generation"],
                grp["species_count"],
                label=algo,
            )

        plt.xlabel("Generation")
        plt.ylabel("Mean number of species")
        plt.title(f"GSA species evolution - {instance}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"species_curve_{instance}.png", dpi=200)
        plt.close()


# =========================================================
# Summary printing
# =========================================================
def print_summary_table(summary_df: pd.DataFrame):
    summary_df = filter_algorithms(summary_df)

    print("\n===== Summary table (GA vs GSA) =====")

    preferred_cols = [
        "instance",
        "algorithm",
        "best",
        "mean",
        "std",
        "worst",
        "mean_runtime_seconds",
        "mean_convergence_generation",
        "mean_species_count",
        "max_species_count",
    ]

    cols = [c for c in preferred_cols if c in summary_df.columns]
    if not summary_df.empty and cols:
        print(summary_df[cols].to_string(index=False))
    else:
        print("No summary data for selected algorithms.")


def print_presentation_summary(summary_df: pd.DataFrame):
    summary_df = filter_algorithms(summary_df)

    print("\n===== Presentation Summary (GA vs GSA) =====")
    for instance in sorted(summary_df["instance"].unique()):
        sub = summary_df[summary_df["instance"] == instance].copy()
        if sub.empty:
            continue

        print(f"\nInstance: {instance}")
        for algo in ["GA", "GSA"]:
            row = sub[sub["algorithm"] == algo]
            if row.empty:
                continue
            row = row.iloc[0]
            print(
                f"  {algo}: best={row['best']}, mean={row['mean']}, "
                f"std={row['std']}, runtime={row['mean_runtime_seconds']}s, "
                f"species_mean={row['mean_species_count']}"
            )


# =========================================================
# Wilcoxon signed-rank test
# =========================================================
def paired_values(
    run_df: pd.DataFrame,
    instance: str,
    algo_a: str,
    algo_b: str,
    metric: str = "best_fitness",
) -> Tuple[List[float], List[float]]:
    a_df = run_df[
        (run_df["instance"] == instance) & (run_df["algorithm"] == algo_a)
    ][["seed", metric]].copy()

    b_df = run_df[
        (run_df["instance"] == instance) & (run_df["algorithm"] == algo_b)
    ][["seed", metric]].copy()

    merged = pd.merge(a_df, b_df, on="seed", suffixes=("_a", "_b")).sort_values("seed")
    return merged[f"{metric}_a"].tolist(), merged[f"{metric}_b"].tolist()


def better_label(mean_a: float, mean_b: float, algo_a: str, algo_b: str) -> str:
    if mean_a < mean_b:
        return algo_a
    if mean_b < mean_a:
        return algo_b
    return "tie"


def run_wilcoxon_tests(run_df: pd.DataFrame):
    run_df = filter_algorithms(run_df)

    if not SCIPY_AVAILABLE:
        print("\n[Warning] scipy is not installed. Skip Wilcoxon tests.")
        return

    comparisons = [
        ("GSA", "GA"),
    ]

    rows = []

    for instance in sorted(run_df["instance"].unique()):
        for algo_a, algo_b in comparisons:
            if algo_a not in run_df["algorithm"].unique():
                continue
            if algo_b not in run_df["algorithm"].unique():
                continue

            vals_a, vals_b = paired_values(
                run_df=run_df,
                instance=instance,
                algo_a=algo_a,
                algo_b=algo_b,
                metric="best_fitness",
            )

            if not vals_a or not vals_b:
                continue
            if len(vals_a) != len(vals_b):
                continue

            mean_a = sum(vals_a) / len(vals_a)
            mean_b = sum(vals_b) / len(vals_b)

            try:
                stat, p_value = wilcoxon(
                    vals_a,
                    vals_b,
                    zero_method="wilcox",
                    alternative="two-sided",
                )
            except ValueError:
                stat, p_value = math.nan, math.nan

            significant = (not math.isnan(p_value)) and (p_value < 0.05)
            winner = better_label(mean_a, mean_b, algo_a, algo_b)

            rows.append({
                "instance": instance,
                "metric": "best_fitness",
                "algo_a": algo_a,
                "algo_b": algo_b,
                "mean_a": round(mean_a, 6),
                "mean_b": round(mean_b, 6),
                "wilcoxon_stat": stat if math.isnan(stat) else round(float(stat), 6),
                "p_value": p_value if math.isnan(p_value) else round(float(p_value), 10),
                "significant_at_0_05": significant,
                "better_mean": winner,
            })

    if rows:
        out_df = pd.DataFrame(rows)
        out_df.to_csv(WILCOXON_CSV, index=False, encoding="utf-8")
        print("\n===== Wilcoxon results (GSA vs GA) =====")
        print(out_df.to_string(index=False))
        print(f"\nSaved: {WILCOXON_CSV}")
    else:
        print("\nNo valid Wilcoxon comparison data for GA vs GSA.")


# =========================================================
# Main
# =========================================================
def main():
    ensure_dirs()
    run_df, summary_df, conv_df, species_df = read_data()

    run_df = filter_algorithms(run_df)
    summary_df = filter_algorithms(summary_df)
    conv_df = filter_algorithms(conv_df)
    species_df = filter_algorithms(species_df)

    print_summary_table(summary_df)
    print_presentation_summary(summary_df)

    plot_convergence_curves(conv_df)
    plot_boxplots(run_df)
    plot_species_curves(species_df)
    run_wilcoxon_tests(run_df)

    print("\nSaved plots under:")
    print(f" - {PLOTS_DIR}")
    print("\nSaved statistics under:")
    print(f" - {STATS_DIR}")


if __name__ == "__main__":
    main()