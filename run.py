import csv
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

from gsa.problem import VRP
from gsa.gsa import GSA


# =========================================================
# Experiment preset
# =========================================================
EXPERIMENT_PRESET = "paper"
RUN_STYLE = "batch"


# =========================================================
# Instances
# =========================================================
INSTANCE_PATHS = [
    "data/vrp/A-n32-k5.vrp",
    "data/vrp/A-n60-k9.vrp",
    "data/vrp/A-n80-k10.vrp",
]

BKS = {
    # "A-n32-k5.vrp": 784,
    # "A-n60-k9.vrp": 1408,
    # "A-n80-k10.vrp": 1763,
}


# =========================================================
# Single mode
# =========================================================
SINGLE_MODE = "GSA"
SINGLE_INSTANCE_PATH = INSTANCE_PATHS[0]
SINGLE_SEED = 42


# =========================================================
# Batch mode
# =========================================================
BATCH_MODES = [
    "GA",
    "GEA",
    "GSA",
]

if EXPERIMENT_PRESET == "debug":
    BATCH_SEEDS = [1, 2, 3]
    COMMON_GENERATIONS = 80
    COMMON_POPULATION_SIZE = 60
else:
    # 10 runs are enough for the midterm report
    BATCH_SEEDS = list(range(1, 11))
    COMMON_GENERATIONS = 300
    COMMON_POPULATION_SIZE = 100


# =========================================================
# Output files
# =========================================================
RESULTS_DIR = Path("results")
RUN_RESULTS_CSV = RESULTS_DIR / "experiment_run_results.csv"
SUMMARY_CSV = RESULTS_DIR / "experiment_summary.csv"
REPORT_SUMMARY_CSV = RESULTS_DIR / "report_summary.csv"
CONVERGENCE_CSV = RESULTS_DIR / "convergence_history.csv"
SPECIES_CSV = RESULTS_DIR / "species_history.csv"
MODE_CONFIG_CSV = RESULTS_DIR / "mode_configurations.csv"


# =========================================================
# Algorithm order
# =========================================================
ALGO_ORDER = {
    "GA": 0,
    "GEA": 1,
    "GSA": 2,
}


# =========================================================
# Algorithm configurations
# =========================================================
def get_mode_config(mode: str) -> Dict[str, Any]:
    """
    Return full configuration for each mode.

    Design logic for the midterm report:
    - GA: baseline
    - GEA: single-population gene engineering algorithm
    - GSA: full proposed framework
    """
    mode = mode.upper().strip()

    common = dict(
        population_size=COMMON_POPULATION_SIZE,
        generations=COMMON_GENERATIONS,
        elite_ratio=0.2,
        tournament_size=3,
        mask_threshold=0.7,
        crossover_rate=0.8,
        mutation_rate=0.1,
        num_species=1,
        immigrant_rate=0.0,
        speciation_method="rank",
        kmeans_random_state=42,
        wishart_neighbors=7,
        wishart_significance_level=0.12,
        wishart_min_cluster_size=3,
        wishart_noise_policy="nearest",
        scenario1_rate=0.0,
        scenario2_rate=0.0,
        scenario3_rate=0.0,
        cross_species_rate=0.0,
        non_elite_injection_bias=0.9,
        worst_injection_ratio=0.3,
        local_search_rate=0.0,
        local_search_top_ratio=0.0,
        local_search_rounds=1,
    )

    if mode == "GA":
        return common

    if mode == "GEA":
        cfg = common.copy()
        cfg.update(
            scenario1_rate=0.5,
            scenario2_rate=0.5,
            scenario3_rate=0.2,
        )
        return cfg

    if mode == "GSA":
        cfg = common.copy()
        cfg.update(
            crossover_rate=0.7,
            mutation_rate=0.3,
            num_species=3,
            immigrant_rate=0.10,
            speciation_method="wishart",
            wishart_neighbors=9,
            wishart_significance_level=0.12,
            wishart_min_cluster_size=4,
            wishart_noise_policy="explorer",
            scenario1_rate=0.15,
            scenario2_rate=0.15,
            scenario3_rate=0.25,
            cross_species_rate=0.6,
            non_elite_injection_bias=0.9,
            worst_injection_ratio=0.3,
            local_search_rate=0.6,
            local_search_top_ratio=0.4,
            local_search_rounds=2,
        )
        return cfg

    raise ValueError(f"Unknown mode: {mode}")


def build_algo(mode: str, problem: VRP, seed: int) -> GSA:
    cfg = get_mode_config(mode)
    return GSA(
        problem=problem,
        seed=seed,
        **cfg,
    )


# =========================================================
# Utilities
# =========================================================
def first_hit_generation(history: List[float]):
    """
    Return the first generation index (1-based)
    where the final best fitness is reached.
    """
    if not history:
        return None
    final_best = min(history)
    for i, v in enumerate(history, start=1):
        if v <= final_best:
            return i
    return None


def mean_species_count(species_history: List[List[Dict[str, Any]]]) -> float:
    if not species_history:
        return 0.0
    counts = [len(gen_info) for gen_info in species_history]
    return sum(counts) / len(counts)


def max_species_count(species_history: List[List[Dict[str, Any]]]) -> int:
    if not species_history:
        return 0
    return max(len(gen_info) for gen_info in species_history)


def extract_species_count_per_generation(
    species_history: List[List[Dict[str, Any]]]
) -> List[int]:
    return [len(gen_info) for gen_info in species_history]


def safe_round(x, n=6):
    if x is None:
        return None
    return round(x, n)


def compute_gap(instance_name: str, fitness: float):
    bks = BKS.get(instance_name)
    if bks is None or bks <= 0:
        return None
    return ((fitness - bks) / bks) * 100.0


def write_csv(rows: List[Dict[str, Any]], output_path: Path):
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = set()
    for row in rows:
        fieldnames.update(row.keys())
    fieldnames = list(fieldnames)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# =========================================================
# Core experiment logic
# =========================================================
def run_one(mode: str, instance_path: str, seed: int, verbose: bool = False):
    problem = VRP(instance_path)
    algo = build_algo(mode=mode, problem=problem, seed=seed)

    start = time.perf_counter()
    best = algo.run(verbose=verbose, verbose_species=False)
    elapsed = time.perf_counter() - start

    errors = problem.validate_solution(best.chromosome)
    feasible = len(errors) == 0

    instance_name = Path(instance_path).name
    gap = compute_gap(instance_name, best.fitness)

    result = {
        "algorithm": mode,
        "instance": instance_name,
        "instance_path": instance_path,
        "seed": seed,
        "best_fitness": safe_round(best.fitness, 6),
        "best_gap_percent": safe_round(gap, 6),
        "feasible": feasible,
        "runtime_seconds": safe_round(elapsed, 6),
        "convergence_generation": first_hit_generation(algo.history),
        "species_count_mean": safe_round(mean_species_count(algo.species_history), 4),
        "species_count_max": max_species_count(algo.species_history),
        "best_chromosome": " ".join(map(str, best.chromosome)),
        "history_length": len(algo.history),
        "final_population_best": safe_round(algo.population.best().fitness, 6),
        "history_json": json.dumps(algo.history),
        "species_count_json": json.dumps(
            extract_species_count_per_generation(algo.species_history)
        ),
    }
    return result, best, algo, problem


def build_convergence_rows(
    mode: str,
    instance_path: str,
    seed: int,
    history: List[float],
) -> List[Dict[str, Any]]:
    instance_name = Path(instance_path).name
    rows = []
    running_best = None

    for gen_idx, value in enumerate(history, start=1):
        if running_best is None:
            running_best = value
        else:
            running_best = min(running_best, value)

        rows.append({
            "algorithm": mode,
            "instance": instance_name,
            "instance_path": instance_path,
            "seed": seed,
            "generation": gen_idx,
            "best_fitness_at_generation": safe_round(value, 6),
            "running_best_fitness": safe_round(running_best, 6),
        })
    return rows


def build_species_rows(
    mode: str,
    instance_path: str,
    seed: int,
    species_history: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    instance_name = Path(instance_path).name
    rows = []

    for gen_idx, gen_info in enumerate(species_history, start=1):
        rows.append({
            "algorithm": mode,
            "instance": instance_name,
            "instance_path": instance_path,
            "seed": seed,
            "generation": gen_idx,
            "species_count": len(gen_info),
        })

        for item in gen_info:
            rows.append({
                "algorithm": mode,
                "instance": instance_name,
                "instance_path": instance_path,
                "seed": seed,
                "generation": gen_idx,
                "species_count": len(gen_info),
                "species_id": item.get("species_id"),
                "species_size": item.get("size"),
                "species_best": item.get("best"),
                "species_worst": item.get("worst"),
                "species_quota": item.get("quota"),
            })

    return rows


def summarize_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for row in results:
        key = (row["algorithm"], row["instance"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (algorithm, instance), rows in grouped.items():
        fitness_values = [float(r["best_fitness"]) for r in rows]
        runtime_values = [float(r["runtime_seconds"]) for r in rows]
        feasible_count = sum(1 for r in rows if str(r["feasible"]).lower() == "true")

        convergence_values = [
            int(r["convergence_generation"])
            for r in rows
            if r["convergence_generation"] is not None
        ]
        species_mean_values = [float(r["species_count_mean"]) for r in rows]
        species_max_values = [int(r["species_count_max"]) for r in rows]

        gap_values = [
            float(r["best_gap_percent"])
            for r in rows
            if r["best_gap_percent"] is not None
        ]

        summary_rows.append({
            "algorithm": algorithm,
            "instance": instance,
            "runs": len(rows),
            "feasible_runs": feasible_count,
            "best": safe_round(min(fitness_values), 4),
            "worst": safe_round(max(fitness_values), 4),
            "mean": safe_round(statistics.mean(fitness_values), 4),
            "std": safe_round(statistics.pstdev(fitness_values), 4) if len(fitness_values) > 1 else 0.0,
            "best_gap_percent": safe_round(min(gap_values), 4) if gap_values else None,
            "mean_gap_percent": safe_round(statistics.mean(gap_values), 4) if gap_values else None,
            "mean_runtime_seconds": safe_round(statistics.mean(runtime_values), 4),
            "mean_convergence_generation": safe_round(statistics.mean(convergence_values), 4) if convergence_values else None,
            "mean_species_count": safe_round(statistics.mean(species_mean_values), 4),
            "max_species_count": max(species_max_values) if species_max_values else 0,
        })

    summary_rows.sort(
        key=lambda x: (x["instance"], ALGO_ORDER.get(x["algorithm"], 999))
    )
    return summary_rows


def build_report_summary(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    report_rows = []
    for row in summary_rows:
        report_rows.append({
            "instance": row["instance"],
            "algorithm": row["algorithm"],
            "best": row["best"],
            "mean": row["mean"],
            "std": row["std"],
            "worst": row["worst"],
            "mean_runtime_seconds": row["mean_runtime_seconds"],
            "mean_convergence_generation": row["mean_convergence_generation"],
            "mean_species_count": row["mean_species_count"],
            "max_species_count": row["max_species_count"],
        })
    return report_rows


def print_single_result(result, best, algo, problem):
    print("\n===== Experiment Result =====")
    print("Algorithm:", result["algorithm"])
    print("Instance:", result["instance"])
    print("Seed:", result["seed"])
    print("Best fitness:", result["best_fitness"])
    print("Best gap (%):", result["best_gap_percent"])
    print("Feasible:", result["feasible"])
    print("Runtime (s):", result["runtime_seconds"])
    print("Convergence generation:", result["convergence_generation"])
    print("Mean species count:", result["species_count_mean"])
    print("Max species count:", result["species_count_max"])

    print("\nBest chromosome:")
    print(best.chromosome)

    print("\nValidation:")
    errors = problem.validate_solution(best.chromosome)
    if not errors:
        print("Solution is valid.")
    else:
        for e in errors:
            print(" -", e)

    print("\nBest routes:")
    for line in problem.pretty_routes(best.chromosome):
        print(line)

    print("\nHistory (last 10 generations):")
    tail = algo.history[-10:] if len(algo.history) >= 10 else algo.history
    print(tail)

    if algo.species_history:
        print("\nSpecies count (last 10 generations):")
        species_tail = extract_species_count_per_generation(algo.species_history)[-10:]
        print(species_tail)


def print_summary(summary_rows: List[Dict[str, Any]]):
    print("\n===== Summary =====")
    for row in summary_rows:
        print(
            f"{row['algorithm']:>6} | "
            f"{row['instance']} | "
            f"runs={row['runs']}, "
            f"feasible={row['feasible_runs']}, "
            f"best={row['best']}, "
            f"mean={row['mean']}, "
            f"std={row['std']}, "
            f"worst={row['worst']}, "
            f"runtime={row['mean_runtime_seconds']}s, "
            f"conv_gen={row['mean_convergence_generation']}, "
            f"mean_species={row['mean_species_count']}, "
            f"max_species={row['max_species_count']}"
        )


def run_batch():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_convergence_rows = []
    all_species_rows = []

    config_rows = []
    for mode in BATCH_MODES:
        cfg = get_mode_config(mode)
        cfg_row = {"algorithm": mode}
        cfg_row.update(cfg)
        config_rows.append(cfg_row)
    write_csv(config_rows, MODE_CONFIG_CSV)

    total_jobs = len(INSTANCE_PATHS) * len(BATCH_MODES) * len(BATCH_SEEDS)
    job_idx = 0

    for instance_path in INSTANCE_PATHS:
        for mode in BATCH_MODES:
            for seed in BATCH_SEEDS:
                job_idx += 1
                print(
                    f"[{job_idx:03d}/{total_jobs:03d}] "
                    f"Running instance={Path(instance_path).name} | "
                    f"mode={mode} | seed={seed}"
                )

                result, _, algo, _ = run_one(
                    mode=mode,
                    instance_path=instance_path,
                    seed=seed,
                    verbose=False,
                )

                all_results.append(result)

                all_convergence_rows.extend(
                    build_convergence_rows(
                        mode=mode,
                        instance_path=instance_path,
                        seed=seed,
                        history=algo.history,
                    )
                )

                all_species_rows.extend(
                    build_species_rows(
                        mode=mode,
                        instance_path=instance_path,
                        seed=seed,
                        species_history=algo.species_history,
                    )
                )

                print(
                    f"  -> best={result['best_fitness']}, "
                    f"feasible={result['feasible']}, "
                    f"runtime={result['runtime_seconds']}s, "
                    f"conv_gen={result['convergence_generation']}, "
                    f"species_mean={result['species_count_mean']}"
                )

    write_csv(all_results, RUN_RESULTS_CSV)

    summary_rows = summarize_results(all_results)
    write_csv(summary_rows, SUMMARY_CSV)

    report_rows = build_report_summary(summary_rows)
    write_csv(report_rows, REPORT_SUMMARY_CSV)

    write_csv(all_convergence_rows, CONVERGENCE_CSV)
    write_csv(all_species_rows, SPECIES_CSV)

    print_summary(summary_rows)

    print("\nSaved files:")
    print(" -", RUN_RESULTS_CSV)
    print(" -", SUMMARY_CSV)
    print(" -", REPORT_SUMMARY_CSV)
    print(" -", CONVERGENCE_CSV)
    print(" -", SPECIES_CSV)
    print(" -", MODE_CONFIG_CSV)


def run_single():
    result, best, algo, problem = run_one(
        mode=SINGLE_MODE,
        instance_path=SINGLE_INSTANCE_PATH,
        seed=SINGLE_SEED,
        verbose=True,
    )
    print_single_result(result, best, algo, problem)


def main():
    if RUN_STYLE == "single":
        run_single()
    elif RUN_STYLE == "batch":
        run_batch()
    else:
        raise ValueError("RUN_STYLE must be 'single' or 'batch'")


if __name__ == "__main__":
    main()