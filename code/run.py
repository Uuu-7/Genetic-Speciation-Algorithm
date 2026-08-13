"""Benchmark script for real VRP instances with optional GSA statistics export."""

import json
import os
import time
import numpy as np

from gsa.problem import VRP
from gsa.population import random_population
from gsa.gea import run_gea
from gsa.gsa import run_gsa


# ---------- VRP loader ----------

def load_vrp_from_file(path):
    """Load a CVRP instance in Augerat-style .vrp format.

    Supported sections:
        CAPACITY
        NODE_COORD_SECTION
        DEMAND_SECTION
        DEPOT_SECTION

    Assumptions:
        - Node ids are 1-based in file
        - Depot is node 1
        - Output arrays are converted to 0-based indexing for this code:
            coords[0] = depot
            demands[0] = 0
            customers = 1..n
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"VRP file not found: {path}")

    capacity = None
    coords_dict = {}
    demands_dict = {}

    section = None

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            upper = line.upper()

            if upper.startswith("CAPACITY"):
                parts = line.replace(":", " ").split()
                capacity = int(parts[-1])
                continue

            if upper.startswith("NODE_COORD_SECTION"):
                section = "coords"
                continue

            if upper.startswith("DEMAND_SECTION"):
                section = "demands"
                continue

            if upper.startswith("DEPOT_SECTION"):
                section = "depot"
                continue

            if upper.startswith("EOF"):
                break

            if section == "coords":
                parts = line.split()
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    coords_dict[node_id] = (x, y)

            elif section == "demands":
                parts = line.split()
                if len(parts) >= 2:
                    node_id = int(parts[0])
                    demand = int(parts[1])
                    demands_dict[node_id] = demand

            elif section == "depot":
                continue

    if capacity is None:
        raise ValueError(f"Failed to read CAPACITY from {path}")
    if not coords_dict:
        raise ValueError(f"Failed to read NODE_COORD_SECTION from {path}")
    if not demands_dict:
        raise ValueError(f"Failed to read DEMAND_SECTION from {path}")

    node_ids = sorted(coords_dict.keys())
    expected = list(range(1, len(node_ids) + 1))
    if node_ids != expected:
        raise ValueError(
            f"Node ids must be contiguous starting from 1 in {path}. "
            f"Got: {node_ids[:10]}..."
        )

    coords = []
    demands = []
    for node_id in node_ids:
        if node_id not in demands_dict:
            raise ValueError(f"Missing demand for node {node_id} in {path}")
        coords.append(coords_dict[node_id])
        demands.append(demands_dict[node_id])

    coords = np.asarray(coords, dtype=float)
    demands = np.asarray(demands, dtype=float)

    return VRP(coords, demands, capacity)


# ---------- Algorithm wrappers ----------

def alg_ga(problem, pop, gens, rng):
    return run_gea(
        problem, pop, gens, rng,
        use_crossover=True,
        use_mutation=True,
        use_s1=False,
        use_s2=False,
        use_s3=False,
    )


def alg_gea(problem, pop, gens, rng):
    return run_gea(
        problem, pop, gens, rng,
        use_crossover=True,
        use_mutation=True,
        use_s1=True,
        use_s2=True,
        use_s3=True,
    )


def alg_gsa(problem, pop, gens, rng, record_stats=False):
    return run_gsa(problem, pop, gens, rng, record_stats=record_stats)


ALGORITHMS = [
    ("GA", alg_ga),
    ("GEA", alg_gea),
    ("GSA", alg_gsa),
]


# ---------- Experiment runner ----------

def summarize_instance(results, timings):
    summary = {}
    for name in results:
        arr = np.array(results[name], dtype=float)
        t_arr = np.array(timings[name], dtype=float)
        summary[name] = {
            "best": float(arr.min()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "avg_s": float(t_arr.mean()),
        }
    return summary


def run_instance(instance_path, n_runs, generations, pop_size, base_seed):
    """Run all algorithms on one VRP instance."""
    problem = load_vrp_from_file(instance_path)

    results = {name: [] for name, _ in ALGORITHMS}
    timings = {name: [] for name, _ in ALGORITHMS}
    histories = {name: [] for name, _ in ALGORITHMS}
    gsa_species_stats = []

    instance_name = os.path.basename(instance_path)
    print(f"\n{'=' * 72}")
    print(f"Instance: {instance_name}")
    print(f"Customers: {problem.n}, Capacity: {problem.capacity}")
    print(f"Runs: {n_runs}, Generations: {generations}, Population: {pop_size}")
    print(f"{'=' * 72}")

    for run in range(n_runs):
        print(f"\n=== Run {run + 1}/{n_runs} ===")

        for name, fn in ALGORITHMS:
            rng = np.random.default_rng(
                base_seed + 1000 * (run + 1) + abs(hash((instance_name, name))) % 997
            )

            pop = random_population(problem, pop_size, rng)

            t0 = time.time()
            if name == "GSA":
                best, history, species_stats = fn(
                    problem, pop, generations, rng, record_stats=True
                )
                gsa_species_stats.append(species_stats)
            else:
                best, history = fn(problem, pop, generations, rng)
            elapsed = time.time() - t0

            assert problem.validate_solution(best.perm), \
                f"{name} returned invalid solution on {instance_name}"

            results[name].append(float(best.fitness))
            timings[name].append(float(elapsed))
            histories[name].append([float(x) for x in history])

            print(f"  {name:6s}: {best.fitness:10.2f}  ({elapsed:6.2f}s)")

    print(f"\n--- Summary: {instance_name} ---")
    print(f"{'algo':<8} {'best':>10} {'mean':>10} {'std':>10} {'avg_s':>10}")
    summary = summarize_instance(results, timings)
    for name in results:
        row = summary[name]
        print(
            f"{name:<8} "
            f"{row['best']:10.2f} "
            f"{row['mean']:10.2f} "
            f"{row['std']:10.2f} "
            f"{row['avg_s']:10.2f}"
        )

    return {
        "meta": {
            "instance": instance_name,
            "customers": int(problem.n),
            "capacity": float(problem.capacity),
            "n_runs": int(n_runs),
            "generations": int(generations),
            "population": int(pop_size),
        },
        "results": results,
        "timings": timings,
        "histories": histories,
        "summary": summary,
        "gsa_species_stats": gsa_species_stats,
    }


def save_results(all_results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to: {save_path}")


def main():
    instances = [
        "data/vrp/A-n32-k5.vrp",
        "data/vrp/A-n60-k9.vrp",
        "data/vrp/A-n80-k10.vrp",
    ]

    n_runs = 5
    generations = 500
    pop_size = 200
    base_seed = 42

    all_results = {}

    for instance_path in instances:
        instance_result = run_instance(
            instance_path=instance_path,
            n_runs=n_runs,
            generations=generations,
            pop_size=pop_size,
            base_seed=base_seed,
        )
        all_results[os.path.basename(instance_path)] = instance_result

    print(f"\n{'=' * 72}")
    print("All experiments finished.")
    print(f"{'=' * 72}")

    save_results(all_results, "results/experiment_results.json")
    return all_results


if __name__ == "__main__":
    main()
