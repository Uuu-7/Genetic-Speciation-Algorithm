"""GEA main loop with configurable operator ratios.

Every generation does, in order:
    (1) standard OX crossover
    (2) standard swap mutation
    (3) Scenario 1  — dominant chromosome crossover
    (4) Scenario 2  — directed mutation
    (5) Scenario 3  — edge injection

Crossover : Mutation ratio changed to 8:2 (0.40 : 0.10).
S1 : S2 : S3 = 0.5 : 0.3 : 0.2
"""

import numpy as np

from .individual import Individual
from .selection import tournament_selection, roulette_wheel_selection
from .operators import (
    order_crossover,
    swap_mutation,
    directed_mutation,
    edge_injection,
)
from .mask import (
    extract_edge_frequency,
    dominant_chromosome,
    informative_positions,
)


def _try_replace_worst(pop, child):
    wi = pop.worst_index()
    if child.fitness < pop.inds[wi].fitness:
        pop.inds[wi] = child
        return True
    return False


def run_gea(
    problem,
    pop,
    generations,
    rng,
    elite_ratio=0.2,
    info_thr=0.5,
    inject_thr=0.4,
    use_crossover=True,
    use_mutation=True,
    use_s1=True,
    use_s2=True,
    use_s3=True,
    crossover_ratio=0.40,   # 8:2 比例（原 0.25）
    mutation_ratio=0.10,    # 8:2 比例（原 0.25）
    s1_ratio=0.5,
    s2_ratio=0.3,
    s3_ratio=0.2,
    diversity_reset_ratio=0.05,
    verbose=False,
):
    """Run GEA and return (best_individual, fitness_history)."""
    pop.evaluate_all(problem)
    pop.sort()
    best_history = [pop.best().fitness]

    N = len(pop.inds)

    for gen in range(generations):
        pop.sort()
        n_elite = max(2, int(N * elite_ratio))
        elites = pop.inds[:n_elite]
        edge_freq = extract_edge_frequency(elites)
        dom = dominant_chromosome(elites)
        mask = informative_positions(dom, edge_freq, n_elite, thr=info_thr)

        n_cross = max(1, int(N * crossover_ratio)) if use_crossover else 0
        n_mut   = max(1, int(N * mutation_ratio))  if use_mutation  else 0
        n_s1    = max(1, int(N * s1_ratio)) if use_s1 and dom is not None else 0
        n_s2    = max(1, int(N * s2_ratio)) if use_s2 else 0
        n_s3    = max(1, int(N * s3_ratio)) if use_s3 else 0

        # (1) Crossover
        for _ in range(n_cross):
            p1 = tournament_selection(pop, 3, rng)
            p2 = tournament_selection(pop, 3, rng)
            c_perm = order_crossover(p1.perm, p2.perm, rng)
            child = Individual(c_perm)
            child.fitness = problem.fast_evaluate(c_perm)
            _try_replace_worst(pop, child)

        # (2) Mutation
        for _ in range(n_mut):
            p = tournament_selection(pop, 3, rng)
            c_perm = swap_mutation(p.perm, rng)
            child = Individual(c_perm)
            child.fitness = problem.fast_evaluate(c_perm)
            _try_replace_worst(pop, child)

        # (3) Scenario 1 — dominant chromosome crossover
        for _ in range(n_s1):
            partner = tournament_selection(pop, 3, rng)
            c_perm = order_crossover(dom.perm, partner.perm, rng)
            child = Individual(c_perm)
            child.fitness = problem.fast_evaluate(c_perm)
            _try_replace_worst(pop, child)

        # (4) Scenario 2 — directed mutation
        for _ in range(n_s2):
            parent = roulette_wheel_selection(pop.inds, rng)
            c_perm = directed_mutation(parent.perm, mask, problem, rng)
            child = Individual(c_perm)
            child.fitness = problem.fast_evaluate(c_perm)
            _try_replace_worst(pop, child)

        # (5) Scenario 3 — edge injection
        if n_s3 > 0:
            pop.sort()
            donors = pop.inds[:n_elite]
            tail = pop.inds[N // 2:]
            if len(tail) == 0:
                tail = pop.inds

            for _ in range(n_s3):
                donor = donors[rng.integers(len(donors))]
                target = tail[rng.integers(len(tail))]
                c_perm = edge_injection(
                    target.perm, donor.perm,
                    edge_freq, inject_thr, n_elite, rng,
                )
                child = Individual(c_perm)
                child.fitness = problem.fast_evaluate(c_perm)
                _try_replace_worst(pop, child)

        # Diversity injection
        pop.sort()
        n_reset = max(1, int(N * diversity_reset_ratio))
        for k in range(n_reset):
            idx = N - 1 - k
            perm = np.arange(1, problem.n + 1, dtype=np.int64)
            rng.shuffle(perm)
            new_ind = Individual(perm)
            new_ind.fitness = problem.fast_evaluate(perm)
            if new_ind.fitness < pop.inds[idx].fitness:
                pop.inds[idx] = new_ind

        pop.sort()
        best_history.append(pop.best().fitness)

        if verbose and (gen % 50 == 0 or gen == generations - 1):
            print(f"  [GEA] gen {gen}: best={pop.best().fitness:.2f}")

    return pop.best(), best_history