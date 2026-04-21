"""Selection operators."""

import numpy as np


def tournament_selection(pop, k, rng):
    """Pick k random individuals and return the best."""
    n = len(pop.inds)
    k = min(k, n)
    idxs = rng.choice(n, size=k, replace=False)
    best_i = idxs[0]
    best_f = pop.inds[best_i].fitness
    for i in idxs[1:]:
        if pop.inds[i].fitness < best_f:
            best_f = pop.inds[i].fitness
            best_i = i
    return pop.inds[best_i]


def roulette_wheel_selection(inds, rng):
    """Fitness-proportional selection (minimization).

    Used for Scenarios 2 and 3 in the paper.
    """
    n = len(inds)
    if n == 1:
        return inds[0]
    fits = np.asarray([ind.fitness for ind in inds], dtype=float)
    max_f = fits.max()
    weights = (max_f - fits) + 1e-9  # larger weight for smaller (better) fitness
    total = weights.sum()
    if total <= 0:
        return inds[rng.integers(n)]
    probs = weights / total
    idx = rng.choice(n, p=probs)
    return inds[idx]