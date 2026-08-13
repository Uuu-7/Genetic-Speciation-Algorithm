"""Genetic operators for permutation-encoded VRP.

This module contains the *strengthened* versions of the operators that
matter for reproducing the paper:

- order_crossover          standard OX
- swap_mutation            light random perturbation (baseline GA)
- directed_mutation        Scenario 2: 2-opt + Or-opt local search
                           restricted to non-informative positions and
                           biased toward high-cost edges
- edge_injection           Scenario 3: for each high-freq donor edge
                           not present in target, relocate the endpoint
                           so the edge is realized (or-opt size 1)
"""

import numpy as np
from .mask import canonical_edge, perm_edges


# ---------- Crossover ----------

def order_crossover(p1, p2, rng):
    """Order Crossover (OX) for permutations."""
    n = len(p1)
    a, b = sorted(rng.choice(n, size=2, replace=False))
    child = -np.ones(n, dtype=np.int64)
    child[a:b + 1] = p1[a:b + 1]
    chosen = set(p1[a:b + 1].tolist())
    idx = (b + 1) % n
    for i in range(n):
        gene = int(p2[(b + 1 + i) % n])
        if gene not in chosen:
            child[idx] = gene
            idx = (idx + 1) % n
    return child


# ---------- Basic mutation (baseline GA) ----------

def swap_mutation(perm, rng, rate=0.1):
    """Randomly swap a few pairs. Used by baseline GA."""
    perm = perm.copy()
    n = len(perm)
    n_swaps = max(1, int(n * rate))
    for _ in range(n_swaps):
        i, j = rng.choice(n, size=2, replace=False)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


# ---------- Scenario 2: directed mutation (the key upgrade) ----------

def directed_mutation(perm, mask, problem, rng,
                      max_2opt_tries=30, max_oropt_tries=20):
    """2-opt + Or-opt local search restricted to non-informative positions.

    Key ideas (what makes this work where random swaps fail):
    1. Only touch positions where mask==False (i.e., positions whose
       incident edges in the dominant chromosome are NOT high-frequency).
       Informative positions are frozen.
    2. For 2-opt, pick pairs of free positions, reverse the segment,
       accept if total cost decreases (first improvement style).
    3. For Or-opt, relocate a short run (1..3) from a free source to a
       free destination, accept if improving.

    Returns a new permutation (np.int64 array).
    """
    perm = perm.copy()
    n = len(perm)

    free = np.where(~mask)[0]
    if len(free) < 2:
        # everything locked — fall back to a tiny random swap so the
        # caller still gets a different individual
        i, j = rng.choice(n, size=2, replace=False)
        perm[i], perm[j] = perm[j], perm[i]
        return perm

    best_cost = problem.fast_evaluate(perm)

    # ----- 2-opt phase -----
    # Bias the choice of the first index toward "currently expensive" edges.
    seq = np.concatenate(([0], perm, [0]))
    edge_lengths = np.array(
        [problem.dist[seq[i], seq[i + 1]] for i in range(len(seq) - 1)],
        dtype=float,
    )
    # For each position p in perm (0..n-1), the incident edges are
    # edge_lengths[p] and edge_lengths[p+1]. Use max of the two as a priority.
    pos_priority = np.maximum(edge_lengths[:-1], edge_lengths[1:])
    # Restrict to free positions only.
    free_priority = pos_priority[free]
    # Convert to a probability (softmax on normalized lengths).
    if free_priority.sum() > 0:
        probs = free_priority / free_priority.sum()
    else:
        probs = None

    for _ in range(max_2opt_tries):
        if len(free) < 2:
            break
        if probs is not None:
            i_idx = rng.choice(len(free), p=probs)
        else:
            i_idx = rng.integers(len(free))
        j_idx = rng.integers(len(free))
        if i_idx == j_idx:
            continue
        i, j = sorted((int(free[i_idx]), int(free[j_idx])))
        if j - i < 1:
            continue
        new_perm = perm.copy()
        new_perm[i:j + 1] = new_perm[i:j + 1][::-1]
        new_cost = problem.fast_evaluate(new_perm)
        if new_cost < best_cost:
            best_cost = new_cost
            perm = new_perm

    # ----- Or-opt phase -----
    for _ in range(max_oropt_tries):
        if len(free) < 2:
            break
        seg_len = int(rng.integers(1, 4))  # 1, 2, or 3
        src = int(rng.choice(free))
        if src + seg_len > n:
            continue
        # Require every position in the segment to be free
        if any(mask[src + k] for k in range(seg_len)):
            continue
        # Choose destination among free positions
        dst_candidates = [int(x) for x in free if (x < src or x >= src + seg_len)]
        if not dst_candidates:
            continue
        dst = dst_candidates[rng.integers(len(dst_candidates))]

        segment = perm[src:src + seg_len].copy()
        rest = np.concatenate([perm[:src], perm[src + seg_len:]])
        insert_at = dst - seg_len if dst > src else dst
        if insert_at < 0 or insert_at > len(rest):
            continue
        new_perm = np.concatenate([rest[:insert_at], segment, rest[insert_at:]])
        if len(new_perm) != n:
            continue
        new_cost = problem.fast_evaluate(new_perm)
        if new_cost < best_cost:
            best_cost = new_cost
            perm = new_perm

    return perm


# ---------- Scenario 3: edge injection ----------

def edge_injection(target_perm, donor_perm, edge_freq, threshold,
                   n_elites, rng, max_injects=5):
    """Inject donor's high-frequency edges into target.

    Strategy: for each candidate edge (a, b) that appears in the donor
    and in >= `threshold` fraction of the elite pool, but NOT in the
    target, pluck `b` out of the target and re-insert it next to `a`.
    This is a size-1 Or-opt that directly creates the desired edge.

    We skip edges that touch the depot (they're never represented in the
    permutation as adjacency — only at the beginning / end).
    """
    target = list(int(x) for x in target_perm)

    donor_edges = perm_edges(donor_perm)
    target_edges = perm_edges(target_perm)

    candidates = []
    denom = max(n_elites, 1)
    for e in donor_edges:
        if e in target_edges:
            continue
        a, b = e
        if a == 0 or b == 0:
            continue
        freq = edge_freq.get(e, 0) / denom
        if freq >= threshold:
            candidates.append((freq, e))
    # Higher frequency first
    candidates.sort(reverse=True)

    injected = 0
    for _, (a, b) in candidates:
        if injected >= max_injects:
            break
        try:
            i = target.index(a)
            j = target.index(b)
        except ValueError:
            continue
        if abs(i - j) == 1:
            continue  # already adjacent, skip

        # Remove b, re-insert right after a (or before, 50/50).
        target.pop(j)
        new_i = target.index(a)
        insert_before = rng.random() < 0.5
        if insert_before:
            target.insert(new_i, b)
        else:
            target.insert(new_i + 1, b)
        injected += 1

    return np.asarray(target, dtype=np.int64)