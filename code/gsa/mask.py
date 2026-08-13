"""Extract 'elite knowledge' from a set of elite individuals.

For VRP (a permutation problem) the key informative unit is an *edge*
between two customers, not the position index. Position-based masks as
used in binary-GA papers do not transfer directly — so here:

    edge_frequency   counts canonical edges (u,v) with u<v across elites
    dominant_chromosome  picks the elite whose edges sum to the highest
                         total frequency (most 'typical' elite)
    informative_positions  marks a position as informative if BOTH its
                           incident edges (in the dominant individual)
                           are high-frequency

This gives us:
    - a 'robust individual' (dominant chromosome)
    - an 'informative mask' derived from it
    - high-frequency edges for gene injection (Scenario 3)
"""

import numpy as np
from collections import Counter


def canonical_edge(a, b):
    a = int(a)
    b = int(b)
    return (a, b) if a < b else (b, a)


def perm_edges(perm):
    """Return the set of edges in the tour 0 -> perm[0] -> ... -> perm[-1] -> 0.

    Edges are canonical (undirected). Depot edges are included: a node
    being 'typically near the depot in good solutions' is useful info.
    """
    seq = [0] + [int(x) for x in perm] + [0]
    return {canonical_edge(seq[i], seq[i + 1]) for i in range(len(seq) - 1)}


def extract_edge_frequency(elites):
    """Count edges across elite individuals."""
    counter = Counter()
    for ind in elites:
        for e in perm_edges(ind.perm):
            counter[e] += 1
    return counter


def dominant_chromosome(elites):
    """Elite with the highest total edge-frequency score ≈ the most
    typical / robust structure in the elite pool."""
    if not elites:
        return None
    edge_freq = extract_edge_frequency(elites)
    best = elites[0]
    best_score = -1
    for ind in elites:
        score = sum(edge_freq[e] for e in perm_edges(ind.perm))
        if score > best_score:
            best_score = score
            best = ind
    return best


def informative_positions(dominant_ind, edge_freq, n_elites, thr=0.5):
    """Boolean mask over positions [0..len(perm)-1].

    A position p is informative if, in the dominant chromosome, both
    edges incident to the customer at position p appear in >= thr fraction
    of the elite pool. Informative positions are PROTECTED from random
    mutation (Scenario 2 paper intent).
    """
    perm = dominant_ind.perm
    L = len(perm)
    seq = [0] + [int(x) for x in perm] + [0]
    edges_seq = [canonical_edge(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]
    mask = np.zeros(L, dtype=bool)
    denom = max(n_elites, 1)
    for p in range(L):
        left = edge_freq.get(edges_seq[p], 0) / denom
        right = edge_freq.get(edges_seq[p + 1], 0) / denom
        if left >= thr and right >= thr:
            mask[p] = True
    return mask