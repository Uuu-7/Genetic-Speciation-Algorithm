from collections import Counter
from typing import List, Tuple, Set


def _infer_depot(problem):
    if hasattr(problem, "depot"):
        return problem.depot
    return 1


def _canonical_edge(a, b):
    return (a, b) if a < b else (b, a)


def build_position_mask(
    elite_individuals,
    chromosome_length: int,
    threshold: float = 0.6,
) -> Tuple[List[int], List[int]]:
    if not elite_individuals or chromosome_length <= 0:
        return [0] * max(0, chromosome_length), []

    valid_elites = [ind for ind in elite_individuals if len(ind.chromosome) >= chromosome_length]
    if not valid_elites:
        min_len = min(len(ind.chromosome) for ind in elite_individuals)
        return [0] * min_len, []

    mask = [0] * chromosome_length
    dominant = [None] * chromosome_length

    for pos in range(chromosome_length):
        genes = [ind.chromosome[pos] for ind in valid_elites]
        counter = Counter(genes)
        gene, count = counter.most_common(1)[0]
        ratio = count / max(1, len(valid_elites))

        if ratio >= threshold:
            mask[pos] = 1
            dominant[pos] = gene

    return mask, dominant


def build_edge_frequency_mask(
    elite_individuals,
    problem,
    threshold: float = 0.6,
):
    if not elite_individuals:
        return set(), None

    lengths = [len(ind.chromosome) for ind in elite_individuals]
    if not lengths:
        return set(), None

    length_counter = Counter(lengths)
    target_len = length_counter.most_common(1)[0][0]
    valid_elites = [ind for ind in elite_individuals if len(ind.chromosome) == target_len]

    if not valid_elites:
        return set(), elite_individuals[0]

    depot = _infer_depot(problem)
    edge_counter = Counter()
    individual_edges = []

    for ind in valid_elites:
        chrom = ind.chromosome
        full = [depot] + list(chrom) + [depot]
        edges = set()

        for a, b in zip(full[:-1], full[1:]):
            e = _canonical_edge(a, b)
            edges.add(e)
            edge_counter[e] += 1

        individual_edges.append(edges)

    min_count = max(1, int(round(len(valid_elites) * threshold)))
    informative_edges = {e for e, c in edge_counter.items() if c >= min_count}

    if not informative_edges:
        return set(), valid_elites[0]

    best_idx = 0
    best_score = -1
    for i, edges in enumerate(individual_edges):
        score = len(edges & informative_edges)
        if score > best_score:
            best_score = score
            best_idx = i

    robust_individual = valid_elites[best_idx]
    return informative_edges, robust_individual


def build_position_informative_mask_from_robust(
    elite_individuals,
    robust_individual,
    threshold: float = 0.6,
):
    if robust_individual is None:
        return []

    robust_len = len(robust_individual.chromosome)
    if robust_len == 0:
        return []

    valid_elites = [ind for ind in elite_individuals if len(ind.chromosome) == robust_len]

    if not valid_elites:
        return [0] * robust_len

    mask = [0] * robust_len

    for pos in range(robust_len):
        gene = robust_individual.chromosome[pos]
        cnt = sum(1 for ind in valid_elites if ind.chromosome[pos] == gene)
        if cnt / max(1, len(valid_elites)) >= threshold:
            mask[pos] = 1

    return mask


def build_adjacency_pair_mask(
    elite_individuals,
    problem,
    threshold: float = 0.6,
) -> Set[Tuple[int, int]]:
    if not elite_individuals:
        return set()

    lengths = [len(ind.chromosome) for ind in elite_individuals]
    if not lengths:
        return set()

    length_counter = Counter(lengths)
    target_len = length_counter.most_common(1)[0][0]
    valid_elites = [ind for ind in elite_individuals if len(ind.chromosome) == target_len]

    if not valid_elites:
        return set()

    depot = _infer_depot(problem)
    pair_counter = Counter()

    for ind in valid_elites:
        full = [depot] + list(ind.chromosome) + [depot]
        seen_pairs = set()

        for a, b in zip(full[:-1], full[1:]):
            pair = _canonical_edge(a, b)
            seen_pairs.add(pair)

        for pair in seen_pairs:
            pair_counter[pair] += 1

    min_count = max(1, int(round(len(valid_elites) * threshold)))
    informative_pairs = {p for p, c in pair_counter.items() if c >= min_count}

    return informative_pairs