import random
from typing import List, Tuple, Optional, Set

from .individual import Individual


def _infer_depot(problem):
    if hasattr(problem, "depot"):
        return problem.depot
    return 1


def _canonical_edge(a, b):
    return (a, b) if a < b else (b, a)


def order_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
    p1 = parent1.chromosome
    p2 = parent2.chromosome
    n = len(p1)

    if n < 2:
        return parent1.copy(), parent2.copy()

    i, j = sorted(random.sample(range(n), 2))

    c1 = [None] * n
    c2 = [None] * n

    c1[i:j + 1] = p1[i:j + 1]
    c2[i:j + 1] = p2[i:j + 1]

    fill1 = [x for x in p2 if x not in c1]
    fill2 = [x for x in p1 if x not in c2]

    idxs = list(range(j + 1, n)) + list(range(0, i))
    for idx, gene in zip(idxs, fill1):
        c1[idx] = gene
    for idx, gene in zip(idxs, fill2):
        c2[idx] = gene

    return Individual(c1), Individual(c2)


def swap_mutation(individual: Individual, mutation_rate: float = 0.2) -> Individual:
    child = individual.copy()
    n = len(child.chromosome)

    if n < 2 or random.random() >= mutation_rate:
        return child

    i, j = random.sample(range(n), 2)
    child.chromosome[i], child.chromosome[j] = child.chromosome[j], child.chromosome[i]
    return child


def insertion_mutation(individual: Individual, mutation_rate: float = 0.2) -> Individual:
    child = individual.copy()
    n = len(child.chromosome)

    if n < 2 or random.random() >= mutation_rate:
        return child

    i, j = random.sample(range(n), 2)
    gene = child.chromosome.pop(i)
    child.chromosome.insert(j, gene)
    return child


def inversion_mutation(individual: Individual, mutation_rate: float = 0.2) -> Individual:
    child = individual.copy()
    n = len(child.chromosome)

    if n < 2 or random.random() >= mutation_rate:
        return child

    i, j = sorted(random.sample(range(n), 2))
    child.chromosome[i:j + 1] = reversed(child.chromosome[i:j + 1])
    return child


def robust_order_crossover(
    robust_parent: Individual,
    other_parent: Individual,
) -> Individual:
    child, _ = order_crossover(robust_parent, other_parent)
    return child


def _worst_edge_position(problem, chromosome: List[int]) -> Optional[int]:
    n = len(chromosome)
    if n < 2:
        return None

    depot = _infer_depot(problem)
    worst_pos = None
    worst_cost = -1.0

    for i in range(n):
        left = depot if i == 0 else chromosome[i - 1]
        cur = chromosome[i]
        right = depot if i == n - 1 else chromosome[i + 1]

        c = problem.distance_matrix[left][cur] + problem.distance_matrix[cur][right]
        if c > worst_cost:
            worst_cost = c
            worst_pos = i

    return worst_pos


def directed_swap_mutation(
    individual: Individual,
    problem,
    informative_mask: Optional[List[int]] = None,
    mutation_rate: float = 0.2,
) -> Individual:
    child = individual.copy()
    n = len(child.chromosome)

    if n < 2 or random.random() >= mutation_rate:
        return child

    protected = set()
    if informative_mask is not None:
        protected = {i for i, v in enumerate(informative_mask) if v == 1}

    target = _worst_edge_position(problem, child.chromosome)
    if target is None:
        child.fitness = problem.evaluate(child.chromosome)
        return child

    candidate_positions = [i for i in range(n) if i not in protected and i != target]

    if target in protected or not candidate_positions:
        child.fitness = problem.evaluate(child.chromosome)
        return child

    best = child.copy()
    best.fitness = problem.evaluate(best.chromosome)

    for _ in range(min(8, len(candidate_positions))):
        j = random.choice(candidate_positions)
        trial = child.copy()
        trial.chromosome[target], trial.chromosome[j] = trial.chromosome[j], trial.chromosome[target]
        trial.fitness = problem.evaluate(trial.chromosome)
        if trial.fitness < best.fitness:
            best = trial

    for _ in range(min(8, len(candidate_positions))):
        j = random.choice(candidate_positions)
        trial = child.copy()
        gene = trial.chromosome.pop(target)
        trial.chromosome.insert(j, gene)
        trial.fitness = problem.evaluate(trial.chromosome)
        if trial.fitness < best.fitness:
            best = trial

    return best


def inject_segment(
    target: Individual,
    donor: Individual,
    problem,
    segment_length: Optional[int] = None,
) -> Individual:
    t = target.copy()
    d = donor.chromosome
    n = len(d)

    if n < 3:
        t.fitness = problem.evaluate(t.chromosome)
        return t

    if segment_length is None:
        segment_length = max(2, n // 6)

    segment_length = min(segment_length, n - 1)
    start = random.randint(0, n - segment_length)
    seg = d[start:start + segment_length]
    seg_set = set(seg)

    rest = [x for x in t.chromosome if x not in seg_set]
    insert_pos = min(start, len(rest))
    new_chrom = rest[:insert_pos] + seg + rest[insert_pos:]

    t.chromosome = new_chrom
    t.fitness = problem.evaluate(t.chromosome)
    return t


def inject_by_mask(
    target: Individual,
    donor: Individual,
    informative_mask,
    problem,
) -> Individual:
    t = target.copy()
    n = len(t.chromosome)

    if informative_mask is None or len(informative_mask) != n:
        t.fitness = problem.evaluate(t.chromosome)
        return t

    protected_positions = [i for i, v in enumerate(informative_mask) if v == 1]
    if not protected_positions:
        t.fitness = problem.evaluate(t.chromosome)
        return t

    new_chrom = t.chromosome[:]
    used = set()

    for pos in protected_positions:
        gene = donor.chromosome[pos]
        new_chrom[pos] = gene
        used.add(gene)

    remaining = [g for g in t.chromosome if g not in used]
    rem_idx = 0

    for i in range(n):
        if i in protected_positions:
            continue
        if rem_idx < len(remaining):
            new_chrom[i] = remaining[rem_idx]
            rem_idx += 1

    t.chromosome = new_chrom
    t.fitness = problem.evaluate(t.chromosome)
    return t


def inject_edges_from_elite_to_non_elite(
    target: Individual,
    donor: Individual,
    informative_pairs: Set[Tuple[int, int]],
    problem,
    max_injections: Optional[int] = None,
) -> Individual:
    t = target.copy()
    chrom = t.chromosome[:]
    n = len(chrom)

    if n < 2 or not informative_pairs:
        t.fitness = problem.evaluate(t.chromosome)
        return t

    donor_chrom = donor.chromosome[:]
    depot = _infer_depot(problem)

    donor_full = [depot] + donor_chrom + [depot]
    donor_pairs_in_order = []
    for a, b in zip(donor_full[:-1], donor_full[1:]):
        pair = _canonical_edge(a, b)
        if pair in informative_pairs:
            donor_pairs_in_order.append((a, b))

    if not donor_pairs_in_order:
        t.fitness = problem.evaluate(t.chromosome)
        return t

    if max_injections is None:
        max_injections = max(1, len(donor_pairs_in_order) // 2)

    applied = 0

    for a, b in donor_pairs_in_order:
        if applied >= max_injections:
            break

        if a == depot or b == depot:
            continue

        if a not in chrom or b not in chrom:
            continue

        ia = chrom.index(a)
        ib = chrom.index(b)

        if abs(ia - ib) == 1:
            applied += 1
            continue

        gene_b = chrom.pop(ib)

        ia = chrom.index(a)
        insert_pos = ia + 1
        chrom.insert(insert_pos, gene_b)

        applied += 1

    t.chromosome = chrom
    t.fitness = problem.evaluate(t.chromosome)
    return t