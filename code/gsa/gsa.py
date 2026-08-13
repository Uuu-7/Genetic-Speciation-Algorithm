"""GSA — 动态分群版本（边集缓存优化）。

性能优化要点：
    species_pairs 从 [global_idx, Individual] 二元组
    改为 [global_idx, Individual, edge_set] 三元组。

    - wishart_cluster 返回 (clusters, edge_sets)，直接填入三元组
    - _species_avg_jaccard 直接读缓存的 edge_set，不再调用 perm_edges
    - _novelty_ok / _crowding_replace 也读缓存，只对新子代计算一次
    - 个体被替换时同步更新 edge_set（三元组第三个元素）

    效果：gsa.py 内部的 perm_edges 调用从 O(N²/代) 降至 O(新子代数/代)
"""

import numpy as np

from .individual import Individual
from .selection import roulette_wheel_selection
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
    perm_edges,
)
from .speciator import wishart_cluster


# ─────────────────────────────── helpers ────────────────────────────────

def _jaccard(a: set, b: set) -> float:
    union = len(a | b)
    return 0.0 if union == 0 else 1.0 - len(a & b) / union


def _species_avg_jaccard(triples: list) -> float:
    """物种内平均 Jaccard 距离（直接使用缓存的 edge_set，第三个元素）。"""
    if len(triples) < 2:
        return 0.0
    total, cnt = 0.0, 0
    for i in range(len(triples)):
        for j in range(i + 1, len(triples)):
            total += _jaccard(triples[i][2], triples[j][2])
            cnt += 1
    return total / cnt if cnt else 0.0


def _novelty_ok(child_es: set, triples: list, sim_thr: float) -> bool:
    """新颖性检查（使用缓存 edge_set，不调用 perm_edges）。"""
    if not triples:
        return True
    sims = [1.0 - _jaccard(child_es, t[2]) for t in triples]
    return float(np.mean(sims)) < sim_thr


def _crowding_replace(triples: list, child: Individual, child_es: set,
                      novelty_thr: float) -> bool:
    """Deterministic crowding + 新颖性过滤。
    child_es 由调用方传入（每个子代只算一次 perm_edges）。
    替换时同步写入 triple[2]（edge_set 缓存）。
    """
    if not _novelty_ok(child_es, triples, novelty_thr):
        return False

    best_sim, best_pos = -1.0, -1
    for pos, (_, ind, es) in enumerate(triples):
        if ind.fitness <= child.fitness:
            continue
        sim = 1.0 - _jaccard(es, child_es)
        if sim > best_sim:
            best_sim, best_pos = sim, pos

    if best_pos == -1:
        return False
    triples[best_pos][1] = child
    triples[best_pos][2] = child_es      # ← 同步更新缓存
    return True


def _diversity_rescue(triples: list, problem, rng,
                      rescue_ratio: float = 0.30) -> None:
    """激进多样性救援（替换最差的 rescue_ratio，并更新 edge_set 缓存）。"""
    sp_size = len(triples)
    n_rescue = max(1, int(sp_size * rescue_ratio))
    triples.sort(key=lambda x: x[1].fitness)
    n = problem.n
    for k in range(sp_size - 1, sp_size - 1 - n_rescue, -1):
        if k < 0:
            break
        perm = np.arange(1, n + 1, dtype=np.int64)
        rng.shuffle(perm)
        new_ind = Individual(perm)
        new_ind.fitness = problem.fast_evaluate(perm)
        triples[k][1] = new_ind
        triples[k][2] = perm_edges(perm)  # ← 新个体的边集


# ────────────────────────────── main loop ───────────────────────────────

def run_gsa(
    problem,
    pop,
    generations: int,
    rng,
    elite_ratio: float = 0.4,
    info_thr: float = 0.55,
    inject_thr: float = 0.35,
    k_nn: int = 4,
    max_species: int = 5,
    recluster_every: int = 20,
    migrate_every: int = 30,
    burst_every: int = 80,
    crossover_ratio: float = 0.25,
    mutation_ratio: float = 0.25,
    s1_ratio: float = 0.5,
    s2_ratio: float = 0.3,
    s3_ratio: float = 0.2,
    diversity_reset_ratio: float = 0.03,
    diversity_collapse_thr: float = 0.04,
    novelty_thr: float = 0.92,
    max_size_ratio: float = 0.45,
    verbose: bool = False,
    record_stats: bool = False,
):
    pop.evaluate_all(problem)
    pop.sort()
    best_history = [pop.best().fitness]
    N = len(pop.inds)

    # species_triples: list of list of [global_idx, Individual, edge_set]
    species_triples: list = None

    species_stats = None
    if record_stats:
        species_stats = {
            "species_count": [],
            "species_sizes": [],
            "species_diversity": [],
        }

    for gen in range(generations):
        global_best = pop.best().copy()

        # ── 周期性多样性爆发 ─────────────────────────────────────
        if gen > 0 and gen % burst_every == 0:
            pop.sort()
            n_burst = max(max_species, int(N * 0.25))
            for k in range(n_burst):
                idx = N - 1 - k
                perm = np.arange(1, problem.n + 1, dtype=np.int64)
                rng.shuffle(perm)
                new_ind = Individual(perm)
                new_ind.fitness = problem.fast_evaluate(perm)
                pop.inds[idx] = new_ind
            species_triples = None  # 强制重分群

        # ── 重分群（wishart 返回 edge_sets 缓存）────────────────────
        if species_triples is None or gen % recluster_every == 0:
            raw_clusters, global_edge_sets = wishart_cluster(
                pop.inds,
                k=k_nn,
                max_species=max_species,
                min_size=3,
                alpha=0.20,
                max_size_ratio=max_size_ratio,
                rng=rng,
            )
            # 构建三元组：[全局索引, Individual, edge_set]
            species_triples = [
                [[idx, pop.inds[idx], global_edge_sets[idx]] for idx in cluster]
                for cluster in raw_clusters
            ]

        if record_stats:
            species_stats["species_count"].append(len(species_triples))
            species_stats["species_sizes"].append([len(sp) for sp in species_triples])
            species_stats["species_diversity"].append(
                [_species_avg_jaccard(sp) for sp in species_triples]
            )

        # ── 各物种独立进化 ───────────────────────────────────────
        for sp in species_triples:
            if len(sp) < 3:
                continue

            sp_size = len(sp)
            sp.sort(key=lambda x: x[1].fitness)

            # 物种多样性检测与救援
            if _species_avg_jaccard(sp) < diversity_collapse_thr:
                _diversity_rescue(sp, problem, rng, rescue_ratio=0.30)

            inds_only = [t[1] for t in sp]

            # 局部 elite 知识（使用 Individual 对象，不需要 edge_set）
            n_elite = max(2, int(sp_size * elite_ratio))
            elites_snap = inds_only[:n_elite]
            edge_freq = extract_edge_frequency(elites_snap)
            dom = dominant_chromosome(elites_snap)
            mask = informative_positions(dom, edge_freq, n_elite, thr=info_thr)

            n_cross = max(1, int(sp_size * crossover_ratio))
            n_mut   = max(1, int(sp_size * mutation_ratio))
            n_s1    = max(1, int(sp_size * s1_ratio)) if dom is not None else 0
            n_s2    = max(1, int(sp_size * s2_ratio))
            n_s3    = max(1, int(sp_size * s3_ratio))

            def _make_child(perm):
                """创建子代并计算边集（每个子代只调用一次 perm_edges）。"""
                c = Individual(perm)
                c.fitness = problem.fast_evaluate(perm)
                return c, perm_edges(perm)

            # (1) 交叉
            for _ in range(n_cross):
                p1 = inds_only[rng.integers(sp_size)]
                p2 = inds_only[rng.integers(sp_size)]
                child, ces = _make_child(order_crossover(p1.perm, p2.perm, rng))
                _crowding_replace(sp, child, ces, novelty_thr)

            # (2) 变异
            for _ in range(n_mut):
                p = inds_only[rng.integers(sp_size)]
                child, ces = _make_child(swap_mutation(p.perm, rng))
                _crowding_replace(sp, child, ces, novelty_thr)

            # (3) S1: 主导染色体交叉
            for _ in range(n_s1):
                partner = inds_only[rng.integers(sp_size)]
                child, ces = _make_child(order_crossover(dom.perm, partner.perm, rng))
                _crowding_replace(sp, child, ces, novelty_thr)

            # (4) S2: 定向变异
            for _ in range(n_s2):
                parent = roulette_wheel_selection(inds_only, rng)
                child, ces = _make_child(directed_mutation(parent.perm, mask, problem, rng))
                _crowding_replace(sp, child, ces, novelty_thr)

            # (5) S3: 边注入
            if n_s3 > 0:
                sp.sort(key=lambda x: x[1].fitness)
                donors_s3 = [t[1] for t in sp[:n_elite]]
                tail      = [t[1] for t in sp[sp_size // 2:]] or [t[1] for t in sp]
                for _ in range(n_s3):
                    donor  = donors_s3[rng.integers(len(donors_s3))]
                    target = tail[rng.integers(len(tail))]
                    child, ces = _make_child(
                        edge_injection(target.perm, donor.perm,
                                       edge_freq, inject_thr, n_elite, rng)
                    )
                    _crowding_replace(sp, child, ces, novelty_thr)

        # ── 写回种群（三元组，顺序正确）────────────────────────────
        for sp in species_triples:
            for global_idx, ind, _es in sp:
                pop.inds[global_idx] = ind

        # ── 双向环形迁移 ─────────────────────────────────────────
        if (migrate_every > 0 and gen > 0
                and gen % migrate_every == 0
                and len(species_triples) >= 2):
            num_sp = len(species_triples)
            best_of = [
                min(sp, key=lambda x: x[1].fitness) if sp else None
                for sp in species_triples
            ]
            for d in [1, -1]:
                for s in range(num_sp):
                    if best_of[s] is None:
                        continue
                    t = (s + d) % num_sp
                    tsp = species_triples[t]
                    if not tsp:
                        continue
                    worst_pos = max(range(len(tsp)), key=lambda i: tsp[i][1].fitness)
                    src_ind, src_es = best_of[s][1], best_of[s][2]
                    if src_ind.fitness < tsp[worst_pos][1].fitness:
                        tsp[worst_pos][1] = src_ind.copy()
                        tsp[worst_pos][2] = src_es          # 同步迁移 edge_set
                        pop.inds[tsp[worst_pos][0]] = tsp[worst_pos][1]

        # ── 全局小量随机注入 ─────────────────────────────────────
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

        # ── 全局最优精英保护 ─────────────────────────────────────
        pop.sort()
        if global_best.fitness < pop.best().fitness:
            pop.inds[-1] = global_best
            pop.sort()

        best_history.append(pop.best().fitness)

        if verbose and (gen % 50 == 0 or gen == generations - 1):
            divs = [_species_avg_jaccard(sp) for sp in species_triples]
            print(
                f"  [GSA] gen {gen:4d}: best={pop.best().fitness:.2f}, "
                f"sp={len(species_triples)}, "
                f"sizes={[len(sp) for sp in species_triples]}, "
                f"avg_div={float(np.mean(divs)):.3f}"
            )

    if record_stats:
        return pop.best(), best_history, species_stats
    return pop.best(), best_history