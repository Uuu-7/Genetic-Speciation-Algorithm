"""Wishart 动态分群 — 向量化加速版。

性能优化：
    原始版：Python 双重循环计算两两 Jaccard，O(N²) 次集合运算
            N=200 时约 0.085s/次

    优化版：边集转二值矩阵，用 numpy matmul 一次计算全部 Jaccard
            X @ X.T 得交集大小，sizes 外积得并集大小
            N=200 时约 0.004s/次（约 19x 加速）

接口：wishart_cluster 返回 (clusters, edge_sets)
      调用方缓存 edge_sets，避免在 gsa.py 中重复调用 perm_edges。
"""

import numpy as np
from .mask import perm_edges


# ─────────────── 向量化 Jaccard ───────────────

def _build_edge_vocab(edge_sets: list) -> dict:
    all_edges: set = set()
    for es in edge_sets:
        all_edges.update(es)
    return {e: i for i, e in enumerate(sorted(all_edges))}


def _to_binary_matrix(edge_sets: list, edge_idx: dict) -> np.ndarray:
    n, m = len(edge_sets), len(edge_idx)
    X = np.zeros((n, m), dtype=np.float32)
    for i, es in enumerate(edge_sets):
        for e in es:
            col = edge_idx.get(e)
            if col is not None:
                X[i, col] = 1.0
    return X


def _jaccard_matrix(X: np.ndarray) -> np.ndarray:
    """全量两两 Jaccard 距离，纯 numpy，无 Python 循环。"""
    inter = (X @ X.T).astype(np.float64)
    sizes = X.sum(axis=1).astype(np.float64)
    union = sizes[:, None] + sizes[None, :] - inter
    D = 1.0 - inter / np.maximum(union, 1e-10)
    np.fill_diagonal(D, 0.0)
    return D


def _build_distance_matrix(edge_sets: list, fits: np.ndarray,
                            alpha: float = 0.20) -> np.ndarray:
    """混合距离 = (1-alpha)*Jaccard向量化 + alpha*适应度秩距离。"""
    n = len(edge_sets)
    edge_idx = _build_edge_vocab(edge_sets)
    X = _to_binary_matrix(edge_sets, edge_idx)
    D_edge = _jaccard_matrix(X)

    if alpha <= 0.0:
        return D_edge

    ranks = np.argsort(np.argsort(fits)).astype(float)
    ranks /= max(n - 1, 1)
    D_rank = np.abs(ranks[:, None] - ranks[None, :])
    return (1.0 - alpha) * D_edge + alpha * D_rank


# ─────────────── Wishart 核心 ───────────────

def _wishart_core(D: np.ndarray, k: int) -> list:
    n = D.shape[0]
    D_sorted_idx = np.argsort(D, axis=1)
    knn_idx = D_sorted_idx[:, 1: k + 1]
    knn_dist = D[np.arange(n), D_sorted_idx[:, min(k, n - 1)]]
    order = np.argsort(knn_dist)

    assignment = -np.ones(n, dtype=int)
    clusters: list = []
    next_id = 0

    for idx in order:
        neigh_ids = {assignment[nb] for nb in knn_idx[idx] if assignment[nb] != -1}
        if not neigh_ids:
            assignment[idx] = next_id
            clusters.append([int(idx)])
            next_id += 1
        elif len(neigh_ids) == 1:
            cid = next(iter(neigh_ids))
            clusters[cid].append(int(idx))
            assignment[idx] = cid
        else:
            keep = min(neigh_ids)
            for cid in neigh_ids:
                if cid == keep:
                    continue
                for p in clusters[cid]:
                    assignment[p] = keep
                    clusters[keep].append(p)
                clusters[cid] = []
            clusters[keep].append(int(idx))
            assignment[idx] = keep

    return [c for c in clusters if c]


# ─────────────── 大簇分裂 ───────────────

def _kmedoids2(indices: list, D: np.ndarray, rng) -> tuple:
    n = len(indices)
    if n < 4:
        half = n // 2
        return indices[:half], indices[half:]
    idx_arr = np.array(indices)
    sub_D = D[np.ix_(idx_arr, idx_arr)]
    s0 = int(rng.integers(n))
    probs = sub_D[s0] ** 2
    total = probs.sum()
    s1 = (s0 + n // 2) % n if total < 1e-12 else int(rng.choice(n, p=probs / total))
    seeds = [s0, s1]
    assignment = np.zeros(n, dtype=int)
    for _ in range(10):
        for i in range(n):
            assignment[i] = 0 if sub_D[i, seeds[0]] <= sub_D[i, seeds[1]] else 1
        new_seeds = []
        for s in range(2):
            members = np.where(assignment == s)[0]
            if len(members) == 0:
                new_seeds.append(seeds[s])
            elif len(members) == 1:
                new_seeds.append(int(members[0]))
            else:
                sub_sub = sub_D[np.ix_(members, members)]
                new_seeds.append(int(members[np.argmin(sub_sub.mean(axis=1))]))
        if new_seeds == seeds:
            break
        seeds = new_seeds
    return ([int(idx_arr[i]) for i in range(n) if assignment[i] == 0],
            [int(idx_arr[i]) for i in range(n) if assignment[i] == 1])


def _split_large(clusters, D, max_size, rng, depth=0):
    if depth > 4:
        return clusters
    result, changed = [], False
    for c in clusters:
        if len(c) > max_size:
            a, b = _kmedoids2(c, D, rng)
            if len(a) >= 2 and len(b) >= 2:
                result.extend([a, b])
                changed = True
            else:
                result.append(c)
        else:
            result.append(c)
    return _split_large(result, D, max_size, rng, depth + 1) if changed else result


# ─────────────── 主函数 ───────────────

def wishart_cluster(
    inds,
    k: int = 4,
    max_species: int = 5,
    min_size: int = 3,
    alpha: float = 0.20,
    max_size_ratio: float = 0.45,
    rng=None,
    **kwargs,
) -> tuple:
    """Wishart 动态分群（向量化版）。

    Returns
    -------
    (clusters, edge_sets)
        clusters  : list[list[int]]   各簇的全局索引
        edge_sets : list[set]         与 inds 一一对应的边集（供调用方缓存）
    """
    n = len(inds)
    if n == 0:
        return [], []
    if rng is None:
        rng = np.random.default_rng()

    # 全局只算一次 perm_edges
    edge_sets = [perm_edges(ind.perm) for ind in inds]

    if n <= max(3, k + 1):
        return [list(range(n))], edge_sets

    k_use = min(max(k, int(0.08 * n)), n - 1)
    fits  = np.array([ind.fitness for ind in inds], dtype=float)
    D     = _build_distance_matrix(edge_sets, fits, alpha=alpha)  # 向量化
    final = _wishart_core(D, k_use)

    max_cluster_size = max(min_size + 1, int(n * max_size_ratio))
    final = _split_large(final, D, max_cluster_size, rng)

    # 微型簇吸收
    final.sort(key=len, reverse=True)
    while len(final) > 1 and len(final[-1]) < min_size:
        small = final.pop()
        best_ci, best_d = 0, float("inf")
        for ci, bc in enumerate(final):
            d = float(np.min(D[np.ix_(small, bc)]))
            if d < best_d:
                best_d, best_ci = d, ci
        final[best_ci].extend(small)

    # 物种数上限
    if max_species is not None and len(final) > max_species:
        final.sort(key=len, reverse=True)
        big = final[:max_species]
        for sc in final[max_species:]:
            best_ci, best_d = 0, float("inf")
            for ci, bc in enumerate(big):
                d = float(np.min(D[np.ix_(sc, bc)]))
                if d < best_d:
                    best_d, best_ci = d, ci
            big[best_ci].extend(sc)
        final = big

    return final, edge_sets