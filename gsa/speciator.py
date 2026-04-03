from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree


def speciate_by_rank(population, num_species=3):
    population.sort_by_fitness()
    individuals = population.individuals

    species_dict = {}
    size = len(individuals)
    chunk_size = max(1, size // max(1, num_species))

    for i in range(num_species):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_species - 1 else size
        members = individuals[start:end]

        if members:
            species_dict[i] = members
            for ind in members:
                ind.species_id = i

    return species_dict


def _infer_depot(problem):
    if hasattr(problem, "depot"):
        return problem.depot
    return 1


def _strip_depot(route, depot):
    route = list(route)
    if route and route[0] == depot:
        route = route[1:]
    if route and route[-1] == depot:
        route = route[:-1]
    return route


def _canonical_edge(a, b):
    return (a, b) if a < b else (b, a)


def _build_edge_index(problem):
    depot = _infer_depot(problem)
    customers = list(problem.customers)
    nodes = [depot] + customers

    edge_to_idx = {}
    idx = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            edge_to_idx[(nodes[i], nodes[j])] = idx
            idx += 1

    return edge_to_idx


def _route_edges(route, depot):
    if len(route) == 0:
        return []

    full_route = [depot] + list(route) + [depot]
    edges = []
    for i in range(len(full_route) - 1):
        edges.append(_canonical_edge(full_route[i], full_route[i + 1]))
    return edges


def _build_edge_features(individuals, problem):
    depot = _infer_depot(problem)
    edge_to_idx = _build_edge_index(problem)
    edge_dim = len(edge_to_idx)

    fitness_values = np.array([ind.fitness for ind in individuals], dtype=float)
    fit_min = float(fitness_values.min()) if len(fitness_values) else 0.0
    fit_max = float(fitness_values.max()) if len(fitness_values) else 1.0
    fit_span = max(1e-8, fit_max - fit_min)

    capacity = getattr(problem, "capacity", None)

    X = []
    for ind in individuals:
        edge_vec = np.zeros(edge_dim, dtype=float)
        decoded_routes = problem.decode(ind.chromosome)
        routes = [_strip_depot(route, depot) for route in decoded_routes]

        route_load_ratios = []

        for route in routes:
            for e in _route_edges(route, depot):
                idx = edge_to_idx.get(e)
                if idx is not None:
                    edge_vec[idx] = 1.0

            if capacity is not None and hasattr(problem, "demands"):
                load = sum(problem.demands[c] for c in route if c in problem.demands)
                route_load_ratios.append(load / max(1.0, float(capacity)))

        route_count = len(routes)
        route_count_feature = np.array(
            [route_count / max(1, len(problem.customers))],
            dtype=float,
        )

        mean_load_ratio = float(np.mean(route_load_ratios)) if route_load_ratios else 0.0
        max_load_ratio = float(np.max(route_load_ratios)) if route_load_ratios else 0.0
        load_features = np.array([mean_load_ratio, max_load_ratio], dtype=float)

        fitness_feature = np.array(
            [(ind.fitness - fit_min) / fit_span],
            dtype=float,
        )

        feat = np.concatenate(
            [edge_vec, route_count_feature, load_features, fitness_feature]
        )
        X.append(feat)

    X = np.vstack(X)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    X = (X - mean) / std

    return X


def _groups_from_labels(individuals, labels):
    raw_groups = defaultdict(list)
    for idx, label in enumerate(labels):
        raw_groups[int(label)].append(individuals[idx])

    species_dict = {}
    new_sid = 0
    for old_sid in sorted(raw_groups.keys()):
        members = raw_groups[old_sid]
        if not members:
            continue

        members.sort(key=lambda ind: ind.fitness)
        for ind in members:
            ind.species_id = new_sid

        species_dict[new_sid] = members
        new_sid += 1

    return species_dict


def speciate_by_kmeans(
    population,
    problem,
    num_species=3,
    random_state=42,
    fallback_to_rank=True,
):
    individuals = population.individuals
    size = len(individuals)

    if size == 0:
        return {}

    if size <= num_species:
        return speciate_by_rank(population, num_species=num_species)

    try:
        X = _build_edge_features(individuals, problem)

        k = max(1, min(int(num_species), size))

        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=random_state,
        )
        labels = model.fit_predict(X)

        species_dict = _groups_from_labels(individuals, labels)

        if len(species_dict) == 0 and fallback_to_rank:
            return speciate_by_rank(population, num_species=num_species)

        return species_dict

    except Exception:
        if fallback_to_rank:
            return speciate_by_rank(population, num_species=num_species)
        raise


class _StableWishartClustering:

    def __init__(
        self,
        wishart_neighbors=7,
        significance_level=0.20,
        pca_components=10,
        random_state=42,
    ):
        self.wishart_neighbors = int(wishart_neighbors)
        self.significance_level = float(significance_level)
        self.pca_components = int(pca_components)
        self.random_state = random_state

        self.object_labels = None
        self.dist_ = None
        self.dk_ = None
        self.embedding_ = None

        self._clusters = None
        self._clusters_to_objects = None
        self._pca = None

    def fit(self, X, verbose=False):
        del verbose

        X = np.asarray(X, dtype=float)
        size = X.shape[0]

        if size == 0:
            self.object_labels = np.array([], dtype=int)
            self.dist_ = np.zeros((0, 0), dtype=float)
            self.dk_ = np.array([], dtype=float)
            self.embedding_ = np.zeros((0, 0), dtype=float)
            return self

        Z = self._reduce_dimensionality(X)
        self.embedding_ = Z

        size, dim = Z.shape
        if size == 1:
            self.object_labels = np.array([1], dtype=int)
            self.dist_ = np.zeros((1, 1), dtype=float)
            self.dk_ = np.array([0.0], dtype=float)
            return self

        k = max(1, min(self.wishart_neighbors, size - 1))
        tree = KDTree(Z, metric="euclidean")
        distances, neighbors = tree.query(Z, k=k + 1, return_distance=True)

        neighbor_idx = neighbors[:, 1:]
        dk = distances[:, -1]

        diff = Z[:, None, :] - Z[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff * diff, axis=2))

        self.dist_ = dist_matrix
        self.dk_ = dk

        order = np.argsort(dk)
        self.object_labels = np.full(size, -1, dtype=int)

        # [min_radius, max_radius, significant_flag]
        self._clusters = np.array([(1.0, 1.0, 0.0)], dtype=float)
        self._clusters_to_objects = defaultdict(list)

        for idx in order:
            nbr_labels = self.object_labels[neighbor_idx[idx]]
            unique_clusters = np.unique(nbr_labels).astype(int)
            unique_clusters = unique_clusters[unique_clusters != -1]

            if len(unique_clusters) == 0:
                self._create_new_cluster(idx, dk[idx])
                continue

            max_cluster = unique_clusters[-1]
            min_cluster = unique_clusters[0]

            if max_cluster == min_cluster:
                if self._clusters[max_cluster, -1] < 0.5:
                    self._add_elem_to_exist_cluster(idx, dk[idx], max_cluster)
                else:
                    self._add_elem_to_noise(idx)
                continue

            my_clusters = self._clusters[unique_clusters]
            flags = my_clusters[:, -1]

            if np.min(flags) > 0.5:
                self._add_elem_to_noise(idx)
                continue

            significance = self._compute_significance(my_clusters, size=size, dim=dim)
            significant_mask = significance >= self.significance_level

            significant_clusters = unique_clusters[significant_mask]
            non_significant_clusters = unique_clusters[~significant_mask]

            if len(significant_clusters) > 1 or min_cluster == 0:
                self._add_elem_to_noise(idx)
                if len(significant_clusters) > 0:
                    self._clusters[significant_clusters, -1] = 1.0

                for cluster_id in non_significant_clusters:
                    if cluster_id == 0:
                        continue
                    for bad_idx in self._clusters_to_objects[cluster_id]:
                        self._add_elem_to_noise(bad_idx)
                    self._clusters_to_objects[cluster_id].clear()
            else:
                for cluster_id in unique_clusters:
                    if cluster_id == min_cluster:
                        continue
                    for bad_idx in self._clusters_to_objects[cluster_id]:
                        self._add_elem_to_exist_cluster(bad_idx, dk[bad_idx], min_cluster)
                    self._clusters_to_objects[cluster_id].clear()

                self._add_elem_to_exist_cluster(idx, dk[idx], min_cluster)

        self.object_labels = self._clean_labels()
        return self

    def _reduce_dimensionality(self, X):
        n, d = X.shape
        target_dim = min(self.pca_components, n - 1, d)
        target_dim = max(2, target_dim) if n >= 3 and d >= 2 else min(1, d)

        if target_dim <= 0:
            return X.copy()

        if d <= target_dim:
            return X.copy()

        self._pca = PCA(n_components=target_dim, random_state=self.random_state)
        Z = self._pca.fit_transform(X)
        return Z

    def _compute_significance(self, cluster_stats, size, dim):
        eps = 1e-10
        min_r = np.maximum(cluster_stats[:, 0], eps)
        max_r = np.maximum(cluster_stats[:, 1], eps)

        log_gap = np.log(max_r + eps) - np.log(min_r + eps)
        density_gap = (dim * log_gap) * (self.wishart_neighbors / max(1.0, float(size)))

        density_gap = np.nan_to_num(
            density_gap,
            nan=0.0,
            posinf=1e6,
            neginf=0.0,
        )
        return density_gap

    def _clean_labels(self):
        labels = np.asarray(self.object_labels, dtype=int)
        unique = np.unique(labels)

        mapping = {}
        next_label = 0

        if 0 in unique:
            mapping[0] = 0
            next_label = 1

        for old_label in sorted(unique):
            if old_label == 0:
                continue
            mapping[old_label] = next_label
            next_label += 1

        cleaned = np.zeros_like(labels)
        for i, lab in enumerate(labels):
            cleaned[i] = mapping[lab]
        return cleaned

    def _add_elem_to_noise(self, index):
        self.object_labels[index] = 0
        self._clusters_to_objects[0].append(index)

    def _create_new_cluster(self, index, dist):
        new_cluster_id = len(self._clusters)
        self.object_labels[index] = new_cluster_id
        self._clusters_to_objects[new_cluster_id].append(index)
        self._clusters = np.append(self._clusters, [(dist, dist, 0.0)], axis=0)

    def _add_elem_to_exist_cluster(self, index, dist, cluster_label):
        self.object_labels[index] = cluster_label
        self._clusters_to_objects[cluster_label].append(index)
        self._clusters[cluster_label, 0] = min(self._clusters[cluster_label, 0], dist)
        self._clusters[cluster_label, 1] = max(self._clusters[cluster_label, 1], dist)


def _cluster_centroids(X, labels):
    centroids = {}
    for c in sorted(set(labels)):
        members = X[labels == c]
        if len(members) == 0:
            continue
        centroids[c] = members.mean(axis=0)
    return centroids


def _small_clusters_to_noise(labels, min_cluster_size=3):
    labels = np.asarray(labels, dtype=int).copy()
    unique, counts = np.unique(labels, return_counts=True)
    sizes = dict(zip(unique.tolist(), counts.tolist()))

    for cluster_id, cnt in sizes.items():
        if cluster_id != 0 and cnt < min_cluster_size:
            labels[labels == cluster_id] = 0

    return labels


def _attach_noise_to_nearest_cluster(X, labels):
    labels = np.asarray(labels, dtype=int).copy()

    non_noise_clusters = sorted([c for c in np.unique(labels) if c != 0])
    if len(non_noise_clusters) == 0:
        return labels

    centroids = _cluster_centroids(X, labels)

    noise_idx = np.where(labels == 0)[0]
    for idx in noise_idx:
        x = X[idx]
        best_cluster = min(
            non_noise_clusters,
            key=lambda c: float(np.sum((x - centroids[c]) ** 2)),
        )
        labels[idx] = best_cluster

    return labels


def _keep_noise_as_explorer_if_possible(labels, min_cluster_size=3):
    labels = np.asarray(labels, dtype=int).copy()
    noise_count = int(np.sum(labels == 0))
    if noise_count >= min_cluster_size:
        return labels
    return labels


def _noise_to_singletons(labels):
    labels = np.asarray(labels, dtype=int).copy()
    max_label = int(labels.max()) if len(labels) else 0
    noise_idx = np.where(labels == 0)[0]

    for idx in noise_idx:
        max_label += 1
        labels[idx] = max_label

    return labels


def _reindex_nonzero_labels_keep_noise(labels):
    labels = np.asarray(labels, dtype=int).copy()

    unique = sorted(set(labels.tolist()))
    mapping = {}
    next_id = 0

    if 0 in unique:
        mapping[0] = 0
        next_id = 1

    for lab in unique:
        if lab == 0:
            continue
        mapping[lab] = next_id
        next_id += 1

    out = np.array([mapping[x] for x in labels], dtype=int)
    return out


def speciate_by_wishart(
    population,
    problem,
    wishart_neighbors=7,
    significance_level=0.25,
    min_cluster_size=3,
    attach_noise="explorer",   # explorer / nearest / singleton
    fallback_to_rank=True,
    pca_components=10,
    random_state=42,
):
    individuals = population.individuals
    size = len(individuals)

    if size == 0:
        return {}

    if size <= 2:
        return speciate_by_rank(population, num_species=1)

    try:
        X = _build_edge_features(individuals, problem)

        model = _StableWishartClustering(
            wishart_neighbors=wishart_neighbors,
            significance_level=significance_level,
            pca_components=pca_components,
            random_state=random_state,
        ).fit(X, verbose=False)

        labels = np.asarray(model.object_labels, dtype=int)

        # Step 1: collapse tiny clusters into noise
        labels = _small_clusters_to_noise(labels, min_cluster_size=min_cluster_size)

        # Step 2: decide what to do with noise
        if attach_noise == "singleton":
            labels = _noise_to_singletons(labels)
        elif attach_noise == "nearest":
            labels = _attach_noise_to_nearest_cluster(model.embedding_, labels)
        else:
            # default: keep noise as explorer pool if meaningful
            labels = _keep_noise_as_explorer_if_possible(
                labels, min_cluster_size=min_cluster_size
            )

        labels = _reindex_nonzero_labels_keep_noise(labels)

        species_dict = _groups_from_labels(individuals, labels)

        if len(species_dict) == 0 and fallback_to_rank:
            return speciate_by_rank(population, num_species=1)

        return species_dict

    except Exception:
        if fallback_to_rank:
            return speciate_by_rank(population, num_species=1)
        raise