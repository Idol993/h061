from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


@dataclass
class DiversityResult:
    scores: np.ndarray
    cluster_labels: np.ndarray
    centroids: np.ndarray
    selected_indices: np.ndarray

    def __len__(self) -> int:
        return len(self.scores)


class DiversitySampler:
    def __init__(
        self,
        num_clusters: Optional[int] = None,
        seed: int = 42,
        metric: str = "euclidean",
    ) -> None:
        self.num_clusters = num_clusters
        self.seed = seed
        self.metric = metric

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return features / norms

    def _kmeans_select(
        self,
        features: np.ndarray,
        n_clusters: int,
    ) -> DiversityResult:
        n_samples = len(features)
        if n_samples == 0:
            return DiversityResult(
                scores=np.array([], dtype=np.float32),
                cluster_labels=np.array([], dtype=np.int32),
                centroids=np.array([], dtype=np.float32),
                selected_indices=np.array([], dtype=np.int64),
            )

        n_clusters = min(n_clusters, n_samples)
        n_init = 10 if n_samples >= 100 else 3

        try:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.seed,
                n_init=n_init,
                max_iter=300,
            )
            cluster_labels = kmeans.fit_predict(features).astype(np.int32)
            centroids = kmeans.cluster_centers_
        except Exception:
            rng = np.random.RandomState(self.seed)
            cluster_labels = rng.randint(0, n_clusters, size=n_samples).astype(np.int32)
            centroids = np.zeros((n_clusters, features.shape[1]), dtype=np.float32)
            for c in range(n_clusters):
                mask = cluster_labels == c
                if mask.any():
                    centroids[c] = features[mask].mean(axis=0)

        selected_indices: list[int] = []
        distances = pairwise_distances(features, centroids, metric=self.metric)

        diversity_scores = np.zeros(n_samples, dtype=np.float32)

        for c in range(n_clusters):
            cluster_mask = cluster_labels == c
            if not cluster_mask.any():
                continue
            cluster_indices = np.where(cluster_mask)[0]
            cluster_distances = distances[cluster_indices, c]
            if len(cluster_distances) > 0:
                min_idx_in_cluster = int(np.argmin(cluster_distances))
                selected_idx = int(cluster_indices[min_idx_in_cluster])
                selected_indices.append(selected_idx)

        max_dist = distances.max() if distances.size > 0 else 1.0
        if max_dist > 0:
            for c in range(n_clusters):
                cluster_mask = cluster_labels == c
                if not cluster_mask.any():
                    continue
                cluster_indices = np.where(cluster_mask)[0]
                cluster_distances = distances[cluster_indices, c]
                normalized = 1.0 - (cluster_distances / max_dist)
                normalized = np.clip(normalized, 0.0, 1.0)
                for i, idx in enumerate(cluster_indices):
                    diversity_scores[idx] = float(normalized[i])
        else:
            diversity_scores[:] = 0.5

        selected_arr = np.array(selected_indices, dtype=np.int64)

        return DiversityResult(
            scores=diversity_scores,
            cluster_labels=cluster_labels,
            centroids=centroids,
            selected_indices=selected_arr,
        )

    def _coreset_select(
        self,
        features: np.ndarray,
        n_select: int,
    ) -> DiversityResult:
        n_samples = len(features)
        n_select = min(n_select, n_samples)

        if n_samples == 0:
            return DiversityResult(
                scores=np.array([], dtype=np.float32),
                cluster_labels=np.array([], dtype=np.int32),
                centroids=np.array([], dtype=np.float32),
                selected_indices=np.array([], dtype=np.int64),
            )

        selected: list[int] = []
        dist_to_selected = np.full(n_samples, np.inf, dtype=np.float64)

        rng = np.random.RandomState(self.seed)
        first = int(rng.randint(0, n_samples))
        selected.append(first)

        while len(selected) < n_select:
            last_idx = selected[-1]
            last_feat = features[last_idx : last_idx + 1]
            new_dists = pairwise_distances(features, last_feat, metric=self.metric).flatten()
            dist_to_selected = np.minimum(dist_to_selected, new_dists)
            for s in selected:
                dist_to_selected[s] = -np.inf
            next_idx = int(np.argmax(dist_to_selected))
            selected.append(next_idx)
            dist_to_selected[next_idx] = -np.inf

        diversity_scores = np.zeros(n_samples, dtype=np.float32)
        selected_set = set(selected)
        max_d = dist_to_selected[dist_to_selected != -np.inf]
        if len(max_d) > 0 and max_d.max() > 0:
            for i in range(n_samples):
                if i in selected_set:
                    diversity_scores[i] = 1.0
                else:
                    d = dist_to_selected[i]
                    if np.isinf(d):
                        d = max_d.max()
                    diversity_scores[i] = float(np.clip(d / max_d.max(), 0.0, 1.0))
        else:
            for i in range(n_samples):
                diversity_scores[i] = 1.0 if i in selected_set else 0.5

        selected_arr = np.array(selected, dtype=np.int64)
        cluster_labels = np.zeros(n_samples, dtype=np.int32)
        for i, s in enumerate(selected):
            cluster_labels[s] = i
        centroids = features[selected_arr] if len(selected_arr) > 0 else np.zeros((0, features.shape[1]), dtype=np.float32)

        return DiversityResult(
            scores=diversity_scores,
            cluster_labels=cluster_labels,
            centroids=centroids,
            selected_indices=selected_arr,
        )

    def score(
        self,
        features: np.ndarray,
        n_select: int,
    ) -> DiversityResult:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            features = features.reshape(len(features), -1)

        features = self._normalize_features(features)
        n_clusters = self.num_clusters or n_select

        return self._kmeans_select(features, n_clusters)

    def select(
        self,
        features: np.ndarray,
        n_select: int,
    ) -> np.ndarray:
        result = self.score(features, n_select)
        if len(result.selected_indices) >= n_select:
            return result.selected_indices[:n_select]

        remaining = n_select - len(result.selected_indices)
        already_selected = set(result.selected_indices.tolist())
        sorted_by_score = np.argsort(-result.scores)
        extra = [i for i in sorted_by_score if int(i) not in already_selected][:remaining]
        return np.concatenate([result.selected_indices, np.array(extra, dtype=np.int64)])
