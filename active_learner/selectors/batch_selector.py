from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import pairwise_distances

from .selector_base import SelectionResult, SelectorBase


class BatchSelector(SelectorBase):
    def __init__(
        self,
        budget: int = 100,
        diversity_penalty: float = 0.3,
        seed: int = 42,
    ) -> None:
        super().__init__(budget=budget, seed=seed)
        self.diversity_penalty = diversity_penalty

    def _greedy_select_with_diversity(
        self,
        scores: np.ndarray,
        features: Optional[np.ndarray],
        n_select: int,
    ) -> np.ndarray:
        n_samples = len(scores)
        n_select = min(n_select, n_samples)

        if features is None or self.diversity_penalty <= 0:
            return np.argsort(-scores)[:n_select]

        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            features = features.reshape(n_samples, -1)

        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        features_norm = features / norms

        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-12:
            scores_norm = (scores - s_min) / (s_max - s_min)
        else:
            scores_norm = np.full_like(scores, 0.5)

        selected: list[int] = []
        remaining_mask = np.ones(n_samples, dtype=bool)
        dist_to_selected = np.full(n_samples, np.inf, dtype=np.float64)

        rng = np.random.RandomState(self.seed)
        adjusted_scores = scores_norm.copy()

        for _ in range(n_select):
            if not remaining_mask.any():
                break

            if selected:
                last_feat = features_norm[selected[-1] : selected[-1] + 1]
                new_dists = pairwise_distances(
                    features_norm[remaining_mask],
                    last_feat,
                    metric="euclidean",
                ).flatten()

                rem_indices = np.where(remaining_mask)[0]
                for idx_local, idx_global in enumerate(rem_indices):
                    if new_dists[idx_local] < dist_to_selected[idx_global]:
                        dist_to_selected[idx_global] = new_dists[idx_local]

                max_dist = dist_to_selected[remaining_mask]
                max_dist = max_dist[np.isfinite(max_dist)]
                if len(max_dist) > 0 and max_dist.max() > 0:
                    norm_dists = np.zeros(n_samples, dtype=np.float64)
                    rem_idx = np.where(remaining_mask)[0]
                    for idx in rem_idx:
                        d = dist_to_selected[idx]
                        if not np.isfinite(d):
                            d = max_dist.max()
                        norm_dists[idx] = d / max_dist.max()
                    adjusted_scores = (
                        (1.0 - self.diversity_penalty) * scores_norm
                        + self.diversity_penalty * norm_dists
                    )
                adjusted_scores[~remaining_mask] = -np.inf

            candidates = np.where(adjusted_scores == adjusted_scores.max())[0]
            if len(candidates) == 1:
                best_idx = int(candidates[0])
            else:
                best_idx = int(candidates[rng.randint(len(candidates))])

            selected.append(best_idx)
            remaining_mask[best_idx] = False
            dist_to_selected[best_idx] = -np.inf

        return np.array(selected, dtype=np.int64)

    def select(
        self,
        scores: np.ndarray,
        features: Optional[np.ndarray] = None,
        uncertainty_scores: Optional[np.ndarray] = None,
        diversity_scores: Optional[np.ndarray] = None,
        file_paths: Optional[list[str]] = None,
        probs: Optional[np.ndarray] = None,
        u_weight: float = 0.6,
        d_weight: float = 0.4,
    ) -> SelectionResult:
        scores = np.asarray(scores, dtype=np.float32).flatten()

        selected_indices = self._greedy_select_with_diversity(
            scores=scores,
            features=features,
            n_select=self.budget,
        )

        selected_scores = np.array([scores[i] for i in selected_indices], dtype=np.float32)

        if uncertainty_scores is None:
            uncertainty_scores = scores
        if diversity_scores is None:
            diversity_scores = np.zeros_like(scores, dtype=np.float32)

        reasons: list[str] = []
        for i in range(len(scores)):
            u = float(uncertainty_scores[i]) if i < len(uncertainty_scores) else 0.0
            d = float(diversity_scores[i]) if i < len(diversity_scores) else 0.0
            parts = []
            if u >= 0.7:
                parts.append("高不确定性")
            if d >= 0.7:
                parts.append("高多样性")
            if not parts:
                if u * u_weight >= d * d_weight:
                    parts.append("不确定性主导")
                else:
                    parts.append("多样性主导")
            reasons.append(" + ".join(parts))

        return SelectionResult(
            indices=selected_indices,
            scores=selected_scores,
            uncertainty_scores=np.asarray(uncertainty_scores, dtype=np.float32),
            diversity_scores=np.asarray(diversity_scores, dtype=np.float32),
            recommended_reasons=reasons,
            file_paths=file_paths,
            probs=probs,
        )
