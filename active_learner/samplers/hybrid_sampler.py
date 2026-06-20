from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .diversity_sampler import DiversitySampler
from .uncertainty_sampler import UncertaintySampler


@dataclass
class HybridResult:
    scores: np.ndarray
    uncertainty_scores: np.ndarray
    diversity_scores: np.ndarray
    uncertainty_weight: float
    diversity_weight: float
    probs: np.ndarray

    def __len__(self) -> int:
        return len(self.scores)


class HybridSampler:
    def __init__(
        self,
        uncertainty_method: str = "entropy",
        uncertainty_weight: float = 0.6,
        diversity_weight: float = 0.4,
        num_clusters: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        total = uncertainty_weight + diversity_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"权重之和必须等于 1，当前: uncertainty={uncertainty_weight}, diversity={diversity_weight}")

        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self.uncertainty_sampler = UncertaintySampler(method=uncertainty_method)
        self.diversity_sampler = DiversitySampler(num_clusters=num_clusters, seed=seed)
        self.seed = seed

    def score(
        self,
        probs: np.ndarray,
        features: np.ndarray,
        n_select: int,
    ) -> HybridResult:
        probs = np.asarray(probs, dtype=np.float64)
        features = np.asarray(features, dtype=np.float64)

        if features.ndim != 2:
            features = features.reshape(len(features), -1)

        uncertainty_result = self.uncertainty_sampler.score(probs)
        diversity_result = self.diversity_sampler.score(features, n_select)

        n = len(uncertainty_result.scores)
        if len(diversity_result.scores) != n:
            raise ValueError(
                f"样本数不一致：uncertainty={n}, diversity={len(diversity_result.scores)}"
            )

        u_scores = uncertainty_result.scores.astype(np.float64)
        d_scores = diversity_result.scores.astype(np.float64)

        u_min, u_max = u_scores.min(), u_scores.max()
        d_min, d_max = d_scores.min(), d_scores.max()

        if u_max - u_min > 1e-12:
            u_norm = (u_scores - u_min) / (u_max - u_min)
        else:
            u_norm = np.full_like(u_scores, 0.5)

        if d_max - d_min > 1e-12:
            d_norm = (d_scores - d_min) / (d_max - d_min)
        else:
            d_norm = np.full_like(d_scores, 0.5)

        hybrid_scores = (
            self.uncertainty_weight * u_norm
            + self.diversity_weight * d_norm
        ).astype(np.float32)

        return HybridResult(
            scores=hybrid_scores,
            uncertainty_scores=u_scores.astype(np.float32),
            diversity_scores=d_scores.astype(np.float32),
            uncertainty_weight=self.uncertainty_weight,
            diversity_weight=self.diversity_weight,
            probs=probs.astype(np.float32),
        )

    def rank_indices(
        self,
        probs: np.ndarray,
        features: np.ndarray,
        n_select: int,
    ) -> np.ndarray:
        result = self.score(probs, features, n_select)
        return np.argsort(-result.scores)

    def select_top_k(
        self,
        probs: np.ndarray,
        features: np.ndarray,
        k: int,
    ) -> np.ndarray:
        ranked = self.rank_indices(probs, features, k)
        return ranked[:k]

    @staticmethod
    def get_reason(
        uncertainty_score: float,
        diversity_score: float,
        u_weight: float,
        d_weight: float,
        threshold: float = 0.7,
    ) -> str:
        u_weighted = uncertainty_score * u_weight
        d_weighted = diversity_score * d_weight
        reasons = []

        if uncertainty_score >= threshold:
            reasons.append("高不确定性")
        if diversity_score >= threshold:
            reasons.append("高多样性")

        if u_weighted > d_weighted:
            primary = "不确定性主导"
        else:
            primary = "多样性主导"

        if not reasons:
            reasons.append(primary)
        else:
            reasons.append(primary)

        return " + ".join(reasons)
