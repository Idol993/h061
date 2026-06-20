from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import entropy


@dataclass
class UncertaintyResult:
    scores: np.ndarray
    probs: np.ndarray
    method: str

    def __len__(self) -> int:
        return len(self.scores)


class UncertaintySampler:
    def __init__(
        self,
        method: Literal["least_confidence", "margin", "entropy"] = "entropy",
    ) -> None:
        self.method = method

    @staticmethod
    def least_confidence(probs: np.ndarray) -> np.ndarray:
        max_probs = np.max(probs, axis=1)
        return 1.0 - max_probs

    @staticmethod
    def margin_sampling(probs: np.ndarray) -> np.ndarray:
        sorted_probs = np.sort(probs, axis=1)[:, ::-1]
        if sorted_probs.shape[1] >= 2:
            margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        else:
            margin = sorted_probs[:, 0]
        return 1.0 - margin

    @staticmethod
    def entropy_sampling(probs: np.ndarray) -> np.ndarray:
        eps = 1e-12
        probs_clipped = np.clip(probs, eps, 1.0)
        ent = entropy(probs_clipped, axis=1, base=2)
        max_ent = np.log2(probs.shape[1]) if probs.shape[1] > 1 else 1.0
        if max_ent > 0:
            return ent / max_ent
        return ent

    def score(self, probs: np.ndarray) -> UncertaintyResult:
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim != 2:
            raise ValueError(f"probs 应该是二维数组，当前 shape={probs.shape}")

        probs_sum = probs.sum(axis=1, keepdims=True)
        probs_sum = np.where(probs_sum == 0, 1.0, probs_sum)
        probs = probs / probs_sum

        if self.method == "least_confidence":
            scores = self.least_confidence(probs)
        elif self.method == "margin":
            scores = self.margin_sampling(probs)
        elif self.method == "entropy":
            scores = self.entropy_sampling(probs)
        else:
            raise ValueError(f"未知的不确定性采样方法: {self.method}")

        scores = scores.astype(np.float32)
        return UncertaintyResult(scores=scores, probs=probs, method=self.method)

    def rank_indices(self, probs: np.ndarray) -> np.ndarray:
        result = self.score(probs)
        return np.argsort(-result.scores)

    def select_top_k(self, probs: np.ndarray, k: int) -> np.ndarray:
        ranked = self.rank_indices(probs)
        return ranked[:k]
