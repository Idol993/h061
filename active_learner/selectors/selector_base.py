from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SelectionResult:
    indices: np.ndarray
    scores: np.ndarray
    uncertainty_scores: Optional[np.ndarray] = None
    diversity_scores: Optional[np.ndarray] = None
    recommended_reasons: Optional[list[str]] = None
    file_paths: Optional[list[str]] = None
    probs: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.indices)

    def to_dataframe(self):
        import pandas as pd

        n = len(self.indices)
        data = {}
        if self.file_paths is not None:
            data["file_path"] = [self.file_paths[i] for i in self.indices]
        else:
            data["index"] = self.indices.tolist()

        data["uncertainty_score"] = (
            [float(self.uncertainty_scores[i]) for i in self.indices]
            if self.uncertainty_scores is not None
            else [0.0] * n
        )
        data["diversity_score"] = (
            [float(self.diversity_scores[i]) for i in self.indices]
            if self.diversity_scores is not None
            else [0.0] * n
        )
        data["recommended_reason"] = (
            [self.recommended_reasons[i] for i in self.indices]
            if self.recommended_reasons is not None
            else [""] * n
        )

        if self.scores is not None and len(self.scores) > 0:
            sorted_scores = [float(self.scores[i]) for i in range(n)]
            order = np.argsort([-s for s in sorted_scores])
            for key in data:
                data[key] = [data[key][o] for o in order]
            self.indices = self.indices[order]
            self.scores = self.scores[order]

        return pd.DataFrame(data)


class SelectorBase(ABC):
    def __init__(self, budget: int = 100, seed: int = 42) -> None:
        self.budget = budget
        self.seed = seed

    @abstractmethod
    def select(
        self,
        scores: np.ndarray,
        features: Optional[np.ndarray] = None,
        **kwargs,
    ) -> SelectionResult:
        raise NotImplementedError
