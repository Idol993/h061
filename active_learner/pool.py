from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


POOL_FILENAME = "sample_pool.json"


@dataclass
class PoolSnapshot:
    unlabeled: int = 0
    pending: int = 0
    labeled: int = 0
    skipped: int = 0


@dataclass
class PoolDelta:
    new_pending: int = 0
    recovered_labeled: int = 0
    remaining_unlabeled: int = 0
    repeated_recommend: int = 0


class SamplePool:
    UNLABELED = "unlabeled"
    PENDING = "pending"
    LABELED = "labeled"
    SKIPPED = "skipped"

    VALID_STATES = {UNLABELED, PENDING, LABELED, SKIPPED}

    def __init__(self, output_dir: str | Path = "outputs") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.states: dict[str, str] = {}
        self.last_export_csv: Optional[str] = None

    def pool_path(self) -> Path:
        return self.output_dir / POOL_FILENAME

    def init_from_file_paths(self, file_paths: list[str], existing_labels: Optional[list] = None) -> int:
        new_count = 0
        norm_map: dict[str, str] = {}
        for fp in file_paths:
            norm = fp.replace("\\", "/")
            norm_map[norm] = fp

        for norm_fp in norm_map:
            if norm_fp not in self.states:
                self.states[norm_fp] = self.UNLABELED
                new_count += 1

        if existing_labels is not None:
            for i, l in enumerate(existing_labels):
                if i < len(file_paths):
                    norm = file_paths[i].replace("\\", "/")
                    if self._is_valid_label(l) and self.states.get(norm) != self.LABELED:
                        self.states[norm] = self.LABELED

        return new_count

    @staticmethod
    def _is_valid_label(l) -> bool:
        if l is None:
            return False
        import numpy as np
        if isinstance(l, (int, np.integer)):
            return l != -1
        if isinstance(l, float):
            if np.isnan(l):
                return False
            return l != -1.0
        if isinstance(l, str):
            return l.strip() != ""
        try:
            import pandas as pd
            if pd.isna(l):
                return False
        except Exception:
            pass
        return True

    def mark_pending(self, file_paths: list[str]) -> tuple[int, int]:
        new_pending = 0
        repeated = 0
        for fp in file_paths:
            norm = fp.replace("\\", "/")
            current = self.states.get(norm)
            if current == self.UNLABELED:
                self.states[norm] = self.PENDING
                new_pending += 1
            elif current == self.PENDING:
                repeated += 1
            elif current is None:
                self.states[norm] = self.PENDING
                new_pending += 1
        return new_pending, repeated

    def apply_labels(
        self,
        labels_map: dict[str, str],
        file_paths: Optional[list[str]] = None,
    ) -> tuple[int, int, int, list[str]]:
        recovered = 0
        dedup_count = 0
        external_paths: list[str] = []

        norm_file_set: Optional[set] = None
        if file_paths is not None:
            norm_file_set = {fp.replace("\\", "/") for fp in file_paths}

        seen: dict[str, str] = {}
        for fp, lbl in labels_map.items():
            norm = fp.replace("\\", "/")
            seen[norm] = lbl
        dedup_count = len(labels_map) - len(seen)

        for norm_fp, lbl in seen.items():
            if not self._is_valid_label(lbl):
                continue
            if norm_file_set is not None and norm_fp not in norm_file_set:
                external_paths.append(norm_fp)
                continue
            current = self.states.get(norm_fp)
            if current == self.PENDING:
                self.states[norm_fp] = self.LABELED
                recovered += 1
            elif current == self.UNLABELED:
                self.states[norm_fp] = self.LABELED
                recovered += 1
            elif current is None:
                if norm_file_set is not None:
                    external_paths.append(norm_fp)
                else:
                    self.states[norm_fp] = self.LABELED
                    recovered += 1

        return recovered, dedup_count, len(external_paths), external_paths

    def mark_skipped(self, file_paths: list[str]) -> int:
        skipped = 0
        for fp in file_paths:
            norm = fp.replace("\\", "/")
            if self.states.get(norm) in (self.UNLABELED, self.PENDING):
                self.states[norm] = self.SKIPPED
                skipped += 1
        return skipped

    def get_available_indices(self, file_paths: list[str]) -> list[int]:
        indices = []
        for i, fp in enumerate(file_paths):
            norm = fp.replace("\\", "/")
            if self.states.get(norm, self.UNLABELED) == self.UNLABELED:
                indices.append(i)
        return indices

    def get_state(self, file_path: str) -> str:
        return self.states.get(file_path.replace("\\", "/"), self.UNLABELED)

    def snapshot(self) -> PoolSnapshot:
        snap = PoolSnapshot()
        for s in self.states.values():
            if s == self.UNLABELED:
                snap.unlabeled += 1
            elif s == self.PENDING:
                snap.pending += 1
            elif s == self.LABELED:
                snap.labeled += 1
            elif s == self.SKIPPED:
                snap.skipped += 1
        return snap

    def delta(self, new_pending: int, recovered_labeled: int, repeated: int) -> PoolDelta:
        snap = self.snapshot()
        return PoolDelta(
            new_pending=new_pending,
            recovered_labeled=recovered_labeled,
            remaining_unlabeled=snap.unlabeled,
            repeated_recommend=repeated,
        )

    def save(self) -> str:
        data = {
            "states": self.states,
            "last_export_csv": self.last_export_csv,
        }
        path = self.pool_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return str(path)

    def load(self) -> "SamplePool":
        path = self.pool_path()
        if not path.exists():
            return self
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.states = data.get("states", {})
        self.last_export_csv = data.get("last_export_csv")
        return self

    def by_state(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {
            self.UNLABELED: [],
            self.PENDING: [],
            self.LABELED: [],
            self.SKIPPED: [],
        }
        for fp, st in self.states.items():
            result.setdefault(st, []).append(fp)
        return result
