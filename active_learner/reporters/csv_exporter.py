from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from ..selectors.selector_base import SelectionResult


class CSVExporter:
    def __init__(
        self,
        output_dir: str | Path = "outputs",
        filename_prefix: str = "selected_samples",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filename_prefix = filename_prefix

    def _generate_filename(self, suffix: Optional[str] = None) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [self.filename_prefix, timestamp]
        if suffix:
            parts.append(suffix)
        return f"{'_'.join(parts)}.csv"

    def export(
        self,
        selection_result: SelectionResult,
        filename: Optional[str] = None,
        sort_by: str = "uncertainty_score",
        ascending: bool = False,
    ) -> str:
        df = selection_result.to_dataframe()

        valid_cols = set(df.columns.tolist())

        if sort_by in valid_cols:
            df = df.sort_values(by=sort_by, ascending=ascending)
        elif "uncertainty_score" in valid_cols:
            df = df.sort_values(by="uncertainty_score", ascending=False)

        if filename is None:
            filename = self._generate_filename()

        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return str(output_path)

    def export_labels(
        self,
        file_paths: list[str],
        labels: list,
        filename: Optional[str] = None,
    ) -> str:
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"new_labels_{timestamp}.csv"

        df = pd.DataFrame({
            "file_path": file_paths,
            "label": labels,
        })

        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return str(output_path)

    def export_json(
        self,
        selection_result: SelectionResult,
        filename: Optional[str] = None,
    ) -> str:
        df = selection_result.to_dataframe()

        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"selected_samples_{timestamp}.json"

        output_path = self.output_dir / filename
        df.to_json(output_path, orient="records", force_ascii=False, indent=2)
        return str(output_path)
