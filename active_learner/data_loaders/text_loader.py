from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chardet
import numpy as np
import pandas as pd


@dataclass
class TextDataset:
    file_paths: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    labels: Optional[list] = field(default=None)
    text_column: str = "text"
    label_column: str = "label"

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "file_path": self.file_paths[idx] if self.file_paths else str(idx),
            "text": self.texts[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


class TextLoader:
    TEXT_COLUMNS = ["text", "content", "sentence", "review", "input", "data", "document"]
    LABEL_COLUMNS = ["label", "category", "class", "target", "y", "output", "sentiment"]

    def __init__(self, text_column: Optional[str] = None, label_column: Optional[str] = None) -> None:
        self.text_column = text_column
        self.label_column = label_column

    @staticmethod
    def _detect_encoding(file_path: Path) -> str:
        with open(file_path, "rb") as f:
            raw = f.read(65536)
        result = chardet.detect(raw)
        return result.get("encoding", "utf-8") or "utf-8"

    def _detect_columns(self, df: pd.DataFrame) -> tuple[str, Optional[str]]:
        text_col = self.text_column
        label_col = self.label_column

        if text_col is None:
            lower_cols = {c.lower(): c for c in df.columns}
            for candidate in self.TEXT_COLUMNS:
                if candidate in lower_cols:
                    text_col = lower_cols[candidate]
                    break
            if text_col is None:
                text_col = df.columns[0]

        if label_col is None and self.label_column is not False:
            lower_cols = {c.lower(): c for c in df.columns}
            for candidate in self.LABEL_COLUMNS:
                if candidate in lower_cols:
                    label_col = lower_cols[candidate]
                    break

        return text_col, label_col

    def load_csv(self, file_path: str | Path) -> TextDataset:
        file_path = Path(file_path)
        encoding = self._detect_encoding(file_path)

        try:
            df = pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="utf-8", errors="ignore")

        text_col, label_col = self._detect_columns(df)

        texts = df[text_col].astype(str).fillna("").tolist()
        file_paths = [f"{file_path}:{i+2}" for i in range(len(texts))]
        labels = None
        if label_col and label_col in df.columns:
            labels = df[label_col].tolist()

        return TextDataset(
            file_paths=file_paths,
            texts=texts,
            labels=labels,
            text_column=text_col,
            label_column=label_col or "label",
        )

    def load_txt(self, file_path: str | Path) -> TextDataset:
        file_path = Path(file_path)
        encoding = self._detect_encoding(file_path)

        with open(file_path, "r", encoding=encoding, errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]

        return TextDataset(
            file_paths=[f"{file_path}:{i+1}" for i in range(len(lines))],
            texts=lines,
            labels=None,
        )

    def load_json(self, file_path: str | Path) -> TextDataset:
        file_path = Path(file_path)
        encoding = self._detect_encoding(file_path)

        df = pd.read_json(file_path, encoding=encoding, lines=True)
        if df.empty:
            df = pd.read_json(file_path, encoding=encoding)

        text_col, label_col = self._detect_columns(df)

        texts = df[text_col].astype(str).fillna("").tolist()
        file_paths = [f"{file_path}:{i+1}" for i in range(len(texts))]
        labels = None
        if label_col and label_col in df.columns:
            labels = df[label_col].tolist()

        return TextDataset(
            file_paths=file_paths,
            texts=texts,
            labels=labels,
            text_column=text_col,
            label_column=label_col or "label",
        )

    def load_directory(self, dir_path: str | Path) -> TextDataset:
        dir_path = Path(dir_path)
        all_texts: list[str] = []
        all_paths: list[str] = []
        all_labels: list = []
        has_labels = True

        csv_files = list(dir_path.glob("*.csv"))
        txt_files = list(dir_path.glob("*.txt"))
        json_files = list(dir_path.glob("*.json"))
        jsonl_files = list(dir_path.glob("*.jsonl"))

        for csv_file in csv_files:
            ds = self.load_csv(csv_file)
            all_texts.extend(ds.texts)
            all_paths.extend(ds.file_paths)
            if ds.labels is not None:
                all_labels.extend(ds.labels)
            else:
                has_labels = False

        for txt_file in txt_files:
            ds = self.load_txt(txt_file)
            all_texts.extend(ds.texts)
            all_paths.extend(ds.file_paths)
            has_labels = False

        for json_file in json_files + jsonl_files:
            ds = self.load_json(json_file)
            all_texts.extend(ds.texts)
            all_paths.extend(ds.file_paths)
            if ds.labels is not None:
                all_labels.extend(ds.labels)
            else:
                has_labels = False

        labels = all_labels if has_labels and len(all_labels) == len(all_texts) else None

        return TextDataset(
            file_paths=all_paths,
            texts=all_texts,
            labels=labels,
        )

    def load(self, path: str | Path) -> TextDataset:
        path = Path(path)
        if path.is_dir():
            return self.load_directory(path)

        suffix = path.suffix.lower()
        if suffix == ".csv":
            return self.load_csv(path)
        elif suffix == ".txt":
            return self.load_txt(path)
        elif suffix in (".json", ".jsonl"):
            return self.load_json(path)
        else:
            raise ValueError(f"不支持的文本文件格式: {suffix}")
