from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


@dataclass
class AudioDataset:
    file_paths: list[str] = field(default_factory=list)
    labels: Optional[list] = field(default=None)
    class_names: Optional[list[str]] = field(default=None)
    sr: int = 22050
    n_mfcc: int = 40

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict:
        item = {"file_path": self.file_paths[idx]}
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


class AudioLoader:
    def __init__(self, sr: int = 22050, n_mfcc: int = 40) -> None:
        self.sr = sr
        self.n_mfcc = n_mfcc

    def _is_audio(self, path: Path) -> bool:
        return path.suffix.lower() in AUDIO_EXTENSIONS

    def load_folder(self, dir_path: str | Path) -> AudioDataset:
        dir_path = Path(dir_path)
        class_names = sorted(
            [p.name for p in dir_path.iterdir() if p.is_dir()]
        )

        if class_names:
            file_paths: list[str] = []
            labels: list[int] = []
            for label_idx, class_name in enumerate(class_names):
                class_dir = dir_path / class_name
                for audio_path in sorted(class_dir.rglob("*")):
                    if audio_path.is_file() and self._is_audio(audio_path):
                        file_paths.append(str(audio_path))
                        labels.append(label_idx)
            return AudioDataset(
                file_paths=file_paths,
                labels=labels,
                class_names=class_names,
                sr=self.sr,
                n_mfcc=self.n_mfcc,
            )
        else:
            file_paths = [
                str(p)
                for p in sorted(dir_path.rglob("*"))
                if p.is_file() and self._is_audio(p)
            ]
            return AudioDataset(
                file_paths=file_paths,
                labels=None,
                class_names=None,
                sr=self.sr,
                n_mfcc=self.n_mfcc,
            )

    def load_csv(self, file_path: str | Path) -> AudioDataset:
        import pandas as pd
        import chardet

        file_path = Path(file_path)
        with open(file_path, "rb") as f:
            raw = f.read(65536)
        encoding = chardet.detect(raw).get("encoding", "utf-8") or "utf-8"

        try:
            df = pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="utf-8", errors="ignore")

        path_col = None
        label_col = None
        lower_cols = {c.lower(): c for c in df.columns}

        for candidate in ["path", "file_path", "audio", "audio_path", "file"]:
            if candidate in lower_cols:
                path_col = lower_cols[candidate]
                break
        if path_col is None:
            path_col = df.columns[0]

        for candidate in ["label", "category", "class", "target", "y"]:
            if candidate in lower_cols:
                label_col = lower_cols[candidate]
                break

        file_paths = df[path_col].astype(str).tolist()
        labels = None
        class_names = None
        if label_col:
            label_values = df[label_col].tolist()
            unique_labels = sorted(set(str(v) for v in label_values if pd.notna(v)))
            label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
            labels = [label_map.get(str(v), -1) if pd.notna(v) else -1 for v in label_values]
            class_names = unique_labels

        return AudioDataset(
            file_paths=file_paths,
            labels=labels,
            class_names=class_names,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
        )

    def load(self, path: str | Path) -> AudioDataset:
        path = Path(path)
        if path.is_dir():
            return self.load_folder(path)
        elif path.suffix.lower() == ".csv":
            return self.load_csv(path)
        else:
            raise ValueError(f"不支持的音频路径类型: {path}")

    def extract_mfcc(self, file_path: str | Path) -> Optional[np.ndarray]:
        try:
            import librosa

            y, sr = librosa.load(file_path, sr=self.sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            return mfcc.astype(np.float32)
        except Exception:
            return None

    def extract_mfcc_batch(
        self,
        file_paths: list[str | Path],
        max_length: int = 500,
    ) -> np.ndarray:
        features = []
        for p in file_paths:
            mfcc = self.extract_mfcc(p)
            if mfcc is None:
                mfcc = np.zeros((self.n_mfcc, max_length), dtype=np.float32)
            if mfcc.shape[1] < max_length:
                pad = np.zeros((self.n_mfcc, max_length - mfcc.shape[1]), dtype=np.float32)
                mfcc = np.concatenate([mfcc, pad], axis=1)
            elif mfcc.shape[1] > max_length:
                mfcc = mfcc[:, :max_length]
            features.append(mfcc)
        return np.stack(features, axis=0)
