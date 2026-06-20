from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


@dataclass
class ImageDataset:
    file_paths: list[str] = field(default_factory=list)
    labels: Optional[list] = field(default=None)
    class_names: Optional[list[str]] = field(default=None)
    image_size: tuple[int, int] = (224, 224)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict:
        item = {"file_path": self.file_paths[idx]}
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


class ImageLoader:
    def __init__(self, image_size: tuple[int, int] = (224, 224)) -> None:
        self.image_size = image_size

    def _is_image(self, path: Path) -> bool:
        return path.suffix.lower() in IMAGE_EXTENSIONS

    def load_image_folder(self, dir_path: str | Path) -> ImageDataset:
        dir_path = Path(dir_path)
        class_names = sorted(
            [p.name for p in dir_path.iterdir() if p.is_dir()]
        )

        if class_names:
            file_paths: list[str] = []
            labels: list[int] = []
            for label_idx, class_name in enumerate(class_names):
                class_dir = dir_path / class_name
                for img_path in sorted(class_dir.rglob("*")):
                    if img_path.is_file() and self._is_image(img_path):
                        file_paths.append(str(img_path))
                        labels.append(label_idx)
            return ImageDataset(
                file_paths=file_paths,
                labels=labels,
                class_names=class_names,
                image_size=self.image_size,
            )
        else:
            file_paths = [
                str(p)
                for p in sorted(dir_path.rglob("*"))
                if p.is_file() and self._is_image(p)
            ]
            return ImageDataset(
                file_paths=file_paths,
                labels=None,
                class_names=None,
                image_size=self.image_size,
            )

    def load_csv(self, file_path: str | Path) -> ImageDataset:
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

        for candidate in ["path", "file_path", "image", "image_path", "file"]:
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
        if label_col:
            label_values = df[label_col].tolist()
            unique_labels = sorted(set(str(v) for v in label_values if pd.notna(v)))
            label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
            labels = [label_map.get(str(v), -1) if pd.notna(v) else -1 for v in label_values]
            class_names = unique_labels
        else:
            class_names = None

        return ImageDataset(
            file_paths=file_paths,
            labels=labels,
            class_names=class_names,
            image_size=self.image_size,
        )

    def load(self, path: str | Path) -> ImageDataset:
        path = Path(path)
        if path.is_dir():
            return self.load_image_folder(path)
        elif path.suffix.lower() == ".csv":
            return self.load_csv(path)
        else:
            raise ValueError(f"不支持的图像路径类型: {path}")

    def load_image_array(
        self,
        file_path: str | Path,
    ) -> Optional[np.ndarray]:
        try:
            from PIL import Image

            img = Image.open(file_path).convert("RGB")
            img = img.resize((self.image_size[1], self.image_size[0]))
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)
            return arr
        except Exception:
            return None

    def load_batch(self, file_paths: list[str | Path]) -> np.ndarray:
        arrays = []
        for p in file_paths:
            arr = self.load_image_array(p)
            if arr is None:
                arr = np.zeros((3, self.image_size[0], self.image_size[1]), dtype=np.float32)
            arrays.append(arr)
        return np.stack(arrays, axis=0)
