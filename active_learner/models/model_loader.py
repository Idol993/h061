from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np


ModelType = Literal["sklearn", "pytorch", "onnx"]


@dataclass
class LoadedModel:
    model: Any
    model_type: ModelType
    num_classes: int
    device: str = "cpu"
    feature_dim: int = 0


class ModelLoader:
    @staticmethod
    def detect_model_type(model_path: str | Path) -> ModelType:
        suffix = Path(model_path).suffix.lower()
        if suffix in (".pkl", ".joblib"):
            return "sklearn"
        elif suffix in (".pth", ".pt", ".bin"):
            return "pytorch"
        elif suffix == ".onnx":
            return "onnx"
        else:
            raise ValueError(f"无法识别的模型文件格式: {suffix}")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            return "cpu"
        return device

    @staticmethod
    def _load_sklearn(model_path: str | Path) -> Any:
        import joblib

        return joblib.load(model_path)

    @staticmethod
    def _load_pytorch(
        model_path: str | Path,
        num_classes: Optional[int] = None,
        device: str = "cpu",
    ) -> Any:
        try:
            import torch

            checkpoint = torch.load(model_path, map_location=device, weights_only=False)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                from torchvision import models

                if num_classes is None:
                    if hasattr(checkpoint, "num_classes"):
                        num_classes = checkpoint["num_classes"]
                    else:
                        num_classes = 10

                model = models.resnet18(weights=None)
                in_features = model.fc.in_features
                model.fc = torch.nn.Linear(in_features, num_classes)
                model.load_state_dict(state_dict)
            elif isinstance(checkpoint, torch.nn.Module):
                model = checkpoint
            else:
                from torchvision import models

                if num_classes is None:
                    num_classes = 10
                model = models.resnet18(weights=None)
                in_features = model.fc.in_features
                model.fc = torch.nn.Linear(in_features, num_classes)
                try:
                    model.load_state_dict(checkpoint)
                except Exception:
                    pass

            model.eval()
            if device.startswith("cuda") and torch.cuda.is_available():
                model = model.to(device)
            return model

        except ImportError:
            raise ImportError("PyTorch 未安装，无法加载 .pth 模型")

    @staticmethod
    def _load_onnx(model_path: str | Path) -> Any:
        try:
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            try:
                import torch

                if torch.cuda.is_available():
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            except ImportError:
                pass

            session = ort.InferenceSession(str(model_path), providers=providers)
            return session
        except ImportError:
            raise ImportError("onnxruntime 未安装，无法加载 .onnx 模型")

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        model_type: Optional[ModelType] = None,
        num_classes: int = 10,
        device: str = "auto",
    ) -> LoadedModel:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        if model_type is None:
            model_type = cls.detect_model_type(model_path)

        resolved_device = cls._resolve_device(device)

        if model_type == "sklearn":
            model = cls._load_sklearn(model_path)
        elif model_type == "pytorch":
            model = cls._load_pytorch(model_path, num_classes, resolved_device)
        elif model_type == "onnx":
            model = cls._load_onnx(model_path)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")

        feature_dim = 0
        if model_type == "pytorch":
            try:
                import torch

                modules = list(model.children())
                if modules and isinstance(modules[-1], torch.nn.Linear):
                    feature_dim = modules[-1].in_features
            except Exception:
                feature_dim = 0

        return LoadedModel(
            model=model,
            model_type=model_type,
            num_classes=num_classes,
            device=resolved_device,
            feature_dim=feature_dim,
        )
