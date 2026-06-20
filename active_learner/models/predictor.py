from __future__ import annotations

from typing import Optional

import numpy as np

from .model_loader import LoadedModel


class Predictor:
    def __init__(self, loaded_model: LoadedModel) -> None:
        self.loaded_model = loaded_model
        self.model = loaded_model.model
        self.model_type = loaded_model.model_type
        self.device = loaded_model.device
        self.num_classes = loaded_model.num_classes

    def _predict_sklearn(self, features: np.ndarray) -> np.ndarray:
        model = self.model

        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(features)
                return np.asarray(probs, dtype=np.float32)
            except Exception:
                pass

        if hasattr(model, "decision_function"):
            try:
                logits = model.decision_function(features)
                logits = np.asarray(logits, dtype=np.float32)
                if logits.ndim == 1:
                    logits = np.stack([-logits, logits], axis=1)
                exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
                return exp_logits / exp_logits.sum(axis=1, keepdims=True)
            except Exception:
                pass

        if hasattr(model, "predict"):
            preds = model.predict(features)
            preds = np.asarray(preds)
            n = len(preds)
            probs = np.zeros((n, self.num_classes), dtype=np.float32)
            for i, p in enumerate(preds):
                idx = int(p) if isinstance(p, (int, np.integer)) else 0
                idx = max(0, min(idx, self.num_classes - 1))
                probs[i, idx] = 1.0
            return probs

        rng = np.random.RandomState(42)
        return rng.dirichlet(np.ones(self.num_classes), size=len(features)).astype(np.float32)

    def _predict_pytorch(self, features: np.ndarray) -> np.ndarray:
        try:
            import torch

            model = self.model
            model.eval()

            features_tensor = torch.from_numpy(features).float()
            if self.device.startswith("cuda") and torch.cuda.is_available():
                features_tensor = features_tensor.to(self.device)

            all_probs = []
            batch_size = 32
            with torch.no_grad():
                for i in range(0, len(features_tensor), batch_size):
                    batch = features_tensor[i : i + batch_size]
                    outputs = model(batch)
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    if outputs.shape[1] != self.num_classes:
                        raise RuntimeError(
                            f"PyTorch 模型输出类别数({outputs.shape[1]})与配置 num_classes({self.num_classes})不匹配，"
                            f"请检查模型最后一层维度或配置中的 num_classes 参数"
                        )
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.append(probs.cpu().numpy())

            if all_probs:
                return np.concatenate(all_probs, axis=0).astype(np.float32)

            raise RuntimeError("PyTorch 推理输出为空")

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"PyTorch 模型推理失败: {e}") from e

    def _predict_onnx(self, features: np.ndarray) -> np.ndarray:
        try:
            session = self.model
            input_name = session.get_inputs()[0].name

            features = features.astype(np.float32)
            all_probs = []
            batch_size = 32

            for i in range(0, len(features), batch_size):
                batch = features[i : i + batch_size]
                outputs = session.run(None, {input_name: batch})
                probs = outputs[0]
                if probs.ndim == 2 and probs.shape[1] == self.num_classes:
                    pass
                elif probs.ndim == 1:
                    probs = np.stack([1 - probs, probs], axis=1)
                else:
                    exp_p = np.exp(probs - probs.max(axis=1, keepdims=True))
                    probs = exp_p / exp_p.sum(axis=1, keepdims=True)

                if probs.shape[1] != self.num_classes:
                    raise RuntimeError(
                        f"ONNX 模型输出类别数({probs.shape[1]})与配置 num_classes({self.num_classes})不匹配"
                    )

                all_probs.append(probs.astype(np.float32))

            if all_probs:
                return np.concatenate(all_probs, axis=0)

            raise RuntimeError("ONNX 推理输出为空")

        except Exception as e:
            raise RuntimeError(f"ONNX 模型推理失败: {e}") from e

    def predict_proba(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        features = np.asarray(features)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if self.model_type == "sklearn":
            probs = self._predict_sklearn(features)
        elif self.model_type == "pytorch":
            probs = self._predict_pytorch(features)
        elif self.model_type == "onnx":
            probs = self._predict_onnx(features)
        else:
            raise ValueError(f"未知模型类型: {self.model_type}")

        probs = np.asarray(probs, dtype=np.float32)
        if probs.shape[1] != self.num_classes:
            raise RuntimeError(
                f"模型输出类别数({probs.shape[1]})与配置 num_classes({self.num_classes})不匹配，"
                f"请检查模型最后一层维度或配置中的 num_classes 参数"
            )

        probs_sum = probs.sum(axis=1, keepdims=True)
        probs_sum = np.where(probs_sum == 0, 1.0, probs_sum)
        probs = probs / probs_sum

        return probs

    def predict(self, features: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(features)
        return np.argmax(probs, axis=1)
