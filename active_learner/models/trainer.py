from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class TrainResult:
    accuracy: float
    f1_macro: float
    epochs_trained: int
    model_path: str
    history: dict
    label_encoder_path: Optional[str] = None


class Trainer:
    def __init__(
        self,
        num_classes: int,
        checkpoint_dir: str | Path = "checkpoints",
        device: str = "auto",
    ) -> None:
        self.num_classes = num_classes
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._resolve_device(device)

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

    def _get_latest_checkpoint(self) -> Optional[Path]:
        ckpt_files = list(self.checkpoint_dir.glob("*.pth")) + list(self.checkpoint_dir.glob("*.pkl"))
        if not ckpt_files:
            return None
        return max(ckpt_files, key=lambda p: p.stat().st_mtime)

    def get_latest_label_encoder(self) -> Optional[Path]:
        le_files = list(self.checkpoint_dir.glob("label_encoder*.joblib"))
        if not le_files:
            return None
        return max(le_files, key=lambda p: p.stat().st_mtime)

    @staticmethod
    def load_label_encoder(path: str | Path):
        import joblib

        return joblib.load(path)

    def _fit_label_encoder(self, y_train: np.ndarray, y_val: Optional[np.ndarray] = None):
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        all_y = list(y_train)
        if y_val is not None:
            all_y.extend(list(y_val))
        le.fit(all_y)
        return le

    def _save_label_encoder(self, le) -> str:
        import joblib

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        le_path = self.checkpoint_dir / f"label_encoder_{timestamp}.joblib"
        joblib.dump(le, le_path)
        return str(le_path)

    def resume(self) -> Optional[Path]:
        return self._get_latest_checkpoint()

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
        from sklearn.metrics import accuracy_score, f1_score

        accuracy = float(accuracy_score(y_true, y_pred))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        return accuracy, f1_macro

    def _train_sklearn(
        self,
        base_model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 10,
        early_stopping_patience: int = 3,
    ) -> TrainResult:
        from sklearn.base import clone

        model = clone(base_model) if hasattr(base_model, "get_params") else base_model

        model.fit(X_train, y_train)

        if X_val is not None and y_val is not None:
            y_pred = model.predict(X_val)
            acc, f1 = self._compute_metrics(y_val, y_pred)
        else:
            y_pred_train = model.predict(X_train)
            acc, f1 = self._compute_metrics(y_train, y_pred_train)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.checkpoint_dir / f"model_sklearn_{timestamp}.pkl"

        import joblib

        joblib.dump(model, model_path)

        return TrainResult(
            accuracy=acc,
            f1_macro=f1,
            epochs_trained=1,
            model_path=str(model_path),
            history={"accuracy": [acc], "f1_macro": [f1]},
        )

    def _train_pytorch(
        self,
        base_model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        early_stopping_patience: int = 3,
        freeze_backbone: bool = True,
    ) -> TrainResult:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if freeze_backbone:
            try:
                children = list(base_model.children())
                if len(children) > 1:
                    for child in children[:-1]:
                        for param in child.parameters():
                            param.requires_grad = False
            except Exception:
                pass

        model = base_model
        model.to(self.device)
        model.train()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
        )

        X_tensor = torch.from_numpy(X_train).float()
        y_tensor = torch.from_numpy(y_train).long()
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            Xv_tensor = torch.from_numpy(X_val).float()
            yv_tensor = torch.from_numpy(y_val).long()
            val_dataset = TensorDataset(Xv_tensor, yv_tensor)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        history: dict[str, list[float]] = {"accuracy": [], "f1_macro": [], "loss": []}
        best_f1 = 0.0
        patience_counter = 0
        best_model_path: Optional[str] = None
        epochs_trained = 0

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_X)

            avg_loss = total_loss / len(X_train)
            epochs_trained = epoch + 1

            if val_loader is not None:
                model.eval()
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        outputs = model(batch_X)
                        if isinstance(outputs, (tuple, list)):
                            outputs = outputs[0]
                        preds = outputs.argmax(dim=1).cpu().numpy()
                        all_preds.extend(preds)
                        all_labels.extend(batch_y.numpy())
                acc, f1 = self._compute_metrics(np.array(all_labels), np.array(all_preds))
            else:
                model.eval()
                all_preds = []
                eval_dataset = TensorDataset(X_tensor, y_tensor)
                eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)
                with torch.no_grad():
                    for batch_X, _ in eval_loader:
                        batch_X = batch_X.to(self.device)
                        outputs = model(batch_X)
                        if isinstance(outputs, (tuple, list)):
                            outputs = outputs[0]
                        preds = outputs.argmax(dim=1).cpu().numpy()
                        all_preds.extend(preds)
                acc, f1 = self._compute_metrics(np.asarray(y_train), np.array(all_preds))

            history["loss"].append(avg_loss)
            history["accuracy"].append(acc)
            history["f1_macro"].append(f1)

            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                model_path = self.checkpoint_dir / f"model_pytorch_{timestamp}.pth"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "num_classes": self.num_classes,
                        "epoch": epoch + 1,
                    },
                    model_path,
                )
                best_model_path = str(model_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break

        if best_model_path is None:
            model_path = self.checkpoint_dir / f"model_pytorch_{timestamp}.pth"
            torch.save({"model_state_dict": model.state_dict(), "num_classes": self.num_classes}, model_path)
            best_model_path = str(model_path)

        return TrainResult(
            accuracy=history["accuracy"][-1],
            f1_macro=best_f1,
            epochs_trained=epochs_trained,
            model_path=best_model_path,
            history=history,
        )

    def train(
        self,
        base_model,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        early_stopping_patience: int = 3,
        freeze_backbone: bool = True,
        existing_label_encoder=None,
    ) -> TrainResult:
        y_train = np.asarray(y_train)
        if X_val is not None and y_val is not None:
            y_val = np.asarray(y_val)

        if existing_label_encoder is not None:
            try:
                _ = existing_label_encoder.transform(y_train)
                if y_val is not None:
                    _ = existing_label_encoder.transform(y_val)
                le = existing_label_encoder
            except Exception:
                le = self._fit_label_encoder(y_train, y_val)
        else:
            le = self._fit_label_encoder(y_train, y_val)

        y_train_enc = le.transform(y_train)
        y_val_enc = le.transform(y_val) if y_val is not None else None

        unique_classes = list(le.classes_)
        if len(unique_classes) != self.num_classes:
            self.num_classes = len(unique_classes)

        le_path = self._save_label_encoder(le)

        if model_type == "sklearn":
            result = self._train_sklearn(
                base_model=base_model,
                X_train=X_train,
                y_train=y_train_enc,
                X_val=X_val,
                y_val=y_val_enc,
                epochs=epochs,
                early_stopping_patience=early_stopping_patience,
            )
            result.label_encoder_path = le_path
            return result
        elif model_type == "pytorch":
            result = self._train_pytorch(
                base_model=base_model,
                X_train=X_train,
                y_train=y_train_enc,
                X_val=X_val,
                y_val=y_val_enc,
                epochs=epochs,
                learning_rate=learning_rate,
                early_stopping_patience=early_stopping_patience,
                freeze_backbone=freeze_backbone,
            )
            result.label_encoder_path = le_path
            return result
        else:
            if hasattr(base_model, "fit") or hasattr(base_model, "train"):
                result = self._train_sklearn(
                    base_model=base_model,
                    X_train=X_train,
                    y_train=y_train_enc,
                    X_val=X_val,
                    y_val=y_val_enc,
                    epochs=epochs,
                    early_stopping_patience=early_stopping_patience,
                )
                result.label_encoder_path = le_path
                return result
            raise ValueError(f"不支持重训练的模型类型: {model_type}")
