from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


def main():
    out_path = Path(__file__).resolve().parent.parent / "checkpoints" / "dummy_model.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    n_samples = 200
    n_features = 384
    n_classes = 4

    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = rng.randint(0, n_classes, size=n_samples)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)

    joblib.dump(model, out_path)
    print(f"Dummy 模型已创建: {out_path}")
    print(f"模型特征维度: {n_features}, 类别数: {n_classes}")


if __name__ == "__main__":
    main()
