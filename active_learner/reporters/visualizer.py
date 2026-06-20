from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import numpy as np


class Visualizer:
    COLOR_LABELED = "#1f77b4"
    COLOR_UNLABELED = "#ff0000"

    def __init__(self, output_dir: str | Path = "outputs") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _pca_reduce(features: np.ndarray, n_components: int = 2) -> np.ndarray:
        if features.shape[1] <= n_components:
            return features[:, :n_components].astype(np.float32)

        try:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=n_components, random_state=42)
            return pca.fit_transform(features).astype(np.float32)
        except Exception:
            rng = np.random.RandomState(42)
            return rng.randn(len(features), n_components).astype(np.float32)

    def plot_feature_space(
        self,
        features: np.ndarray,
        labeled_mask: Optional[np.ndarray],
        selected_indices: Optional[np.ndarray],
        labels: Optional[np.ndarray] = None,
        filename: Optional[str] = None,
        title: str = "特征空间可视化",
    ) -> str:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            features = features.reshape(len(features), -1)

        features_2d = self._pca_reduce(features, 2)

        n = len(features_2d)
        is_labeled = np.zeros(n, dtype=bool)
        if labeled_mask is not None:
            is_labeled = np.asarray(labeled_mask)
        is_selected = np.zeros(n, dtype=bool)
        if selected_indices is not None:
            for idx in selected_indices:
                if 0 <= idx < n:
                    is_selected[idx] = True

        try:
            import plotly.graph_objects as go

            fig = go.Figure()

            labeled_indices = np.where(is_labeled)[0]
            if len(labeled_indices) > 0:
                hover_texts = [f"样本 {i}" for i in labeled_indices]
                if labels is not None:
                    hover_texts = [f"样本 {i}<br>标签: {labels[i]}" for i in labeled_indices]
                fig.add_trace(
                    go.Scatter(
                        x=features_2d[labeled_indices, 0].tolist(),
                        y=features_2d[labeled_indices, 1].tolist(),
                        mode="markers",
                        name="已标注",
                        marker=dict(
                            size=8,
                            color=self.COLOR_LABELED,
                            symbol="circle",
                        ),
                        text=hover_texts,
                        hoverinfo="text",
                    )
                )

            unlabeled_indices = np.where(~is_labeled)[0]
            if len(unlabeled_indices) > 0:
                selected_unlabeled = [i for i in unlabeled_indices if not is_selected[i]]
                if len(selected_unlabeled) > 0:
                    hover_texts = [f"样本 {i}" for i in selected_unlabeled]
                    fig.add_trace(
                        go.Scatter(
                            x=features_2d[selected_unlabeled, 0].tolist(),
                            y=features_2d[selected_unlabeled, 1].tolist(),
                            mode="markers",
                            name="未标注（未推荐",
                            marker=dict(
                                size=6,
                                color="lightgray",
                                symbol="circle-open",
                            ),
                            text=hover_texts,
                            hoverinfo="text",
                        )
                    )

            recommended_idx = np.where(is_selected & ~is_labeled)[0]
            if len(recommended_idx) > 0:
                hover_texts = [f"样本 {i}<br>推荐标注" for i in recommended_idx]
                fig.add_trace(
                    go.Scatter(
                        x=features_2d[recommended_idx, 0].tolist(),
                        y=features_2d[recommended_idx, 1].tolist(),
                        mode="markers",
                        name="推荐标注",
                        marker=dict(
                            size=10,
                            color="white",
                            line=dict(color=self.COLOR_UNLABELED, width=2),
                            symbol="circle-open",
                        ),
                        text=hover_texts,
                        hoverinfo="text",
                    )
                )

            fig.update_layout(
                title=title,
                xaxis_title="PCA 维度 1",
                yaxis_title="PCA 维度 2",
                template="plotly_white",
                legend=dict(x=0, y=1),
                margin=dict(l=40, r=40, t=60, b=40),
            )

            if filename is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"feature_space_{timestamp}.html"
            output_path = self.output_dir / filename
            fig.write_html(str(output_path))
            return str(output_path)

        except ImportError:
            return self._fallback_text_summary(features_2d, is_labeled, is_selected, title)

    def _fallback_text_summary(
        self,
        features_2d: np.ndarray,
        is_labeled: np.ndarray,
        is_selected: np.ndarray,
        title: str,
    ) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"feature_space_{timestamp}.txt"
        output_path = self.output_dir / filename

        n_labeled = int(is_labeled.sum())
        n_selected = int(is_selected.sum())
        n_total = len(features_2d)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"{title}\n")
            f.write(f"总样本数: {n_total}\n")
            f.write(f"已标注: {n_labeled}\n")
            f.write(f"待标注: {n_total - n_labeled}\n")
            f.write(f"推荐标注: {n_selected}\n")
            f.write("=" * 50 + "\n")
            f.write("(plotly 未安装，无法生成交互式图表)\n")

        return str(output_path)

    def plot_learning_curve(
        self,
        iterations: list[int] | np.ndarray,
        labeled_counts: list[int] | np.ndarray,
        accuracies: list[float] | np.ndarray,
        f1_scores: Optional[list[float] | np.ndarray] = None,
        filename: Optional[str] = None,
        title: str = "主动学习曲线",
    ) -> str:
        try:
            import plotly.graph_objects as go

            fig = go.Figure()

            x_values = list(labeled_counts)

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=list(accuracies),
                    mode="lines+markers",
                    name="准确率",
                    line=dict(color=self.COLOR_LABELED, width=2),
                    marker=dict(size=8),
                )
            )

            if f1_scores is not None:
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=list(f1_scores),
                        mode="lines+markers",
                        name="F1 (Macro)",
                        line=dict(color="#ff7f0e", width=2),
                        marker=dict(size=8),
                    )
                )

            fig.update_layout(
                title=title,
                xaxis_title="已标注样本数",
                yaxis_title="性能",
                yaxis=dict(range=[0, 1.05]),
                template="plotly_white",
                legend=dict(x=0, y=1),
                margin=dict(l=40, r=40, t=60, b=40),
            )

            if filename is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"learning_curve_{timestamp}.html"
            output_path = self.output_dir / filename
            fig.write_html(str(output_path))
            return str(output_path)

        except ImportError:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"learning_curve_{timestamp}.txt"
            output_path = self.output_dir / filename
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"{title}\n")
                f.write("=" * 50 + "\n")
                for i, (n, acc) in enumerate(zip(labeled_counts, accuracies)):
                    line = f"轮次 {i+1}: 标注 {n} 样本, 准确率 {acc:.4f}"
                    if f1_scores is not None:
                        line += f", F1 {f1_scores[i]:.4f}"
                    f.write(line + "\n")
            return str(output_path)
