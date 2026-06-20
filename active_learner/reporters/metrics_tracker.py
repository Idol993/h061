from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class IterationMetric:
    iteration: int
    labeled_count: int
    accuracy: float
    f1_macro: float
    selected_count: int
    model_path: Optional[str] = None
    extra: dict = field(default_factory=dict)


@dataclass
class RunRecord:
    run_id: str
    command: str
    start_time: str
    end_time: Optional[str] = None
    duration_sec: Optional[float] = None
    status: str = "running"
    config_path: Optional[str] = None
    config_summary: dict = field(default_factory=dict)
    model_path: Optional[str] = None
    model_type: Optional[str] = None
    num_classes: Optional[int] = None
    label_encoder_path: Optional[str] = None
    class_names: list = field(default_factory=list)
    sampler_strategy: Optional[str] = None
    sampler_uncertainty_method: Optional[str] = None
    sampler_uncertainty_weight: Optional[float] = None
    sampler_diversity_weight: Optional[float] = None
    budget: Optional[int] = None
    data_type: Optional[str] = None
    data_path: Optional[str] = None
    total_samples: Optional[int] = None
    labeled_samples: Optional[int] = None
    output_csv: Optional[str] = None
    output_visualization: Optional[str] = None
    output_learning_curve: Optional[str] = None
    output_dir: Optional[str] = None
    error_message: Optional[str] = None
    extra: dict = field(default_factory=dict)


class MetricsTracker:
    DEFAULT_FILENAME = "metrics_history.json"
    RUNS_FILENAME = "runs_history.json"

    def __init__(
        self,
        output_dir: str | Path = "outputs",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: list[IterationMetric] = []
        self.runs: list[RunRecord] = []

    def create_run(
        self,
        command: str,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> RunRecord:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run = RunRecord(
            run_id=run_id,
            command=command,
            start_time=datetime.datetime.now().isoformat(),
            config_path=config_path,
            **kwargs,
        )
        self.runs.append(run)
        return run

    def finish_run(
        self,
        run: RunRecord,
        status: str = "success",
        error_message: Optional[str] = None,
        **kwargs,
    ) -> RunRecord:
        run.end_time = datetime.datetime.now().isoformat()
        try:
            start_dt = datetime.datetime.fromisoformat(run.start_time)
            end_dt = datetime.datetime.fromisoformat(run.end_time)
            run.duration_sec = round((end_dt - start_dt).total_seconds(), 3)
        except Exception:
            run.duration_sec = None
        run.status = status
        run.error_message = error_message
        for k, v in kwargs.items():
            if hasattr(run, k):
                setattr(run, k, v)
            else:
                run.extra[k] = v
        self.save_runs()
        return run

    def runs_history_path(self) -> Path:
        return self.output_dir / self.RUNS_FILENAME

    def load_runs(self) -> list[RunRecord]:
        hp = self.runs_history_path()
        if not hp.exists():
            return self.runs
        with open(hp, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.runs = []
        import dataclasses as _dc

        field_names = {f.name for f in _dc.fields(RunRecord)}
        for d in data:
            run_kwargs = {k: v for k, v in d.items() if k in field_names}
            self.runs.append(RunRecord(**run_kwargs))
        return self.runs

    def save_runs(self) -> str:
        import dataclasses as _dc

        output_path = self.runs_history_path()
        data = []
        for r in self.runs:
            data.append(_dc.asdict(r))
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return str(output_path)

    def add(
        self,
        iteration: int,
        labeled_count: int,
        accuracy: float,
        f1_macro: float,
        selected_count: int,
        model_path: Optional[str] = None,
        **kwargs,
    ) -> IterationMetric:
        metric = IterationMetric(
            iteration=iteration,
            labeled_count=labeled_count,
            accuracy=float(accuracy),
            f1_macro=float(f1_macro),
            selected_count=selected_count,
            model_path=model_path,
            extra=dict(kwargs),
        )
        self.metrics.append(metric)
        return metric

    @property
    def iterations(self) -> list[int]:
        return [m.iteration for m in self.metrics]

    @property
    def labeled_counts(self) -> list[int]:
        return [m.labeled_count for m in self.metrics]

    @property
    def accuracies(self) -> list[float]:
        return [m.accuracy for m in self.metrics]

    @property
    def f1_scores(self) -> list[float]:
        return [m.f1_macro for m in self.metrics]

    @property
    def next_iteration(self) -> int:
        if not self.metrics:
            return 1
        return max(m.iteration for m in self.metrics) + 1

    def history_path(self) -> Path:
        return self.output_dir / self.DEFAULT_FILENAME

    def load_history(self) -> list[IterationMetric]:
        hp = self.history_path()
        if hp.exists():
            self.load(hp)
        return self.metrics

    def save_history(self) -> str:
        return self.save(self.DEFAULT_FILENAME)

    def save(self, filename: Optional[str] = None) -> str:
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"

        output_path = self.output_dir / filename

        data = []
        for m in self.metrics:
            d = {
                "iteration": m.iteration,
                "labeled_count": m.labeled_count,
                "accuracy": m.accuracy,
                "f1_macro": m.f1_macro,
                "selected_count": m.selected_count,
                "model_path": m.model_path,
                "extra": m.extra,
            }
            data.append(d)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return str(output_path)

    def load(self, path: str | Path) -> list[IterationMetric]:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.metrics = []
        for d in data:
            self.metrics.append(
                IterationMetric(
                    iteration=d["iteration"],
                    labeled_count=d["labeled_count"],
                    accuracy=d["accuracy"],
                    f1_macro=d["f1_macro"],
                    selected_count=d.get("selected_count", 0),
                    model_path=d.get("model_path"),
                    extra=d.get("extra", {}),
                )
            )
        return self.metrics

    def summary(self) -> str:
        if not self.metrics:
            return "暂无记录"

        lines = ["主动学习指标跟踪"]
        lines.append("=" * 60)
        lines.append(f"{'轮次':<6}{'标注数':<10}{'准确率':<12}{'F1(Macro)':<12}{'选择数':<10}")
        lines.append("-" * 60)

        for m in self.metrics:
            lines.append(
                f"{m.iteration:<6}{m.labeled_count:<10}{m.accuracy:<12.4f}{m.f1_macro:<12.4f}{m.selected_count:<10}"
            )

        if len(self.metrics) > 1:
            lines.append("-" * 60)
            acc_gain = self.metrics[-1].accuracy - self.metrics[0].accuracy
            f1_gain = self.metrics[-1].f1_macro - self.metrics[0].f1_macro
            lines.append(
                f"性能提升: 准确率 +{acc_gain:.4f}, F1 +{f1_gain:.4f}"
            )

        return "\n".join(lines)
