from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Optional

import click
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .data_loaders import AudioLoader, ImageLoader, TextLoader
from .models import ModelLoader, Predictor, Trainer
from .reporters import CSVExporter, MetricsTracker, Visualizer
from .samplers import DiversitySampler, HybridSampler, QBCSampler, UncertaintySampler
from .selectors import BatchSelector
from .utils import ConfigParser, FeatureExtractor

console = Console()


def _resolve_device(device_str: str) -> str:
    if device_str == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device_str


def _load_dataset(config, console: Console):
    data_type = config.data.data_type
    data_path = Path(config.data.data_path)

    if data_type == "text":
        loader = TextLoader(
            text_column=config.data.text_column,
            label_column=config.data.label_column,
        )
        dataset = loader.load(data_path)
        file_paths = dataset.file_paths if dataset.file_paths else [str(i) for i in range(len(dataset.texts))]
        raw_items = dataset.texts
        labels = dataset.labels
    elif data_type == "image":
        loader = ImageLoader(image_size=tuple(config.data.image_size))
        dataset = loader.load(data_path)
        file_paths = dataset.file_paths
        raw_items = file_paths
        labels = dataset.labels
    elif data_type == "audio":
        loader = AudioLoader(sr=config.data.audio_sr, n_mfcc=config.data.audio_mfcc)
        dataset = loader.load(data_path)
        file_paths = dataset.file_paths
        raw_items = file_paths
        labels = dataset.labels
    else:
        raise click.ClickException(f"不支持的数据类型: {data_type}")

    console.print(f"[green]✓[/green] 加载 {data_type} 数据集: [bold]{len(dataset)}[/bold] 个样本")
    return dataset, file_paths, raw_items, labels


def _extract_features(
    raw_items, data_type: str, device: str, progress: Progress, task_id,
    batch_size: int = 32,
) -> np.ndarray:
    extractor = FeatureExtractor(data_type=data_type, device=device)
    n = len(raw_items)

    if data_type == "text":
        batch_size = 32
        all_feats = []
        for i in range(0, n, batch_size):
            batch = raw_items[i : i + batch_size]
            feats = extractor.extract_text(list(batch), batch_size=batch_size)
            all_feats.append(feats)
            progress.update(task_id, advance=len(batch))
        return np.concatenate(all_feats, axis=0) if all_feats else np.zeros((0, 384), dtype=np.float32)
    elif data_type == "image":
        all_feats = []
        for i in range(0, n, batch_size):
            batch = raw_items[i : i + batch_size]
            feats = extractor.extract_images(list(batch), batch_size=batch_size)
            all_feats.append(feats)
            progress.update(task_id, advance=len(batch))
        return np.concatenate(all_feats, axis=0) if all_feats else np.zeros((0, 2048), dtype=np.float32)
    elif data_type == "audio":
        loader = AudioLoader()
        all_feats = []
        for i in range(0, n, batch_size):
            batch = raw_items[i : i + batch_size]
            mfcc = loader.extract_mfcc_batch(list(batch))
            feats = extractor.extract_audio_features(mfcc)
            all_feats.append(feats)
            progress.update(task_id, advance=len(batch))
        return np.concatenate(all_feats, axis=0) if all_feats else np.zeros((0, 40), dtype=np.float32)
    else:
        return np.zeros((n, 128), dtype=np.float32)


def _predict_probs(
    predictor: Predictor, features: np.ndarray, batch_size: int, progress: Progress, task_id,
) -> np.ndarray:
    n = len(features)
    all_probs = []
    for i in range(0, n, batch_size):
        batch = features[i : i + batch_size]
        probs = predictor.predict_proba(batch)
        all_probs.append(probs)
        progress.update(task_id, advance=len(batch))
    return np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, predictor.num_classes), dtype=np.float32)


def _display_table(
    indices: np.ndarray, file_paths: list[str], uncertainty_scores: np.ndarray,
    diversity_scores: np.ndarray, reasons: list[str], top_k: int = 20,
) -> Table:
    table = Table(title="Top-K 推荐样本", show_lines=False)
    table.add_column("#", justify="right", style="cyan")
    table.add_column("文件路径", style="white", no_wrap=False)
    table.add_column("不确定性", justify="right", style="magenta")
    table.add_column("多样性", justify="right", style="green")
    table.add_column("推荐理由", style="yellow")

    display_k = min(top_k, len(indices))
    for rank in range(display_k):
        idx = int(indices[rank])
        fp = str(file_paths[idx]) if idx < len(file_paths) else str(idx)
        if len(fp) > 60:
            fp = "..." + fp[-57:]
        u_score = float(uncertainty_scores[idx]) if idx < len(uncertainty_scores) else 0.0
        d_score = float(diversity_scores[idx]) if idx < len(diversity_scores) else 0.0
        reason = reasons[idx] if idx < len(reasons) else ""

        u_str = f"{u_score:.4f}"
        if u_score > 0.8:
            u_str = f"[bold red]{u_score:.4f}[/bold red]"
        elif u_score > 0.6:
            u_str = f"[yellow]{u_score:.4f}[/yellow]"
        d_str = f"{d_score:.4f}"
        if d_score > 0.7:
            d_str = f"[bold green]{d_score:.4f}[/bold green]"

        table.add_row(str(rank + 1), fp, u_str, d_str, reason)

    return table


@click.group(help="主动学习样本选择工具")
def cli():
    pass


@cli.command(help="从模型推理，选择最值得标注的样本")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="YAML 配置文件路径")
@click.option("--budget", "-b", type=int, help="每轮选择样本数，覆盖配置文件")
@click.option("--strategy", "-s",
              type=click.Choice(["uncertainty", "diversity", "hybrid", "qbc"]),
              help="采样策略")
@click.option("--output-dir", "-o", type=click.Path(), help="输出目录")
def select(config: str, budget: Optional[int], strategy: Optional[str], output_dir: Optional[str]):
    cfg = ConfigParser.load(config)
    if budget is not None:
        cfg.sampler.budget = budget
    if strategy is not None:
        cfg.sampler.strategy = strategy
    if output_dir is not None:
        cfg.output.output_dir = output_dir

    output_dir_path = Path(cfg.output.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    console.print(Panel.fit(
        f"[bold cyan]主动学习样本选择[/bold cyan]\n"
        f"策略: [yellow]{cfg.sampler.strategy}[/yellow]\n"
        f"预算: [yellow]{cfg.sampler.budget}[/yellow] 样本\n"
        f"数据: [yellow]{cfg.data.data_type}[/yellow]",
        border_style="cyan",
    ))

    device = _resolve_device(cfg.model.device)
    console.print(f"使用设备: [bold]{device}[/bold]")

    dataset, file_paths, raw_items, labels = _load_dataset(cfg, console)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        feat_task = progress.add_task("[cyan]提取特征...", total=len(dataset))
        features = _extract_features(
            raw_items, cfg.data.data_type, device, progress, feat_task, cfg.model.batch_size,
        )

    console.print(f"[green]✓[/green] 特征提取完成，特征维度: [bold]{features.shape[1]}[/bold]")

    model_path = Path(cfg.model.model_path)
    loaded_model = ModelLoader.load(
        model_path=model_path,
        model_type=cfg.model.model_type,
        num_classes=cfg.model.num_classes,
        device=device,
    )
    console.print(f"[green]✓[/green] 加载模型: [bold]{loaded_model.model_type}[/bold]")

    predictor = Predictor(loaded_model)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        pred_task = progress.add_task("[magenta]模型推理...", total=len(features))
        probs = _predict_probs(predictor, features, cfg.model.batch_size, progress, pred_task,
        )

    console.print(f"[green]✓[/green] 推理完成")

    strategy_name = cfg.sampler.strategy
    budget = cfg.sampler.budget
    u_scores = np.zeros(len(features), dtype=np.float32)
    d_scores = np.zeros(len(features), dtype=np.float32)
    hybrid_scores = np.zeros(len(features), dtype=np.float32)
    reasons = [""] * len(features)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        samp_task = progress.add_task(f"[yellow]执行采样策略: {strategy_name}...", total=100)

        if strategy_name == "uncertainty":
            sampler = UncertaintySampler(method=cfg.sampler.uncertainty_method)
            result = sampler.score(probs)
            u_scores = result.scores
            d_scores = np.zeros_like(u_scores)
            hybrid_scores = u_scores
            for i in range(len(reasons)):
                reasons[i] = f"不确定性({cfg.sampler.uncertainty_method})"
        elif strategy_name == "diversity":
            sampler = DiversitySampler(
                num_clusters=cfg.sampler.num_clusters or budget,
                seed=cfg.seed,
            )
            div_result = sampler.score(features, budget)
            d_scores = div_result.scores
            u_scores = np.zeros_like(d_scores)
            hybrid_scores = d_scores
            for i in range(len(reasons)):
                reasons[i] = "多样性(KMeans聚类)"
        elif strategy_name == "hybrid":
            sampler = HybridSampler(
                uncertainty_method=cfg.sampler.uncertainty_method,
                uncertainty_weight=cfg.sampler.uncertainty_weight,
                diversity_weight=cfg.sampler.diversity_weight,
                num_clusters=cfg.sampler.num_clusters or budget,
                seed=cfg.seed,
            )
            h_result = sampler.score(probs, features, budget)
            u_scores = h_result.uncertainty_scores
            d_scores = h_result.diversity_scores
            hybrid_scores = h_result.scores
            for i in range(len(reasons)):
                reasons[i] = HybridSampler.get_reason(
                    float(u_scores[i]),
                    float(d_scores[i]),
                    cfg.sampler.uncertainty_weight,
                    cfg.sampler.diversity_weight,
                )
        elif strategy_name == "qbc":
            sampler = QBCSampler(seed=cfg.seed)
            X_train = None
            y_train = None
            if labels is not None and len(labels) > 0:
                valid_mask = np.array([l is not None and l != -1 for l in labels])
                if valid_mask.any():
                    X_train = features[valid_mask]
                    y_train = np.array([l for l, v in zip(labels, valid_mask) if v])
            qbc_result = sampler.score(features, X_train, y_train, predictor)
            hybrid_scores = qbc_result.scores
            u_scores = hybrid_scores
            d_scores = np.zeros_like(hybrid_scores)
            for i in range(len(reasons)):
                reasons[i] = "委员会投票(QBC)"
        else:
            raise click.ClickException(f"未知策略: {strategy_name}")

        progress.update(samp_task, advance=50)

        selector = BatchSelector(budget=budget, seed=cfg.seed)
        selection = selector.select(
            scores=hybrid_scores,
            features=features,
            uncertainty_scores=u_scores,
            diversity_scores=d_scores,
            file_paths=file_paths,
            probs=probs,
            u_weight=cfg.sampler.uncertainty_weight,
            d_weight=cfg.sampler.diversity_weight,
        )

        progress.update(samp_task, advance=50)

    console.print(f"[green]✓[/green] 采样完成，选择 [bold]{len(selection)}[/bold] 个样本")

    table = _display_table(
        selection.indices, file_paths, u_scores, d_scores, reasons, cfg.output.top_k_display,
    )
    console.print(table)

    exporter = CSVExporter(output_dir=cfg.output.output_dir)
    csv_path = exporter.export(selection)
    console.print(f"[green]✓[/green] CSV 已导出: [bold]{csv_path}[/bold]")

    if cfg.output.export_visualization:
        visualizer = Visualizer(output_dir=cfg.output.output_dir)
        labeled_mask = None
        if labels is not None:
            labeled_mask = np.array([l is not None and l != -1 for l in labels])
        feat_path = visualizer.plot_feature_space(
            features=features,
            labeled_mask=labeled_mask,
            selected_indices=selection.indices,
            labels=np.array(labels) if labels is not None else None,
        )
        console.print(f"[green]✓[/green] 特征空间图: [bold]{feat_path}[/bold]")

    console.print(Panel.fit(
        f"[bold green]采样完成![/bold green]\n"
        f"已选择 [bold cyan]{len(selection)}[/bold cyan] 个样本用于标注\n"
        f"输出目录: [bold]{cfg.output.output_dir}[/bold]",
        border_style="green",
    ))


@cli.command(help="查看历史迭代指标和学习曲线")
@click.option("--metrics", "-m", type=click.Path(exists=True), help="指标 JSON 文件路径")
@click.option("--output-dir", "-o", type=click.Path(), help="输出目录")
def review(metrics: Optional[str], output_dir: Optional[str]):
    metrics_path = metrics
    out_dir = Path(output_dir or "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    if metrics_path:
        tracker = MetricsTracker(output_dir=out_dir)
        tracker.load(metrics_path)
        console.print(tracker.summary())

        if tracker.metrics:
            visualizer = Visualizer(output_dir=out_dir)
            curve_path = visualizer.plot_learning_curve(
                iterations=tracker.iterations,
                labeled_counts=tracker.labeled_counts,
                accuracies=tracker.accuracies,
                f1_scores=tracker.f1_scores,
            )
            console.print(f"[green]✓[/green] 学习曲线: [bold]{curve_path}[/bold]")
    else:
        console.print("[yellow]提示:[/yellow] 请指定 --metrics 参数查看历史指标")


@cli.command(help="用新标注数据重训练模型，并进行下一轮采样")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="YAML 配置文件路径")
@click.option("--labels", "-l", required=True, type=click.Path(exists=True), help="新标注 CSV (file_path, label)")
@click.option("--resume/--no-resume", default=False, help="从最新 checkpoint 恢复")
@click.option("--output-dir", "-o", type=click.Path(), help="输出目录")
def iterate(config: str, labels: str, resume: bool, output_dir: Optional[str]):
    cfg = ConfigParser.load(config)
    if output_dir is not None:
        cfg.output.output_dir = output_dir

    output_dir_path = Path(cfg.output.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    console.print(Panel.fit(
        "[bold cyan]迭代: 标注 → 重训练 → 下一轮采样[/bold cyan]",
        border_style="cyan",
    ))

    device = _resolve_device(cfg.model.device)

    dataset, file_paths, raw_items, existing_labels = _load_dataset(cfg, console)

    import pandas as pd
    import chardet

    labels_path = Path(labels)
    with open(labels_path, "rb") as f:
        enc = chardet.detect(f.read(65536)).get("encoding", "utf-8") or "utf-8"
    try:
        labels_df = pd.read_csv(labels_path, encoding=enc)
    except UnicodeDecodeError:
        labels_df = pd.read_csv(labels_path, encoding="utf-8", errors="ignore")

    console.print(f"[green]✓[/green] 加载新标注数据: [bold]{len(labels_df)}[/bold] 条")

    path_col = None
    label_col = None
    lower_cols = {c.lower(): c for c in labels_df.columns}
    for cand in ["path", "file_path", "file", "image", "audio", "sample"]:
        if cand in lower_cols:
            path_col = lower_cols[cand]
            break
    if path_col is None:
        path_col = labels_df.columns[0]
    for cand in ["label", "category", "class", "target", "y"]:
        if cand in lower_cols:
            label_col = lower_cols[cand]
            break
    if label_col is None:
        label_col = labels_df.columns[-1]

    new_labels_map = {}
    for _, row in labels_df.iterrows():
        fp = str(row[path_col])
        lbl = row[label_col]
        new_labels_map[fp] = lbl

    all_labels = [None] * len(file_paths)
    for i, fp in enumerate(file_paths):
        if fp in new_labels_map:
            all_labels[i] = new_labels_map[fp]

    if existing_labels is not None:
        for i in range(len(existing_labels)):
            if existing_labels[i] is not None and existing_labels[i] != -1:
                all_labels[i] = existing_labels[i]

    with Progress(console=console) as progress:
        feat_task = progress.add_task("[cyan]提取特征...", total=len(dataset))
        features = _extract_features(raw_items, cfg.data.data_type, device, progress, feat_task, cfg.model.batch_size)

    labeled_mask = np.array([l is not None and l != -1 for l in all_labels])
    X_labeled = features[labeled_mask]
    y_labeled = np.array([l for l, v in zip(all_labels, labeled_mask) if v])

    console.print(f"已标注样本数: [bold]{X_labeled.shape[0]}[/bold] / [bold]{len(features)}[/bold]")

    model_path = Path(cfg.model.model_path)
    if resume:
        trainer = Trainer(
            num_classes=cfg.model.num_classes,
            checkpoint_dir=cfg.train.checkpoint_dir,
            device=device,
        )
        ckpt = trainer.resume()
        if ckpt:
            console.print(f"[green]✓[/green] 从 checkpoint 恢复: [bold]{ckpt}[/bold]")
            model_path = ckpt

    loaded_model = ModelLoader.load(
        model_path=model_path,
        model_type=cfg.model.model_type,
        num_classes=cfg.model.num_classes,
        device=device,
    )
    console.print(f"[green]✓[/green] 加载基线模型")

    if cfg.train.enabled and len(X_labeled) > 10:
        console.print("[cyan]开始重训练...")
        trainer = Trainer(
            num_classes=cfg.model.num_classes,
            checkpoint_dir=cfg.train.checkpoint_dir,
            device=device,
        )

        from sklearn.model_selection import train_test_split

        if len(X_labeled) > 50:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_labeled, y_labeled, test_size=0.2, random_state=cfg.seed,
            )
        else:
            X_tr, y_tr = X_labeled, y_labeled
            X_val, y_val = None, None

        train_result = trainer.train(
            base_model=loaded_model.model,
            model_type=loaded_model.model_type,
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_val,
            y_val=y_val,
            epochs=cfg.train.epochs,
            learning_rate=cfg.train.learning_rate,
            early_stopping_patience=cfg.train.early_stopping_patience,
            freeze_backbone=cfg.train.freeze_backbone,
        )

        console.print(
            f"[green]✓[/green] 重训练完成: 准确率 [bold]{train_result.accuracy:.4f}[/bold], "
            f"F1 [bold]{train_result.f1_macro:.4f}[/bold]"
        )
        console.print(f"新模型保存: [bold]{train_result.model_path}[/bold]")

        tracker = MetricsTracker(output_dir=cfg.output.output_dir)
        tracker.add(
            iteration=1,
            labeled_count=len(X_labeled),
            accuracy=train_result.accuracy,
            f1_macro=train_result.f1_macro,
            selected_count=cfg.sampler.budget,
            model_path=train_result.model_path,
        )
        metrics_path = tracker.save()
        console.print(f"[green]✓[/green] 指标保存: [bold]{metrics_path}[/bold]")

        if cfg.output.export_visualization:
            visualizer = Visualizer(output_dir=cfg.output.output_dir)
            curve_path = visualizer.plot_learning_curve(
                iterations=tracker.iterations,
                labeled_counts=tracker.labeled_counts,
                accuracies=tracker.accuracies,
                f1_scores=tracker.f1_scores,
            )
            console.print(f"[green]✓[/green] 学习曲线: [bold]{curve_path}[/bold]")

        new_model_path = train_result.model_path
    else:
        if not cfg.train.enabled:
            console.print("[yellow]训练已禁用[/yellow]")
        else:
            console.print("[yellow]标注样本不足，跳过训练[/yellow]")
        new_model_path = str(model_path)

    console.print("[cyan]开始下一轮采样...")
    loaded_new = ModelLoader.load(
        model_path=new_model_path,
        model_type=cfg.model.model_type,
        num_classes=cfg.model.num_classes,
        device=device,
    )
    predictor = Predictor(loaded_new)

    with Progress(console=console) as progress:
        pred_task = progress.add_task("[magenta]模型推理...", total=len(features))
        probs = _predict_probs(predictor, features, cfg.model.batch_size, progress, pred_task)

    budget = cfg.sampler.budget
    strategy_name = cfg.sampler.strategy
    u_scores = np.zeros(len(features), dtype=np.float32)
    d_scores = np.zeros(len(features), dtype=np.float32)
    hybrid_scores = np.zeros(len(features), dtype=np.float32)
    reasons = [""] * len(features)

    if strategy_name == "uncertainty":
        sampler = UncertaintySampler(method=cfg.sampler.uncertainty_method)
        result = sampler.score(probs)
        u_scores = result.scores
        hybrid_scores = u_scores
        for i in range(len(reasons)):
            reasons[i] = f"不确定性({cfg.sampler.uncertainty_method})"
    elif strategy_name == "diversity":
        sampler = DiversitySampler(num_clusters=cfg.sampler.num_clusters or budget, seed=cfg.seed)
        div_result = sampler.score(features, budget)
        d_scores = div_result.scores
        hybrid_scores = d_scores
        for i in range(len(reasons)):
            reasons[i] = "多样性(KMeans)"
    else:
        sampler = HybridSampler(
            uncertainty_method=cfg.sampler.uncertainty_method,
            uncertainty_weight=cfg.sampler.uncertainty_weight,
            diversity_weight=cfg.sampler.diversity_weight,
            num_clusters=cfg.sampler.num_clusters or budget,
            seed=cfg.seed,
        )
        h_result = sampler.score(probs, features, budget)
        u_scores = h_result.uncertainty_scores
        d_scores = h_result.diversity_scores
        hybrid_scores = h_result.scores
        for i in range(len(reasons)):
            reasons[i] = HybridSampler.get_reason(
                float(u_scores[i]),
                float(d_scores[i]),
                cfg.sampler.uncertainty_weight,
                cfg.sampler.diversity_weight,
            )

    selector = BatchSelector(budget=budget, seed=cfg.seed)
    selection = selector.select(
        scores=hybrid_scores,
        features=features,
        uncertainty_scores=u_scores,
        diversity_scores=d_scores,
        file_paths=file_paths,
        probs=probs,
        u_weight=cfg.sampler.uncertainty_weight,
        d_weight=cfg.sampler.diversity_weight,
    )

    table = _display_table(
        selection.indices, file_paths, u_scores, d_scores, reasons, cfg.output.top_k_display,
    )
    console.print(table)

    exporter = CSVExporter(output_dir=cfg.output.output_dir)
    csv_path = exporter.export(selection)
    console.print(f"[green]✓[/green] 下一轮待标注 CSV: [bold]{csv_path}[/bold]")

    console.print(Panel.fit(
        "[bold green]迭代完成![/bold green]\n"
        f"输出目录: [bold]{cfg.output.output_dir}[/bold]",
        border_style="green",
    ))


@cli.command(help="生成示例配置文件")
@click.option("--output", "-o", default="active_learner/data/sample_config.yaml", help="输出路径")
def init_config(output: str):
    ConfigParser.dump_example(output)
    console.print(f"[green]✓[/green] 示例配置已生成: [bold]{output}[/bold]")


def main():
    cli()


if __name__ == "__main__":
    main()
