from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
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
from .pool import SamplePool, PoolDelta, PoolSnapshot
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


def _is_valid_label(l) -> bool:
    if l is None:
        return False
    if isinstance(l, (int, np.integer)):
        return l != -1
    if isinstance(l, float):
        if np.isnan(l):
            return False
        return l != -1.0
    if isinstance(l, str):
        return l.strip() != ""
    try:
        if pd.isna(l):
            return False
    except Exception:
        pass
    return True


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

    tracker = MetricsTracker(output_dir=cfg.output.output_dir)
    tracker.load_runs()

    class_names: list = []
    feat_path: Optional[str] = None
    csv_path: Optional[str] = None
    run_status = "failed"
    err_msg: Optional[str] = None

    try:
        run = tracker.create_run(
            command="select",
            config_path=str(config),
            sampler_strategy=cfg.sampler.strategy,
            sampler_uncertainty_method=cfg.sampler.uncertainty_method,
            sampler_uncertainty_weight=cfg.sampler.uncertainty_weight,
            sampler_diversity_weight=cfg.sampler.diversity_weight,
            budget=cfg.sampler.budget,
            output_dir=str(cfg.output.output_dir),
            config_summary={
                "data_type": cfg.data.data_type,
                "num_classes": cfg.model.num_classes,
                "seed": cfg.seed,
            },
        )
    
        console.print(Panel.fit(
            f"[bold cyan]主动学习样本选择[/bold cyan]\n"
            f"Run ID: [dim]{run.run_id}[/dim]\n"
            f"策略: [yellow]{cfg.sampler.strategy}[/yellow]\n"
            f"预算: [yellow]{cfg.sampler.budget}[/yellow] 样本\n"
            f"数据: [yellow]{cfg.data.data_type}[/yellow]",
            border_style="cyan",
        ))
    
        device = _resolve_device(cfg.model.device)
        console.print(f"使用设备: [bold]{device}[/bold]")
    
        dataset, file_paths, raw_items, labels = _load_dataset(cfg, console)
        n_total = len(raw_items)
        n_labeled = sum(1 for l in (labels or []) if _is_valid_label(l))
        run.data_type = cfg.data.data_type
        run.data_path = str(cfg.data.data_path)
        run.total_samples = n_total
        run.labeled_samples = n_labeled
        tracker.save_runs()

        pool = SamplePool(output_dir=cfg.output.output_dir)
        pool.load()
        pool.init_from_file_paths(file_paths, labels)
        pool.save()

        available_indices = pool.get_available_indices(file_paths)
        n_available = len(available_indices)
        snap_before = pool.snapshot()
        console.print(
            f"样本池: [bold]{n_total}[/bold] 总计 | "
            f"[green]{snap_before.labeled}[/green] 已标注 | "
            f"[yellow]{snap_before.pending}[/yellow] 待标注 | "
            f"[cyan]{n_available}[/cyan] 可采样 | "
            f"[dim]{snap_before.skipped}[/dim] 已跳过"
        )
    
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
        run.model_path = str(model_path)
        run.model_type = loaded_model.model_type
        run.num_classes = loaded_model.num_classes
        tracker.save_runs()
    
        trainer_aux = Trainer(
            num_classes=cfg.model.num_classes,
            checkpoint_dir=cfg.train.checkpoint_dir,
            device="cpu",
        )
        latest_le = trainer_aux.get_latest_label_encoder()
        if latest_le is not None:
            try:
                le = Trainer.load_label_encoder(latest_le)
                class_names = list(le.classes_)
                run.label_encoder_path = str(latest_le)
                run.class_names = class_names
                tracker.save_runs()
            except Exception:
                pass
    
        predictor = Predictor(loaded_model)
    
        try:
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
        except Exception as e:
            console.print(f"[bold red]✗ 模型推理失败:[/bold red] {e}")
            console.print("[yellow]已停止执行，未导出待标注 CSV，请检查模型输入维度/格式是否匹配。[/yellow]")
            raise click.ClickException(f"模型推理失败: {e}")
    
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
                    valid_mask = np.array([_is_valid_label(l) for l in labels])
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

            available_set = set(available_indices)
            for i in range(len(hybrid_scores)):
                if i not in available_set:
                    hybrid_scores[i] = -1e9

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
        run.output_csv = csv_path

        selected_fps = [file_paths[idx] for idx in selection.indices if idx < len(file_paths)]
        new_pending, repeated = pool.mark_pending(selected_fps)
        pool.last_export_csv = csv_path
        pool.save()
        if repeated > 0:
            console.print(f"[yellow]⚠[/yellow] {repeated} 个样本已在待标注状态（重复推荐）")
        console.print(f"[green]✓[/green] 样本池: [bold]{new_pending}[/bold] 个新标记为待标注")
    
        if cfg.output.export_visualization:
            visualizer = Visualizer(output_dir=cfg.output.output_dir)
            labeled_mask = None
            if labels is not None:
                labeled_mask = np.array([_is_valid_label(l) for l in labels])
            feat_path = visualizer.plot_feature_space(
                features=features,
                labeled_mask=labeled_mask,
                selected_indices=selection.indices,
                labels=np.array(labels) if labels is not None else None,
            )
            console.print(f"[green]✓[/green] 特征空间图: [bold]{feat_path}[/bold]")
            run.output_visualization = feat_path
    
        run_status = "success"
    
        console.print(Panel.fit(
            f"[bold green]采样完成![/bold green]\n"
            f"已选择 [bold cyan]{len(selection)}[/bold cyan] 个样本用于标注\n"
            f"输出目录: [bold]{cfg.output.output_dir}[/bold]",
            border_style="green",
        ))
    finally:
        if "run" in locals():
            pool_snap = pool.snapshot() if "pool" in locals() else None
            pool_extra = {}
            if pool_snap is not None:
                pool_extra["pool_unlabeled"] = pool_snap.unlabeled
                pool_extra["pool_pending"] = pool_snap.pending
                pool_extra["pool_labeled"] = pool_snap.labeled
                pool_extra["pool_skipped"] = pool_snap.skipped
            tracker.finish_run(
                run,
                status=run_status,
                error_message=err_msg,
                output_csv=csv_path,
                output_visualization=feat_path,
                class_names=class_names,
                **pool_extra,
            )


@cli.command(help="查看历史迭代指标和学习曲线")
@click.option("--metrics", "-m", type=click.Path(exists=True), help="指标 JSON 文件路径")
@click.option("--output-dir", "-o", type=click.Path(), help="输出目录")
def review(metrics: Optional[str], output_dir: Optional[str]):
    out_dir = Path(output_dir or "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = MetricsTracker(output_dir=out_dir)

    if metrics:
        tracker.load(metrics)
    else:
        hp = tracker.history_path()
        if hp.exists():
            tracker.load_history()
            console.print(f"[dim]自动加载历史指标: {hp}[/dim]")
        else:
            console.print("[yellow]提示:[/yellow] 未找到历史指标文件，请先运行 iterate 或指定 --metrics 参数")

    if tracker.metrics:
        console.print(tracker.summary())

        visualizer = Visualizer(output_dir=out_dir)
        curve_path = visualizer.plot_learning_curve(
            iterations=tracker.iterations,
            labeled_counts=tracker.labeled_counts,
            accuracies=tracker.accuracies,
            f1_scores=tracker.f1_scores,
        )
        console.print(f"[green]✓[/green] 学习曲线: [bold]{curve_path}[/bold]")

    tracker.load_runs()
    if tracker.runs:
        console.print(Panel.fit(
            "[bold cyan]历史实验 Run 记录[/bold cyan]",
            border_style="cyan",
        ))

        table = Table(show_lines=False)
        table.add_column("Run ID", style="cyan")
        table.add_column("Command", style="white")
        table.add_column("状态", style="white")
        table.add_column("开始时间", style="white")
        table.add_column("耗时(s)", justify="right", style="white")
        table.add_column("策略", style="yellow")
        table.add_column("预算", justify="right", style="yellow")
        table.add_column("输出CSV", style="green")
        table.add_column("模型路径", style="magenta")

        for run in tracker.runs:
            status_str = run.status or "unknown"
            if status_str == "success":
                status_display = f"[green]{status_str}[/green]"
            elif status_str == "failed":
                status_display = f"[red]{status_str}[/red]"
            elif status_str == "running":
                status_display = f"[yellow]{status_str}[/yellow]"
            else:
                status_display = status_str

            duration_str = ""
            if run.duration_sec is not None:
                duration_str = f"{run.duration_sec:.1f}"

            table.add_row(
                str(run.run_id),
                str(run.command or ""),
                status_display,
                str(run.start_time or ""),
                duration_str,
                str(run.sampler_strategy or ""),
                str(run.budget or ""),
                str(run.output_csv or ""),
                str(run.model_path or ""),
            )

        console.print(table)

        has_pool_data = any(
            r.extra.get("pool_unlabeled") is not None or r.extra.get("pool_pending") is not None
            for r in tracker.runs
        )
        if has_pool_data:
            console.print(Panel.fit(
                "[bold cyan]每轮样本池变化[/bold cyan]",
                border_style="cyan",
            ))
            pool_table = Table(show_lines=False)
            pool_table.add_column("Run ID", style="cyan")
            pool_table.add_column("Command", style="white")
            pool_table.add_column("已标注", justify="right", style="green")
            pool_table.add_column("待标注", justify="right", style="yellow")
            pool_table.add_column("未标注", justify="right", style="cyan")
            pool_table.add_column("已跳过", justify="right", style="dim")
            for run in tracker.runs:
                ex = run.extra or {}
                if ex.get("pool_unlabeled") is None and ex.get("pool_pending") is None:
                    continue
                pool_table.add_row(
                    str(run.run_id),
                    str(run.command or ""),
                    str(ex.get("pool_labeled", "")),
                    str(ex.get("pool_pending", "")),
                    str(ex.get("pool_unlabeled", "")),
                    str(ex.get("pool_skipped", "")),
                )
            console.print(pool_table)


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

    tracker = MetricsTracker(output_dir=cfg.output.output_dir)
    tracker.load_runs()

    class_names: list = []
    csv_path: Optional[str] = None
    curve_path: Optional[str] = None
    feat_path: Optional[str] = None
    final_model_path: Optional[str] = None
    total_samples: int = 0
    labeled_samples: int = 0
    run_status = "failed"
    err_msg: Optional[str] = None

    try:
        run = tracker.create_run(
            command="iterate",
            config_path=str(config),
            sampler_strategy=cfg.sampler.strategy,
            sampler_uncertainty_method=cfg.sampler.uncertainty_method,
            sampler_uncertainty_weight=cfg.sampler.uncertainty_weight,
            sampler_diversity_weight=cfg.sampler.diversity_weight,
            budget=cfg.sampler.budget,
            output_dir=str(cfg.output.output_dir),
            config_summary={
                "data_type": cfg.data.data_type,
                "num_classes": cfg.model.num_classes,
                "seed": cfg.seed,
            },
        )

        console.print(Panel.fit(
            f"[bold cyan]迭代: 标注 → 重训练 → 下一轮采样[/bold cyan]\n"
            f"Run ID: [dim]{run.run_id}[/dim]\n"
            f"策略: [yellow]{cfg.sampler.strategy}[/yellow]\n"
            f"预算: [yellow]{cfg.sampler.budget}[/yellow] 样本\n"
            f"数据: [yellow]{cfg.data.data_type}[/yellow]",
            border_style="cyan",
        ))

        device = _resolve_device(cfg.model.device)

        dataset, file_paths, raw_items, existing_labels = _load_dataset(cfg, console)
        n_total = len(raw_items)
        n_labeled = sum(1 for l in (existing_labels or []) if _is_valid_label(l))
        total_samples = n_total
        labeled_samples = n_labeled
        run.data_type = cfg.data.data_type
        run.data_path = str(cfg.data.data_path)
        run.total_samples = n_total
        run.labeled_samples = n_labeled
        tracker.save_runs()

        pool = SamplePool(output_dir=cfg.output.output_dir)
        pool.load()
        pool.init_from_file_paths(file_paths, existing_labels)
        pool.save()

        snap_before = pool.snapshot()
        console.print(
            f"样本池: [bold]{n_total}[/bold] 总计 | "
            f"[green]{snap_before.labeled}[/green] 已标注 | "
            f"[yellow]{snap_before.pending}[/yellow] 待标注 | "
            f"[dim]{snap_before.skipped}[/dim] 已跳过"
        )

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
            fp = str(row[path_col]).replace("\\", "/")
            lbl = row[label_col]
            new_labels_map[fp] = lbl

        recovered, dedup_count, n_external, external_paths = pool.apply_labels(
            new_labels_map, file_paths,
        )
        pool.save()
        if dedup_count > 0:
            console.print(f"[yellow]去重:[/yellow] 标注 CSV 中 [bold]{dedup_count}[/bold] 条重复路径，按最后一条生效")
        console.print(
            f"[green]✓[/green] 样本池更新: [bold]{recovered}[/bold] 个待标注→已标注"
        )
        if n_external > 0:
            console.print(f"[yellow]⚠[/yellow] [bold]{n_external}[/bold] 条标注路径不在数据集中（外部路径）:")
            for ep in external_paths[:5]:
                console.print(f"  [dim]- {ep}[/dim]")
            if len(external_paths) > 5:
                console.print(f"  [dim]... 共 {len(external_paths)} 条[/dim]")

        all_labels = [None] * len(file_paths)
        for i, fp in enumerate(file_paths):
            norm_fp = fp.replace("\\", "/")
            if norm_fp in new_labels_map:
                all_labels[i] = new_labels_map[norm_fp]

        if existing_labels is not None:
            for i in range(len(existing_labels)):
                if _is_valid_label(existing_labels[i]):
                    all_labels[i] = existing_labels[i]

        with Progress(console=console) as progress:
            feat_task = progress.add_task("[cyan]提取特征...", total=len(dataset))
            features = _extract_features(raw_items, cfg.data.data_type, device, progress, feat_task, cfg.model.batch_size)

        labeled_mask = np.array([_is_valid_label(l) for l in all_labels])
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
        run.model_path = str(model_path)
        run.model_type = loaded_model.model_type
        run.num_classes = loaded_model.num_classes
        tracker.save_runs()

        if cfg.train.enabled and len(X_labeled) > 10:
            console.print("[cyan]开始重训练...")
            trainer = Trainer(
                num_classes=cfg.model.num_classes,
                checkpoint_dir=cfg.train.checkpoint_dir,
                device=device,
            )

            existing_le = None
            latest_le_path = trainer.get_latest_label_encoder()
            if latest_le_path is not None:
                try:
                    existing_le = Trainer.load_label_encoder(latest_le_path)
                    console.print(f"[green]✓[/green] 复用已有类别映射: [bold]{latest_le_path}[/bold] ({len(existing_le.classes_)} 类)")
                except Exception as e:
                    console.print(f"[yellow]警告: 加载已有 label_encoder 失败，将重新生成: {e}[/yellow]")

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
                existing_label_encoder=existing_le,
            )

            console.print(
                f"[green]✓[/green] 重训练完成: 准确率 [bold]{train_result.accuracy:.4f}[/bold], "
                f"F1 [bold]{train_result.f1_macro:.4f}[/bold]"
            )
            console.print(f"新模型保存: [bold]{train_result.model_path}[/bold]")
            if train_result.label_encoder_path:
                console.print(f"类别映射保存: [bold]{train_result.label_encoder_path}[/bold]")
                try:
                    le = Trainer.load_label_encoder(train_result.label_encoder_path)
                    class_names = list(le.classes_)
                    run.label_encoder_path = str(train_result.label_encoder_path)
                    run.class_names = class_names
                    tracker.save_runs()
                except Exception:
                    pass

            tracker_metrics = MetricsTracker(output_dir=cfg.output.output_dir)
            tracker_metrics.load_history()
            current_iter = tracker_metrics.next_iteration
            tracker_metrics.add(
                iteration=current_iter,
                labeled_count=len(X_labeled),
                accuracy=train_result.accuracy,
                f1_macro=train_result.f1_macro,
                selected_count=cfg.sampler.budget,
                model_path=train_result.model_path,
            )
            metrics_path = tracker_metrics.save_history()
            console.print(f"[green]✓[/green] 第 [bold]{current_iter}[/bold] 轮指标保存: [bold]{metrics_path}[/bold]")

            if cfg.output.export_visualization:
                visualizer = Visualizer(output_dir=cfg.output.output_dir)
                curve_path = visualizer.plot_learning_curve(
                    iterations=tracker_metrics.iterations,
                    labeled_counts=tracker_metrics.labeled_counts,
                    accuracies=tracker_metrics.accuracies,
                    f1_scores=tracker_metrics.f1_scores,
                )
                console.print(f"[green]✓[/green] 学习曲线: [bold]{curve_path}[/bold]")
                run.output_learning_curve = curve_path
                tracker.save_runs()

            new_model_path = train_result.model_path
        else:
            if not cfg.train.enabled:
                console.print("[yellow]训练已禁用[/yellow]")
            else:
                console.print("[yellow]标注样本不足，跳过训练[/yellow]")
            new_model_path = str(model_path)

        final_model_path = new_model_path

        console.print("[cyan]开始下一轮采样...")
        loaded_new = ModelLoader.load(
            model_path=new_model_path,
            model_type=cfg.model.model_type,
            num_classes=cfg.model.num_classes,
            device=device,
        )
        predictor = Predictor(loaded_new)

        try:
            with Progress(console=console) as progress:
                pred_task = progress.add_task("[magenta]模型推理...", total=len(features))
                probs = _predict_probs(predictor, features, cfg.model.batch_size, progress, pred_task)
        except Exception as e:
            console.print(f"[bold red]✗ 下一轮采样推理失败:[/bold red] {e}")
            console.print("[yellow]已停止执行，未导出待标注 CSV，请检查模型。[/yellow]")
            raise click.ClickException(f"下一轮采样推理失败: {e}")

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
        elif strategy_name == "qbc":
            sampler = QBCSampler(seed=cfg.seed)
            valid_mask = np.array([_is_valid_label(l) for l in all_labels])
            X_train_qbc = features[valid_mask] if valid_mask.any() else None
            y_train_qbc = np.array([l for l, v in zip(all_labels, valid_mask) if v]) if valid_mask.any() else None
            qbc_result = sampler.score(features, X_train_qbc, y_train_qbc, predictor)
            hybrid_scores = qbc_result.scores
            u_scores = hybrid_scores
            d_scores = np.zeros_like(hybrid_scores)
            for i in range(len(reasons)):
                reasons[i] = "委员会投票(QBC)"
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

        available_indices_iter = pool.get_available_indices(file_paths)
        available_set_iter = set(available_indices_iter)
        for i in range(len(hybrid_scores)):
            if i not in available_set_iter:
                hybrid_scores[i] = -1e9

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
        run.output_csv = csv_path

        selected_fps_iter = [file_paths[idx] for idx in selection.indices if idx < len(file_paths)]
        new_pending_iter, repeated_iter = pool.mark_pending(selected_fps_iter)
        pool.last_export_csv = csv_path
        pool.save()
        if repeated_iter > 0:
            console.print(f"[yellow]⚠[/yellow] {repeated_iter} 个样本已在待标注状态（重复推荐）")
        console.print(f"[green]✓[/green] 样本池: [bold]{new_pending_iter}[/bold] 个新标记为待标注")

        if cfg.output.export_visualization:
            visualizer = Visualizer(output_dir=cfg.output.output_dir)
            labeled_mask_viz = np.array([_is_valid_label(l) for l in all_labels])
            feat_path = visualizer.plot_feature_space(
                features=features,
                labeled_mask=labeled_mask_viz,
                selected_indices=selection.indices,
                labels=np.array(all_labels) if all_labels is not None else None,
            )
            console.print(f"[green]✓[/green] 特征空间图: [bold]{feat_path}[/bold]")
            run.output_visualization = feat_path

        run_status = "success"

        console.print(Panel.fit(
            "[bold green]迭代完成![/bold green]\n"
            f"输出目录: [bold]{cfg.output.output_dir}[/bold]",
            border_style="green",
        ))
    finally:
        if "run" in locals():
            pool_snap = pool.snapshot() if "pool" in locals() else None
            pool_extra = {}
            if pool_snap is not None:
                pool_extra["pool_unlabeled"] = pool_snap.unlabeled
                pool_extra["pool_pending"] = pool_snap.pending
                pool_extra["pool_labeled"] = pool_snap.labeled
                pool_extra["pool_skipped"] = pool_snap.skipped
            tracker.finish_run(
                run,
                status=run_status,
                error_message=err_msg,
                output_csv=csv_path,
                output_visualization=feat_path,
                output_learning_curve=curve_path,
                model_path=final_model_path,
                class_names=class_names,
                total_samples=total_samples,
                labeled_samples=labeled_samples,
                **pool_extra,
            )


@cli.command(help="预检查模式：验证数据/模型/配置是否就绪，不执行训练和导出")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="YAML 配置文件路径")
@click.option("--labels", "-l", type=click.Path(exists=True), help="标注 CSV 路径（检查标注文件时必传）")
def dry_run(config: str, labels: Optional[str]):
    cfg = ConfigParser.load(config)
    checks_passed = 0
    checks_failed = 0
    checks_warned = 0

    console.print(Panel.fit(
        "[bold cyan]预检查模式 (dry-run)[/bold cyan]",
        border_style="cyan",
    ))

    console.print("\n[bold]1. 数据文件检查[/bold]")
    data_path = Path(cfg.data.data_path)
    if data_path.exists():
        try:
            dataset, file_paths, raw_items, existing_labels = _load_dataset(cfg, console)
            n_total = len(raw_items)
            n_labeled = sum(1 for l in (existing_labels or []) if _is_valid_label(l))
            console.print(f"  [green]✓[/green] 数据文件: [bold]{data_path}[/bold] ({n_total} 个样本, {n_labeled} 个已标注)")
            checks_passed += 1
        except Exception as e:
            console.print(f"  [red]✗[/red] 数据文件加载失败: {e}")
            checks_failed += 1
            n_total = 0
            n_labeled = 0
    else:
        console.print(f"  [red]✗[/red] 数据文件不存在: [bold]{data_path}[/bold]")
        checks_failed += 1
        n_total = 0
        n_labeled = 0

    if cfg.data.labeled_path:
        labeled_path = Path(cfg.data.labeled_path)
        if labeled_path.exists():
            console.print(f"  [green]✓[/green] 已标注文件: [bold]{labeled_path}[/bold]")
            checks_passed += 1
        else:
            console.print(f"  [yellow]⚠[/yellow] 已标注文件不存在: [bold]{labeled_path}[/bold]")
            checks_warned += 1

    console.print("\n[bold]2. 标注文件检查[/bold]")
    if labels:
        labels_path = Path(labels)
        if labels_path.exists():
            try:
                import chardet as _cd
                with open(labels_path, "rb") as f:
                    enc = _cd.detect(f.read(65536)).get("encoding", "utf-8") or "utf-8"
                labels_df = pd.read_csv(labels_path, encoding=enc)
                n_new = len(labels_df)

                lower_cols = {c.lower(): c for c in labels_df.columns}
                path_col = None
                label_col = None
                for cand in ["path", "file_path", "file", "image", "audio", "sample"]:
                    if cand in lower_cols:
                        path_col = lower_cols[cand]
                        break
                if path_col is None and len(labels_df.columns) > 0:
                    path_col = labels_df.columns[0]
                for cand in ["label", "category", "class", "target", "y"]:
                    if cand in lower_cols:
                        label_col = lower_cols[cand]
                        break
                if label_col is None and len(labels_df.columns) > 1:
                    label_col = labels_df.columns[-1]

                if path_col and label_col:
                    unique_labels = labels_df[label_col].dropna().unique()
                    console.print(f"  [green]✓[/green] 标注文件: [bold]{labels_path}[/bold] ({n_new} 条, {len(unique_labels)} 个类别: {list(unique_labels[:5])}{'...' if len(unique_labels) > 5 else ''})")
                    checks_passed += 1
                else:
                    console.print(f"  [red]✗[/red] 标注文件列名无法识别: {list(labels_df.columns)}")
                    checks_failed += 1
            except Exception as e:
                console.print(f"  [red]✗[/red] 标注文件读取失败: {e}")
                checks_failed += 1
        else:
            console.print(f"  [red]✗[/red] 标注文件不存在: [bold]{labels_path}[/bold]")
            checks_failed += 1
    else:
        console.print("  [dim]未指定 --labels，跳过标注文件检查[/dim]")

    console.print("\n[bold]3. 模型文件检查[/bold]")
    model_path = Path(cfg.model.model_path)
    if model_path.exists():
        try:
            loaded_model = ModelLoader.load(
                model_path=model_path,
                model_type=cfg.model.model_type,
                num_classes=cfg.model.num_classes,
                device="cpu",
            )
            console.print(f"  [green]✓[/green] 模型文件: [bold]{model_path}[/bold] (类型: {loaded_model.model_type}, 类别数: {loaded_model.num_classes})")
            checks_passed += 1

            if loaded_model.model_type == "sklearn" and hasattr(loaded_model.model, "n_classes_"):
                real_n = loaded_model.model.n_classes_
                if real_n != cfg.model.num_classes:
                    console.print(f"  [yellow]⚠[/yellow] 模型实际类别数({real_n})与配置 num_classes({cfg.model.num_classes})不一致")
                    checks_warned += 1
        except Exception as e:
            console.print(f"  [red]✗[/red] 模型加载失败: {e}")
            checks_failed += 1
    else:
        console.print(f"  [red]✗[/red] 模型文件不存在: [bold]{model_path}[/bold]")
        checks_failed += 1

    console.print("\n[bold]4. 类别映射检查[/bold]")
    trainer = Trainer(
        num_classes=cfg.model.num_classes,
        checkpoint_dir=cfg.train.checkpoint_dir,
        device="cpu",
    )
    latest_le = trainer.get_latest_label_encoder()
    le_classes: Optional[list] = None
    if latest_le is not None:
        try:
            le = Trainer.load_label_encoder(latest_le)
            le_classes = list(le.classes_)
            console.print(f"  [green]✓[/green] 类别映射: [bold]{latest_le}[/bold] ({len(le.classes_)} 类: {list(le.classes_)})")
            checks_passed += 1
        except Exception as e:
            console.print(f"  [yellow]⚠[/yellow] 类别映射加载失败: {e}")
            checks_warned += 1
    else:
        console.print("  [dim]无已有类别映射，首次训练将自动生成[/dim]")
        checks_passed += 1

    num_classes_cfg = cfg.model.num_classes
    labels_classes: Optional[list] = None
    if labels:
        labels_path = Path(labels)
        try:
            import chardet as _cd2
            with open(labels_path, "rb") as f:
                enc2 = _cd2.detect(f.read(65536)).get("encoding", "utf-8") or "utf-8"
            labels_df_check = pd.read_csv(labels_path, encoding=enc2)
            lower_cols2 = {c.lower(): c for c in labels_df_check.columns}
            label_col2 = None
            for cand in ["label", "category", "class", "target", "y"]:
                if cand in lower_cols2:
                    label_col2 = lower_cols2[cand]
                    break
            if label_col2 is None and len(labels_df_check.columns) > 1:
                label_col2 = labels_df_check.columns[-1]
            if label_col2 is not None:
                valid_labels = [l for l in labels_df_check[label_col2] if _is_valid_label(l)]
                labels_classes = sorted(list(set(valid_labels)))
        except Exception:
            pass

    if labels_classes is None and labels:
        console.print("  [yellow]⚠[/yellow] 无法从标注 CSV 提取类别，跳过标注类别比较")
        checks_warned += 1
    if le_classes is None and latest_le is not None:
        console.print("  [yellow]⚠[/yellow] 已有 label_encoder 无法加载，跳过映射类别比较")
        checks_warned += 1

    tri_parts_ok = True
    if labels_classes is not None and le_classes is not None:
        if num_classes_cfg != len(labels_classes) or len(labels_classes) != len(le_classes):
            console.print(f"  [red]✗ 类别数量不一致[/red]: 配置{num_classes_cfg}个 vs 标注{len(labels_classes)}个 vs 已有映射{len(le_classes)}个")
            checks_failed += 1
            tri_parts_ok = False
        labels_set = set(labels_classes)
        le_set = set(le_classes)
        if labels_set != le_set:
            only_labels = labels_set - le_set
            only_le = le_set - labels_set
            if only_labels:
                console.print(f"  [red]✗[/red] 仅标注有、映射无的类别: {list(only_labels)}")
            if only_le:
                console.print(f"  [red]✗[/red] 仅映射有、标注无的类别: {list(only_le)}")
            checks_failed += 1
            tri_parts_ok = False
        if tri_parts_ok:
            console.print("  [green]✓ 类别映射三方一致[/green]")
            checks_passed += 1
    elif labels_classes is not None and le_classes is None:
        console.print("  [yellow]⚠[/yellow] 无已有 label_encoder，仅检查配置与标注")
        checks_warned += 1
        if num_classes_cfg != len(labels_classes):
            console.print(f"  [red]✗ 类别数量不一致[/red]: 配置{num_classes_cfg}个 vs 标注{len(labels_classes)}个")
            checks_failed += 1
        else:
            console.print("  [green]✓ 配置与标注类别一致[/green]")
            checks_passed += 1
    elif labels_classes is None and le_classes is not None:
        console.print("  [yellow]⚠[/yellow] 未传标注 CSV，仅检查配置与已有映射")
        checks_warned += 1
        if num_classes_cfg != len(le_classes):
            console.print(f"  [red]✗ 类别数量不一致[/red]: 配置{num_classes_cfg}个 vs 已有映射{len(le_classes)}个")
            checks_failed += 1
        else:
            console.print("  [green]✓ 配置与已有映射类别一致[/green]")
            checks_passed += 1

    console.print("\n[bold]5. 输出目录检查[/bold]")
    output_dir = Path(cfg.output.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"  [green]✓[/green] 输出目录: [bold]{output_dir}[/bold]")
        checks_passed += 1
    except Exception as e:
        console.print(f"  [red]✗[/red] 输出目录创建失败: {e}")
        checks_failed += 1

    checkpoint_dir = Path(cfg.train.checkpoint_dir)
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"  [green]✓[/green] Checkpoint 目录: [bold]{checkpoint_dir}[/bold]")
        checks_passed += 1
    except Exception as e:
        console.print(f"  [red]✗[/red] Checkpoint 目录创建失败: {e}")
        checks_failed += 1

    console.print("\n[bold]6. 历史指标检查[/bold]")
    tracker = MetricsTracker(output_dir=cfg.output.output_dir)
    tracker.load_history()
    if tracker.metrics:
        console.print(f"  [green]✓[/green] 已有 {len(tracker.metrics)} 轮历史记录，下一轮为第 [bold]{tracker.next_iteration}[/bold] 轮")
        checks_passed += 1
    else:
        console.print("  [dim]无历史指标，首次迭代将从第 1 轮开始[/dim]")
        checks_passed += 1

    if "dataset" in locals() and dataset is not None:
        dry_pool = SamplePool(output_dir=cfg.output.output_dir)
        dry_pool.load()
        dry_pool.init_from_file_paths(file_paths, existing_labels)
        n_new_labels = 0
        n_external_labels = 0
        n_dedup = 0
        if labels:
            try:
                import chardet as _cd3
                labels_path3 = Path(labels)
                with open(labels_path3, "rb") as f:
                    enc3 = _cd3.detect(f.read(65536)).get("encoding", "utf-8") or "utf-8"
                labels_df3 = pd.read_csv(labels_path3, encoding=enc3)
                lower_cols3 = {c.lower(): c for c in labels_df3.columns}
                path_col3 = None
                label_col3 = None
                for cand in ["path", "file_path", "file", "image", "audio", "sample"]:
                    if cand in lower_cols3:
                        path_col3 = lower_cols3[cand]
                        break
                if path_col3 is None and len(labels_df3.columns) > 0:
                    path_col3 = labels_df3.columns[0]
                for cand in ["label", "category", "class", "target", "y"]:
                    if cand in lower_cols3:
                        label_col3 = lower_cols3[cand]
                        break
                if label_col3 is None and len(labels_df3.columns) > 1:
                    label_col3 = labels_df3.columns[-1]
                if path_col3 and label_col3:
                    dry_labels_map = {}
                    for _, row in labels_df3.iterrows():
                        fp3 = str(row[path_col3]).replace("\\", "/")
                        lbl3 = row[label_col3]
                        if _is_valid_label(lbl3):
                            dry_labels_map[fp3] = lbl3
                    n_new_labels = len(dry_labels_map)
                    recovered, n_dedup, n_external_labels, _ = dry_pool.apply_labels(
                        dry_labels_map, file_paths,
                    )
            except Exception:
                pass

        dry_snap = dry_pool.snapshot()
        n_available_for_sampling = dry_snap.unlabeled

        stat_table = Table(show_lines=False)
        stat_table.add_column("统计项", style="cyan")
        stat_table.add_column("数量", justify="right", style="white")
        stat_table.add_row("总样本数", str(len(file_paths)))
        stat_table.add_row("已有标注", str(dry_snap.labeled))
        stat_table.add_row("待标注(挂起)", str(dry_snap.pending))
        if n_new_labels > 0:
            stat_table.add_row("本次新标注", f"{n_new_labels} (去重 {n_dedup}, 外部 {n_external_labels})")
        stat_table.add_row("未标注(可采样)", str(n_available_for_sampling))
        stat_table.add_row("已跳过", str(dry_snap.skipped))

        console.print("\n[bold]样本统计:[/bold]")
        console.print(stat_table)

        if n_available_for_sampling < cfg.sampler.budget:
            console.print(f"  [yellow]⚠[/yellow] 可采样样本数({n_available_for_sampling})不足采样预算({cfg.sampler.budget})")

    console.print(Panel.fit(
        f"[bold]检查结果:[/bold] "
        f"[green]{checks_passed} 通过[/green], "
        f"[yellow]{checks_warned} 警告[/yellow], "
        f"[red]{checks_failed} 失败[/red]",
        border_style="green" if checks_failed == 0 else "red",
    ))

    if checks_failed > 0:
        raise click.ClickException(f"预检查未通过 ({checks_failed} 项失败)，请修复后重试")


@cli.command(help="查看样本池概况或按状态导出 CSV")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="YAML 配置文件路径")
@click.option("--export", "-e", type=click.Choice(["unlabeled", "pending", "labeled", "skipped", "all"]),
              help="按状态导出 CSV（默认仅查看概况）")
@click.option("--output-dir", "-o", type=click.Path(), help="输出目录（默认用配置文件中的值）")
def pool(config: str, export: Optional[str], output_dir: Optional[str]):
    cfg = ConfigParser.load(config)
    out_dir = Path(output_dir or cfg.output.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_pool = SamplePool(output_dir=out_dir)
    sample_pool.load()

    dataset, file_paths, raw_items, existing_labels = _load_dataset(cfg, console)
    sample_pool.init_from_file_paths(file_paths, existing_labels)
    sample_pool.save()

    snap = sample_pool.snapshot()
    n_total = len(file_paths)

    console.print(Panel.fit(
        f"[bold cyan]样本池概况[/bold cyan]\n"
        f"总样本: [bold]{n_total}[/bold]\n"
        f"[green]已标注: {snap.labeled}[/green]\n"
        f"[yellow]待标注: {snap.pending}[/yellow]\n"
        f"[cyan]未标注: {snap.unlabeled}[/cyan]\n"
        f"[dim]已跳过: {snap.skipped}[/dim]"
        + (f"\n最近导出: [bold]{sample_pool.last_export_csv}[/bold]" if sample_pool.last_export_csv else ""),
        border_style="cyan",
    ))

    if export:
        by_state = sample_pool.by_state()
        states_to_export = [export] if export != "all" else ["unlabeled", "pending", "labeled", "skipped"]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        for st in states_to_export:
            fps = by_state.get(st, [])
            if not fps:
                console.print(f"[dim]状态 {st} 无样本，跳过[/dim]")
                continue
            rows = []
            for fp in fps:
                idx = None
                for i, orig_fp in enumerate(file_paths):
                    if orig_fp.replace("\\", "/") == fp:
                        idx = i
                        break
                lbl = ""
                if existing_labels is not None and idx is not None and idx < len(existing_labels):
                    lbl = str(existing_labels[idx]) if _is_valid_label(existing_labels[idx]) else ""
                rows.append({"file_path": fp, "label": lbl, "state": st})
            export_df = pd.DataFrame(rows)
            export_path = out_dir / f"pool_{st}_{timestamp}.csv"
            export_df.to_csv(export_path, index=False, encoding="utf-8-sig")
            console.print(f"[green]✓[/green] 导出 {st} ({len(fps)} 条): [bold]{export_path}[/bold]")


@cli.command(help="生成示例配置文件")
@click.option("--output", "-o", default="active_learner/data/sample_config.yaml", help="输出路径")
def init_config(output: str):
    ConfigParser.dump_example(output)
    console.print(f"[green]✓[/green] 示例配置已生成: [bold]{output}[/bold]")


def main():
    cli()


if __name__ == "__main__":
    main()
