from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DataConfig(BaseModel):
    data_type: Literal["text", "image", "audio"] = Field(
        ..., description="数据类型: text / image / audio"
    )
    data_path: str = Field(..., description="未标注数据集路径")
    labeled_path: Optional[str] = Field(None, description="已标注数据 CSV 路径")
    text_column: Optional[str] = Field(None, description="文本列名（自动检测）")
    label_column: Optional[str] = Field(None, description="标签列名（自动检测）")
    image_size: tuple[int, int] = Field((224, 224), description="图像输入尺寸 (H, W)")
    audio_sr: int = Field(22050, description="音频采样率")
    audio_mfcc: int = Field(40, description="MFCC 特征维度")


class ModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_path: str = Field(..., description="基线模型路径 (.pkl / .pth / .onnx)")
    model_type: Optional[Literal["sklearn", "pytorch", "onnx"]] = Field(
        None, description="模型类型（自动根据后缀推断）"
    )
    num_classes: int = Field(..., ge=2, description="类别数量")
    device: Literal["auto", "cpu", "cuda"] = Field("auto", description="推理设备")
    batch_size: int = Field(32, ge=1, description="推理批大小")


class SamplerConfig(BaseModel):
    strategy: Literal["uncertainty", "diversity", "hybrid", "qbc"] = Field(
        "hybrid", description="采样策略"
    )
    uncertainty_method: Literal["least_confidence", "margin", "entropy"] = Field(
        "entropy", description="不确定性采样方法"
    )
    uncertainty_weight: float = Field(0.6, ge=0.0, le=1.0, description="不确定性权重")
    diversity_weight: float = Field(0.4, ge=0.0, le=1.0, description="多样性权重")
    budget: int = Field(100, ge=1, description="每轮选择样本数")
    feature_extractor: Optional[str] = Field(
        None, description="特征提取器模型名（文本默认 all-MiniLM-L6-v2）"
    )
    num_clusters: Optional[int] = Field(None, description="KMeans 簇数（默认=budget）")

    @field_validator("diversity_weight")
    @classmethod
    def check_weights_sum(cls, v: float, info) -> float:
        uw = info.data.get("uncertainty_weight", 0.6)
        total = uw + v
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"uncertainty_weight + diversity_weight 必须等于 1，当前为 {total}")
        return v


class TrainConfig(BaseModel):
    enabled: bool = Field(True, description="是否启用重训练")
    epochs: int = Field(10, ge=1, description="训练轮数")
    learning_rate: float = Field(1e-4, gt=0, description="学习率")
    early_stopping_patience: int = Field(3, ge=1, description="早停耐心值")
    freeze_backbone: bool = Field(True, description="冻结骨干网络（仅训练最后一层）")
    checkpoint_dir: str = Field("checkpoints", description="模型 checkpoint 保存目录")


class OutputConfig(BaseModel):
    output_dir: str = Field("outputs", description="输出目录")
    export_csv: bool = Field(True, description="是否导出 CSV")
    export_visualization: bool = Field(True, description="是否导出来可视化 HTML")
    top_k_display: int = Field(20, ge=1, description="终端显示 Top-K 推荐样本")


class ActiveLearningConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    seed: int = Field(42, description="随机种子")


class ConfigParser:
    @staticmethod
    def load(config_path: str | Path) -> ActiveLearningConfig:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError("配置文件格式错误：根节点必须是字典")

        return ActiveLearningConfig(**raw)

    @staticmethod
    def dump_example(output_path: str | Path) -> None:
        example = {
            "data": {
                "data_type": "text",
                "data_path": "./data/unlabeled.csv",
                "labeled_path": "./data/labeled.csv",
                "text_column": "text",
                "label_column": "label",
                "image_size": [224, 224],
                "audio_sr": 22050,
                "audio_mfcc": 40,
            },
            "model": {
                "model_path": "./models/baseline.pkl",
                "num_classes": 10,
                "device": "auto",
                "batch_size": 32,
            },
            "sampler": {
                "strategy": "hybrid",
                "uncertainty_method": "entropy",
                "uncertainty_weight": 0.6,
                "diversity_weight": 0.4,
                "budget": 100,
            },
            "train": {
                "enabled": True,
                "epochs": 10,
                "learning_rate": 1e-4,
                "early_stopping_patience": 3,
                "freeze_backbone": True,
                "checkpoint_dir": "checkpoints",
            },
            "output": {
                "output_dir": "outputs",
                "export_csv": True,
                "export_visualization": True,
                "top_k_display": 20,
            },
            "seed": 42,
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(example, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
