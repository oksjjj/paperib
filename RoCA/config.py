"""Default RoCA hyperparams for paperib (multivariate ≈ SWaT config)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Augmentations:
    scale_ratio: float = 0.8
    jitter_ratio: float = 0.2


@dataclass
class RoCAConfig:
    """Mirrors ``conf/roca/SWaT_Configs.Config`` with paperib overrides."""

    dataset: str = "SWaT"  # multivariate CNN path in official network
    input_channels: int = 46
    kernel_size: int = 8
    stride: int = 1
    final_out_channels: int = 32
    hidden_size: int = 64
    num_layers: int = 3
    project_channels: int = 64
    dropout: float = 0.45
    window_size: int = 32
    time_step: int = 16

    num_epoch: int = 20
    change_center_epoch: int = 1
    center_eps: float = 0.01
    omega: float = 0.02

    beta1: float = 0.9
    beta2: float = 0.99
    lr: float = 3e-4
    weight: float = 5e-4

    drop_last: bool = False
    batch_size: int = 256

    warmup: int = 10
    train_method: str = "contaminated"
    nu: float = 0.01
    mu: float = 0.85

    detect_nu: float = 0.0005
    threshold_determine: str = "floating"
    anomaly_ratio: float = 1.0  # paperib train-percentile thr (%)

    augmentation: Augmentations = field(default_factory=Augmentations)
