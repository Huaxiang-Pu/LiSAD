"""Centralized experiment configuration shared across scripts and modules."""

from pydantic import BaseModel, Field
from pathlib import Path
from typing import List
from typing import Tuple


class DataConfig(BaseModel):
    """Dataset selection and preprocessing defaults."""

    root_dir: Path = Path("~/workspace/DATABASES/FOBSS")
    foldername: str = (
        "Ri Jumps,Ri Jumps 25A,Ri Jumps 25A Run 2,Ri Jumps 25A Run 3,Ri Jumps 25A Run 4"
    )
    # foldername: str = "Ri Jumps 25A Run 3"
    slave: str = "0"
    n_cell: int = 11
    interval: float = 10.0  # in seconds
    window_length: int = 100  # in time steps
    method: str = "linear"  # "linear" / "nearest"


class TrainConfig(BaseModel):
    """Optimization, batching and runtime defaults."""

    split_val: float = 0.2
    step: int = 1
    batch_size: int = 512
    shuffle: bool = True
    num_workers: int = 16
    pin_memory: bool = True
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 2.0
    log_interval: int = 25
    save_interval: int = 20
    seed: int = 42
    device: str = "cuda"  # "cpu" / "cuda"


# Model-specific hyperparameters for LiBPINN and related modules
class ModelConfig(BaseModel):
    """Model architecture and physical-prior defaults for ``LiBPINN``."""

    n_cell: int = 11
    dt: float = 10.0
    m: float = 0.8
    cp: float = 900.0
    eta: float = 1.0
    Uoc_min: float = 2.8
    Uoc_max: float = 4.2
    Qcap: float = 40 * 3600.0
    hA: float = 10.0
    R0_p: tuple = (5e-3, 0.1)
    R1_p: tuple = (1e-3, 0.1)
    R2_p: tuple = (5e-3, 0.1)
    C1_p: tuple = (1e3, 0.1)
    C2_p: tuple = (5e3, 0.1)
    D_in: int = 3
    D_hidden: int = 16
    n_lstm_layers: int = 2
    bidirectional: bool = True
    lstm_dropout: float = 0.0
    n_heads: int = 4
    n_trans_layers: int = 1
    dim_feedforward: int = 32
    trans_dropout: float = 0.0
    require_adj: bool = True
    use_adj_bias: bool = True
    alpha_init: float = 1.0
    beta_init: float = 0.1
    V1_clip: float = 0.03
    V2_clip: float = 0.20
    tau_SOC: float = 1.0
    eps_SOC: float = 0.01
    tau_Uoc: float = 1.0
    eps_Uoc: float = 0.01
    device: str = "cuda"


class BalancerConfig(BaseModel):
    """Default hyperparameters for the covariance-based loss balancer."""

    n_losses: int = 9
    diag_eps: float = 1e-4
    init_scale: float = 0.001
    log_base: float = 10.0
    log_eps: float = 1e-12
    tau: float = 1.25


class Config(BaseModel):
    """Top-level project configuration container."""

    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    model: ModelConfig = ModelConfig()
    balancer: BalancerConfig = BalancerConfig()
    save_dir: Path = Path("~/workspace/LiSAD/checkpoints/")
    data_dir: Path = Path("~/workspace/LiSAD/data/")
    log_dir: Path = Path("~/workspace/LiSAD/logs/")


settings = Config()
