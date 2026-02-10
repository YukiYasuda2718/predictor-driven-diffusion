import dataclasses
from typing import Literal, Optional

from .base_config import BaseConfig


@dataclasses.dataclass()
class KolmogorovFlowUnetConfig(BaseConfig):
    laplacian_factor: float
    mass_factor: float
    window_size: int
    noise_amplitude_squared: Optional[float]
    seed: int
    #
    data_min_indices: dict[str, int] = dataclasses.field(
        default_factory=lambda: {"train": 1, "valid": 61, "test": 62}
    )
    data_max_indices: dict[str, int] = dataclasses.field(
        default_factory=lambda: {"train": 60, "valid": 61, "test": 62}
    )
    data_dir_name: str = "kolmogorov_flow"
    nt: int = 40
    ny: int = 40
    nx: int = 40
    mean: float = -0.0304
    std: float = 4.58
    #
    dim: int = 32
    dim_mults: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 4, 6])
    att_block_indices: list[int] = dataclasses.field(default_factory=lambda: [3])
    time_base: float = 1000.0
    init_kernel_size: int = 5
    missing_value: float = 0.0
    #
    num_timesteps: int = 1_000
    noise_type: Literal["white", "zero"] = "white"
    std_ratio: float = 1.0
    #
    train_batch_size: int = 40
    train_lr: float = 2e-4
    train_num_steps: int = 30_001
    save_and_sample_every: int = 2_000
    #
    step_start_ema: int = 999_999
    update_ema_every: int = 999_999
    #
    use_ratio: bool = False
    ratio_value: float = 1.0
    #
    nn_name: str = "unet2d"
