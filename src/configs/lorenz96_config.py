import dataclasses
from typing import Literal, Optional

from .base_config import BaseConfig


@dataclasses.dataclass()
class Lorenz96UnetConfig(BaseConfig):
    laplacian_factor: float
    mass_factor: float
    window_size: int
    noise_amplitude_squared: Optional[float]
    #
    data_min_indices: dict[str, int] = dataclasses.field(
        default_factory=lambda: {"train": 0, "valid": 3000, "test": 3100}
    )
    data_max_indices: dict[str, int] = dataclasses.field(
        default_factory=lambda: {"train": 3000, "valid": 3100, "test": 3200}
    )
    data_dir_name: str = "lorenz96"
    data_file_name: str = "lorenz96__K32J04_b10p0_c10p0_F10p0_sigma0p0.nc"
    n_spaces: int = 128
    means: list[float] = dataclasses.field(default_factory=lambda: [2.62, 0.0887])
    stds: list[float] = dataclasses.field(default_factory=lambda: [4.08, 0.261])
    #
    K: int = 32
    J: int = 4
    F: float = 10.0
    b: float = 10.0
    c: float = 10.0
    #
    dim: int = 32
    dim_mults: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 4, 4, 6])
    att_block_indices: list[int] = dataclasses.field(default_factory=lambda: [4])
    time_base: float = 1000.0
    missing_value: float = 0.0
    #
    num_timesteps: int = 1_000
    noise_type: Literal["white", "zero"] = "white"
    std_ratio: float = 1.0
    #
    train_batch_size: int = 50
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
    nn_name: str = "unet1d"
    seed: int = 42
