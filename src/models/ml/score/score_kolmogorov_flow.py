from logging import getLogger

import torch
from torch import Tensor

from src.models.ml.networks.sliding_window_wrapper import SlidingWindowWrapper

from .abstract_score import ScoreFramework

logger = getLogger(__name__)


class ScoreKolmogorovFlow(ScoreFramework):
    def __init__(
        self,
        surrogate_model: SlidingWindowWrapper,
        mean: float,
        std: float,
        n_channels: int,
        n_spaces: int,
        dt: float,
        use_ratio: bool,
        ratio_value: float,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            trainable_closure=surrogate_model,
            mean=mean,
            std=std,
            n_channels=n_channels,
            n_spaces=n_spaces,
            dt=dt,
            use_ratio=use_ratio,
            ratio_value=ratio_value,
            device=device,
            dtype=dtype,
        )

    def _rhs(self, x_dimensionalized: Tensor) -> Tensor:
        return torch.tensor(
            0.0, device=self.device, dtype=self.dtype, requires_grad=False
        )

    def calc_rhs(
        self, x_dimensionless: Tensor, return_dimensionless: bool = True
    ) -> Tensor:
        return torch.tensor(
            0.0, device=self.device, dtype=self.dtype, requires_grad=False
        )
