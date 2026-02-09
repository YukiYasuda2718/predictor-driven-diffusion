from logging import getLogger
from typing import Literal

import torch
from torch import Tensor

from src.models.ml.networks.sliding_window_wrapper import SlidingWindowWrapper

from .abstract_score import ScoreFramework

logger = getLogger(__name__)


class ScoreLorenz96(ScoreFramework):

    def __init__(
        self,
        trainable_closure: SlidingWindowWrapper,
        mean: torch.Tensor,
        std: torch.Tensor,
        n_channels: int,
        n_spaces: int,
        dt: float,
        use_ratio: bool,
        ratio_value: float,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        assert (
            mean.shape == std.shape == (n_channels, 1, 1)
        )  # channel, time, space dims

        super().__init__(
            trainable_closure=trainable_closure,
            mean=mean.to(device=device, dtype=dtype),
            std=std.to(device=device, dtype=dtype),
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
