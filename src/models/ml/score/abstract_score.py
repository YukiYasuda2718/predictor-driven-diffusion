import abc
from logging import getLogger

import torch
from torch import Tensor

from src.models.ml.networks.sliding_window_wrapper import SlidingWindowWrapper

logger = getLogger(__name__)


class ScoreFramework(torch.nn.Module, metaclass=abc.ABCMeta):

    def __init__(
        self,
        trainable_closure: SlidingWindowWrapper,
        mean: float | torch.Tensor,
        std: float | torch.Tensor,
        n_channels: int,
        n_spaces: int,
        dt: float,
        use_ratio: bool,
        ratio_value: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        assert n_spaces % 2 == 0
        assert isinstance(mean, float) or isinstance(mean, torch.Tensor)
        assert isinstance(std, float) or isinstance(std, torch.Tensor)
        if isinstance(std, float):
            assert std > 0.0
        else:
            assert torch.all(std > 0.0)

        super().__init__()

        self.mean = mean
        self.std = std
        self.n_channels = n_channels
        self.n_spaces = n_spaces
        self.dt = dt
        self.use_ratio = use_ratio
        self.device = device

        self.closure: SlidingWindowWrapper = trainable_closure
        self.dtype = dtype

        self.ratio: torch.nn.Parameter | float = ratio_value
        if self.use_ratio:
            self.ratio = torch.nn.Parameter(
                torch.tensor([ratio_value], device=self.device, dtype=self.dtype)
            )
            logger.info(f"Ratio is trainable with the initial value {self.ratio=}")
        else:
            assert isinstance(self.ratio, float)
            logger.info(f"Ratio is a fixed value {self.ratio=}")

        self.fac = self.dt / self.std

    def _dimensionalize(self, x: Tensor) -> Tensor:
        return x * self.std + self.mean

    def _nondimensionalize(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std

    def _velocity(self, x_dimensionalized: Tensor) -> Tensor:
        dx = torch.diff(x_dimensionalized, dim=2, n=1)  # along time dimension
        return dx / self.dt

    @abc.abstractmethod
    def _rhs(self, x_dimensionalized: Tensor) -> Tensor:
        raise NotImplementedError()

    def _closure(self, x_dimensionalized: Tensor, diffusion_time: Tensor) -> Tensor:
        n_batches, n_channels, n_times, n_spaces = x_dimensionalized.shape
        assert diffusion_time.shape == (n_batches,)

        x_view = x_dimensionalized[:, :, :-1, :]
        # batch, channel, time, space (B, C, T, L)

        dimensionless = self._nondimensionalize(x_view)

        closure = self.closure(dimensionless, diffusion_time)
        assert closure.shape == (n_batches, n_channels, n_times - 1, n_spaces)

        return closure * self.std / self.dt  # dimensionalized

    def _residual(self, x_dimensionalized: Tensor, diffusion_time: Tensor) -> Tensor:
        velocity = self._velocity(x_dimensionalized)
        rhs = self._rhs(x_dimensionalized)
        closure = self._closure(x_dimensionalized, diffusion_time)
        return velocity - rhs - closure

    def _exponent(
        self,
        x_dimensionalized: Tensor,
        diffusion_time: Tensor,
        diffusion_std: Tensor,
    ) -> Tensor:

        residual = self._residual(x_dimensionalized, diffusion_time)
        fac = self.fac / (diffusion_std * self.ratio)

        # batch dim remains (summation over channels, times, and spaces)
        return 0.5 * torch.sum((residual * fac) ** 2, dim=(1, 2, 3))

    def forward(
        self,
        x: Tensor,
        time: Tensor,
        vars: Tensor,
        create_graph: bool = True,
        retain_graph: bool = True,
        return_score: bool = False,
        **kwargs,
    ) -> Tensor:

        n_batches, n_channels, n_times, n_spaces = x.shape
        assert n_channels == self.n_channels
        assert n_spaces == self.n_spaces
        assert time.shape == (n_batches,)
        assert vars.shape == (n_batches,)
        assert torch.all(vars > 0.0)

        x_dimensionless = x.detach().clone().requires_grad_(True)
        x_dimensionalized = self._dimensionalize(x_dimensionless)

        diffusion_time = time.detach().clone().requires_grad_(False)

        s = torch.sqrt(vars)[:, None, None, None]  # add channel, time, and space dims
        diffusion_std = s.detach().clone().requires_grad_(False)

        # exponent = - ln P, which has batch dim
        exponent = self._exponent(
            x_dimensionalized=x_dimensionalized,
            diffusion_std=diffusion_std,
            diffusion_time=diffusion_time,
        )

        # score, d [-ln P] / dx = dS / dx (S is an action functional)
        (dS_dx,) = torch.autograd.grad(
            outputs=exponent,
            inputs=x_dimensionless,
            grad_outputs=torch.ones_like(exponent),
            create_graph=create_graph,
            retain_graph=retain_graph,
        )

        if return_score:
            return -dS_dx
        else:
            return diffusion_std * dS_dx  # estimated noise

    def calc_velocity(
        self, x_dimensionless: Tensor, return_dimensionless: bool = True
    ) -> Tensor:
        dx = torch.diff(x_dimensionless, dim=2, n=1)  # along time dimension
        if return_dimensionless:
            return dx
        else:
            return dx * self.std / self.dt  # dimensionalized

    @abc.abstractmethod
    def calc_rhs(
        self, x_dimensionless: Tensor, return_dimensionless: bool = True
    ) -> Tensor:
        raise NotImplementedError()

    def calc_closure(
        self,
        x_dimensionless: Tensor,
        diffusion_time: Tensor,
        return_dimensionless: bool = True,
    ) -> Tensor:

        n_batches, n_channels, n_times, n_spaces = x_dimensionless.shape
        assert n_channels == self.n_channels
        assert n_spaces == self.n_spaces

        closure = self.closure(x_dimensionless, diffusion_time)
        assert closure.shape == (n_batches, n_channels, n_times, n_spaces)

        if return_dimensionless:
            return closure
        else:
            return closure * self.std / self.dt  # dimensionalized
