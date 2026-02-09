import math
import sys
from logging import getLogger
from typing import Literal, Optional

import torch
from torch import Tensor

from src.models.ml.networks.sliding_window_wrapper import SlidingWindowWrapper

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

logger = getLogger()


def run_integration(
    x0: Tensor,
    steps: int,
    dt: float,
    sigma: Optional[Tensor | float],
    mean: Tensor | float,
    std: Tensor | float,
    surrogate_model: SlidingWindowWrapper,
    diffusion_times: Tensor,
    disable_tqdm: bool = False,
) -> Tensor:

    assert isinstance(steps, int) and steps > 0
    assert isinstance(dt, float) and dt > 0.0

    n_batches, n_channels, n_frames, n_spaces = x0.shape
    dtype, device = x0.dtype, x0.device
    assert n_frames == surrogate_model.window_size
    assert isinstance(diffusion_times, Tensor) and diffusion_times.shape == (n_batches,)

    noise = None
    if sigma is not None:
        noise = (
            torch.randn(
                size=(n_batches, n_channels, steps, n_spaces),
                dtype=dtype,
                device=device,
            )
            * (sigma[None, :, None, None] if isinstance(sigma, Tensor) else sigma)
            * math.sqrt(float(dt))
        )

    _x0 = x0.clone().detach().cpu()
    states = [_x0[:, :, i, :] for i in range(n_frames)]
    window_size = surrogate_model.window_size

    current = x0[:, :, -1, :]  # (B, C, L)
    assert current.shape == (n_batches, n_channels, n_spaces)

    for it in tqdm(range(steps), disable=disable_tqdm):

        prevs = _extract_previous_states(
            snaps=states, window_size=window_size, interval=1
        )
        assert prevs.shape == (n_batches, n_channels, window_size, n_spaces)
        cls_term = _calc_time_tendency(
            closure=surrogate_model,
            states=prevs.to(device),
            diffusion_times=diffusion_times.to(device),
            mean=mean,
            std=std,
            dt=dt,
        )
        assert cls_term.shape == current.shape == (n_batches, n_channels, n_spaces)

        current = current + cls_term * dt

        if noise is not None:
            assert current.shape == noise[:, :, it, :].shape
            current = current + noise[:, :, it, :]

        states.append(current.clone().detach().cpu())

    ret = torch.stack(states, dim=2)  # B, C, T, L
    assert ret.shape[0] == n_batches
    assert ret.shape[1] == n_channels
    assert ret.shape[3] == n_spaces

    return ret


def _extract_previous_states(
    snaps: list[Tensor], window_size: int, interval: int
) -> Tensor:
    assert window_size > 0 and interval > 0

    tmp = snaps[::-interval]
    assert len(tmp) >= window_size

    prevs = torch.stack(tmp[:window_size][::-1], dim=2).detach().clone()
    # shape = (B, C, T, L)

    n_batches, n_channels, n_spaces = snaps[0].shape
    assert n_channels == 2
    assert prevs.shape == (n_batches, n_channels, window_size, n_spaces)

    return prevs


def _calc_time_tendency(
    closure: torch.nn.Module,
    states: Tensor,
    diffusion_times: Tensor,
    mean: Tensor | float,
    std: Tensor | float,
    dt: float,
) -> Tensor:
    #
    assert states.ndim == 4
    B, C, T, L = states.shape
    assert diffusion_times.shape == (B,)
    if C > 1:
        assert mean.shape == std.shape == (C,)

    states_dimensionless = states.clone().detach()
    m = mean[None, :, None, None] if isinstance(mean, Tensor) else mean
    states_dimensionless = states_dimensionless - m
    s = std[None, :, None, None] if isinstance(std, Tensor) else std
    states_dimensionless = states_dimensionless / s

    with torch.inference_mode():
        cls_term = closure(x=states_dimensionless, time=diffusion_times)
    assert cls_term.shape == (B, C, T, L)
    cls_term = cls_term[:, :, -1, :].detach().clone()  # take last time step
    assert cls_term.shape == (B, C, L)

    s = std[None, :, None] if isinstance(std, Tensor) else std
    return cls_term * s / dt  # dimensionalized
