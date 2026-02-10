import copy
import math
import sys
from logging import getLogger
from typing import Literal, Optional

import numpy as np
import torch
from torch import Tensor

from src.configs.kolmogorov_flow_config import KolmogorovFlowUnetConfig
from src.configs.lorenz96_config import Lorenz96UnetConfig
from src.datasets.dataset_kolmogorov_flow import DatasetKolmogorovFlow
from src.datasets.dataset_lorenz96 import DatasetLorenz96
from src.models.ml.diffusion.gaussian_diffusion import GaussianDiffusion
from src.models.ml.networks.sliding_window_wrapper import SlidingWindowWrapper
from src.models.ml.score.abstract_score import ScoreFramework
from src.training.trainer import Trainer

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

logger = getLogger()


def run_simulation(
    *,
    trainer: Trainer,
    dataset: DatasetLorenz96 | DatasetKolmogorovFlow,
    config: Lorenz96UnetConfig | KolmogorovFlowUnetConfig,
    n_batches: int,
    diffusion_index: int,
    dt: float,
    n_steps: int,
    device: torch.device,
    is_noise_off: bool
):
    if isinstance(config, Lorenz96UnetConfig):
        assert isinstance(dataset, DatasetLorenz96)
    elif isinstance(config, KolmogorovFlowUnetConfig):
        assert isinstance(dataset, DatasetKolmogorovFlow)
    else:
        raise ValueError()

    diffusion: GaussianDiffusion = trainer.model
    score: ScoreFramework = diffusion.noise_estimate_fn

    batched = torch.stack([dataset[i] for i in range(n_batches)], dim=0)
    x_0 = copy.deepcopy(batched).to(device)
    if isinstance(config, KolmogorovFlowUnetConfig):
        b, c, t, _, _ = x_0.shape
        x_0 = x_0.contiguous().view(b, c, t, -1)
    time = torch.ones((n_batches,), dtype=torch.long, device=device) * diffusion_index
    zeros = torch.zeros_like(x_0)
    x_t, _, _ = diffusion._calc_q_samples(x_0, time, noise=zeros)

    ground_truth = score._dimensionalize(x_t).cpu().numpy()
    init_cond = ground_truth[:, :, : config.window_size, :]

    # non-dimensionalized simulation variances -> dimensionalized variances
    std_scales = torch.tensor(dataset.std, dtype=torch.float32)
    var = _dimensionalize_vars(
        vars=diffusion.var_t[diffusion_index].item(), var_scale=std_scales**2, dt=dt
    )
    r = (
        float(score.ratio.item())
        if isinstance(score.ratio, torch.nn.Parameter)
        else float(score.ratio)
    )
    sigma = (torch.sqrt(var) * r).to(device=device, dtype=score.dtype)

    _ = score.closure.eval()

    results = _run_simulation(
        x0=torch.from_numpy(init_cond).to(device=device, dtype=score.dtype),
        steps=n_steps,
        dt=dt,
        sigma=(None if is_noise_off else sigma),
        mean=torch.tensor(dataset.mean, device=device, dtype=score.dtype),
        std=torch.tensor(dataset.std, device=device, dtype=score.dtype),
        surrogate_model=score.closure,
        diffusion_times=time,
    )
    ret = results.detach().clone().cpu().numpy()

    if isinstance(config, KolmogorovFlowUnetConfig):
        b, c, t, _ = ground_truth.shape
        ground_truth = ground_truth.reshape(b, c, t, config.ny, config.nx)
        b, c, t, _ = ret.shape
        ret = ret.reshape(b, c, t, config.ny, config.nx)

    return ground_truth, ret


def _dimensionalize_vars(
    vars: float | np.ndarray | torch.Tensor,
    var_scale: float | np.ndarray | torch.Tensor,
    dt: float,
) -> float | np.ndarray | torch.Tensor:

    return vars * var_scale / dt


def _run_simulation(
    x0: Tensor,
    steps: int,
    dt: float,
    sigma: Optional[Tensor],
    mean: Tensor,
    std: Tensor,
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
            * (sigma[None, :, None, None] if sigma.ndim > 0 else sigma)
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
    assert prevs.shape == (n_batches, n_channels, window_size, n_spaces)

    return prevs


def _calc_time_tendency(
    closure: torch.nn.Module,
    states: Tensor,
    diffusion_times: Tensor,
    mean: Tensor,
    std: Tensor,
    dt: float,
) -> Tensor:
    #
    assert states.ndim == 4
    B, C, T, L = states.shape
    assert diffusion_times.shape == (B,)
    if C > 1:
        assert mean.shape == std.shape == (C,)

    states_dimensionless = states.clone().detach()
    m = mean[None, :, None, None] if mean.ndim > 0 else mean
    states_dimensionless = states_dimensionless - m
    s = std[None, :, None, None] if std.ndim > 0 else std
    states_dimensionless = states_dimensionless / s

    with torch.inference_mode():
        cls_term = closure(x=states_dimensionless, time=diffusion_times)
    assert cls_term.shape == (B, C, T, L)
    cls_term = cls_term[:, :, -1, :].detach().clone()  # take last time step
    assert cls_term.shape == (B, C, L)

    s = std[None, :, None] if std.ndim > 0 else std
    return cls_term * s / dt  # dimensionalized
