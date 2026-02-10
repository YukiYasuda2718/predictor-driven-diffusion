import math
import sys
from logging import getLogger
from typing import Literal, Optional

import torch
from torch import Tensor

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

logger = getLogger()


def simulate_lorenz96(
    *,
    x0: Tensor,
    K: int,
    J: int,
    F: float,
    h: float,
    b: float,
    c: float,
    steps: int,
    dt: float,
    sigma: Optional[Tensor] = None,
    disable_progress: bool = False,
) -> Tensor:

    assert x0.ndim == 3 and x0.shape[1] == 2  # channels
    assert isinstance(steps, int) and steps > 0
    assert isinstance(dt, float) and dt > 0.0
    logger.info(
        f"simulate_lorenz96: {K=}, {J=}, {F=}, {h=}, {b=}, {c=}, {steps=}, {dt=}, {sigma=}"
    )

    n_batches, n_channels, n_spaces = x0.shape
    dtype, device = x0.dtype, x0.device

    noise = None
    if sigma is not None:
        assert isinstance(sigma, Tensor) and torch.all(sigma >= 0.0)
        assert sigma.shape == (n_channels,)
        noise = (
            torch.randn(
                size=(n_batches, n_channels, steps, n_spaces),
                dtype=dtype,
                device=device,
            )
            * sigma[None, :, None, None]
            * math.sqrt(float(dt))
        )

    current = x0.clone().detach()
    states = [current.clone().detach().cpu()]

    for it in tqdm(range(steps), disable=disable_progress):
        rhs = _rhs(state=current, K=K, J=J, F=F, h=h, b=b, c=c)
        current = current + rhs * dt

        if noise is not None:
            assert current.shape == noise[:, :, it, :].shape
            current = current + noise[:, :, it, :]

        states.append(current.clone().detach().cpu())

    ret = torch.stack(states, dim=2).cpu()  # B, C, T, L
    assert ret.shape[0] == n_batches
    assert ret.shape[1] == n_channels == 2
    assert ret.shape[3] == n_spaces

    return ret


def _rhs(
    *, state: Tensor, K: int, J: int, F: float, h: float, b: float, c: float
) -> Tensor:

    B, C, N = state.shape
    assert C == 2
    assert N == K * J

    X, Y = state[:, 0, :], state[:, 1, :]  # B, N
    hcb = (h * c) / b

    Y_blocks = Y.view(B, K, J)  # B, N -> B, K, J
    Y_sum_k = Y_blocks.sum(dim=2)  # B, K

    X_blocks = X.view(B, K, J)
    Xk = torch.mean(X_blocks, dim=2)  # B, K
    Xk_m1 = torch.roll(Xk, shifts=1, dims=1)  # X_{k-1}
    Xk_m2 = torch.roll(Xk, shifts=2, dims=1)  # X_{k-2}
    Xk_p1 = torch.roll(Xk, shifts=-1, dims=1)  # X_{k+1}
    dXk = -Xk_m1 * (Xk_m2 - Xk_p1) - Xk + F - hcb * Y_sum_k  # B, K

    # (B, K, 1) -> repeat -> (B, K, J) -> view -> (B, N)
    dXdt = dXk.unsqueeze(-1).repeat(1, 1, J).view(B, N)

    Yj_p1 = torch.roll(Y, shifts=-1, dims=1)  # Y_{j+1,k}
    Yj_p2 = torch.roll(Y, shifts=-2, dims=1)  # Y_{j+2,k}
    Yj_m1 = torch.roll(Y, shifts=1, dims=1)  # Y_{j-1,k}

    adv_fast = -b * c * Yj_p1 * (Yj_p2 - Yj_m1)
    dYdt = adv_fast - c * Y + hcb * X  # B, N

    ret = torch.stack([dXdt, dYdt], dim=1)  # B, 2, N
    assert ret.shape == (B, 2, N)

    return ret
