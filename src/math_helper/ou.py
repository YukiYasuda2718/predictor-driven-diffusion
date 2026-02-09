import math
from typing import Optional

import torch
from torch import Tensor


def laplacian_matrix(
    L: int, h: float, device: str = "cpu", dtype: torch.dtype = torch.float64
) -> Tensor:
    M = torch.zeros((L, L), dtype=dtype, device=device)
    for i in range(L):
        M[i, i] = -2
        M[i, (i + 1) % L] = 1
        M[i, (i - 1) % L] = 1
    return M / (h**2)


def make_AB_t_matrix(
    N: int,
    a: float,
    b: float,
    m: float,
    t: float,
    h: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    min_variance: Optional[float] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    #
    assert a >= 0.0 and b > 0.0 and m > 0.0 and h > 0.0

    L = laplacian_matrix(N, h, device, dtype)
    M = a * L - m * torch.eye(N, dtype=dtype, device=device)

    # A_t = exp(M t)
    A_t = torch.linalg.matrix_exp(M * t)

    # Covariance = Q_t
    evals, evecs = torch.linalg.eigh(M)
    denom = 2 * evals
    frac = torch.zeros_like(evals)
    mask = torch.abs(denom) > 1e-14
    frac[mask] = (torch.exp(denom[mask] * t) - 1.0) / denom[mask]
    frac[~mask] = t
    variances = b**2 * frac
    if min_variance is not None:
        variances = torch.clamp(variances, min=min_variance)
    Q_t = evecs @ torch.diag(variances) @ evecs.T

    # B_t = sqrt(Q_t)
    evals_Q, evecs_Q = torch.linalg.eigh(Q_t)
    B_t = evecs_Q @ torch.diag(torch.sqrt(torch.clamp(evals_Q, min=0))) @ evecs_Q.T

    return A_t, B_t, Q_t, M


def ou_spectrum_1d_eigenvalues(
    N: int,
    a: float,
    m: float,
    h: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    #
    # Eigenvalues of the Laplacian
    k = torch.fft.fftfreq(N, d=1.0, device=device, dtype=dtype)
    lam = -4.0 * torch.sin(torch.pi * k) ** 2 / (h**2)

    # Effective eigenvalues including mass term
    lam_eff = a * lam - m

    return lam_eff


def make_AB_t_fft(
    N: int,
    a: float,
    b: float,
    m: float,
    t: float,
    h: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.complex128,
) -> tuple[Tensor, Tensor]:
    #
    lam_eff = ou_spectrum_1d_eigenvalues(N=N, a=a, m=m, h=h, device=device)

    # A_hat
    A_hat = torch.exp(lam_eff * t)

    # Q_hat
    Q_hat = torch.zeros_like(lam_eff)
    mask = lam_eff != 0
    Q_hat[mask] = b**2 * (torch.exp(2 * lam_eff[mask] * t) - 1.0) / (2 * lam_eff[mask])
    Q_hat[~mask] = b**2 * t
    B_hat = torch.sqrt(torch.clamp(Q_hat, min=0))

    # FFT matrices
    F = torch.fft.fft(torch.eye(N, dtype=dtype, device=device), dim=0)
    Finv = torch.conj(F).T / N

    A_t = (Finv @ torch.diag(A_hat.to(dtype)) @ F).real
    B_t = (Finv @ torch.diag(B_hat.to(dtype)) @ F).real

    return A_t, B_t


def ou_spectrum_2d_eigenvalues(
    Ny: int,
    Nx: int,
    a: float,
    m: float,
    dy: float,
    dx: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    ky = torch.arange(Ny, device=device, dtype=dtype)
    kx = torch.arange(Nx, device=device, dtype=dtype)

    lam_y = -4.0 * torch.sin(math.pi * ky / Ny) ** 2 / (dy**2)
    lam_x = -4.0 * torch.sin(math.pi * kx / Nx) ** 2 / (dx**2)

    lam_2d = lam_y[:, None] + lam_x[None, :]  # (Ny, Nx)
    lam_eff = a * lam_2d - m  # drift M = aΔ - mI の固有値

    return lam_eff
