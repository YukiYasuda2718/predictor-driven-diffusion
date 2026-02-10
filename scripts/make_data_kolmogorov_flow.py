import math
import os
import pathlib
import sys
from logging import INFO, StreamHandler, getLogger

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

from src.models.dynamics.cfd_model_in_doubly_periodic_domain import TorchSpectralModel2D
from src.util.random_seed_helper import set_seeds

set_seeds(42)

logger = getLogger()
logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)

ROOT_DIR = pathlib.Path(os.environ["PYTHONPATH"].split(":")[0]).resolve()

out_dir = ROOT_DIR / "data" / "DL_data" / "kolmogorov_flow"
os.makedirs(out_dir, exist_ok=True)

LX = 2 * math.pi
LY = 2 * math.pi
NX = 256
NY = 256
COEFF_LINEAR_DRAG = 0.1
COEFF_DIFFUSION = 1e-3
ORDER_DIFFUSION = 1
FORCING_WAVENUMBER = 4.0
NOISE_AMPLITUDE = 5.0

DT = 0.001
N_OUT_STEPS = 500
T_BURN_IN = 50.0

N_SPACE = 40
N_FRAME = 40

DTYPE = torch.complex128


def initialize_model(n_ens: int, seed: int, device: str):
    model = TorchSpectralModel2D(
        nx=NX,
        ny=NY,
        coeff_linear_drag=COEFF_LINEAR_DRAG,
        coef_diffusion=COEFF_DIFFUSION,
        order_diffusion=ORDER_DIFFUSION,
        device=device,
        dtype=DTYPE,
    )

    xs = np.linspace(0, LX, num=NX, endpoint=False)
    ys = np.linspace(0, LY, num=NY, endpoint=False)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    forcing = (-FORCING_WAVENUMBER * np.cos(FORCING_WAVENUMBER * Y)).transpose()[
        None, ...
    ]

    set_seeds(seed)
    noise = NOISE_AMPLITUDE * np.random.rand(n_ens * NX * NY).reshape(n_ens, NY, NX)
    omega0 = forcing + noise

    model.initialize(
        t0=0.0,
        omega0=torch.tensor(omega0, dtype=torch.float64, device=device),
        forcing=torch.tensor(forcing, dtype=torch.float64, device=device),
    )
    model.calc_grid_data()

    return model


def generate_data(n_ens: int, seed: int, device: str):
    model = initialize_model(n_ens=n_ens, seed=seed, device=device)
    lst = []

    for _ in tqdm(range(N_FRAME + int(T_BURN_IN / (DT * N_OUT_STEPS)) + 1)):
        model.time_integrate(dt=DT, nt=N_OUT_STEPS, hide_progress_bar=True)
        if model.t < T_BURN_IN:
            continue

        model.calc_grid_data()
        omega = model.omega.detach().clone().cpu()[:, None].permute(0, 1, 3, 2)
        assert omega.shape == (n_ens, 1, NX, NY)
        omega = torch.nn.functional.interpolate(
            omega, size=(N_SPACE, N_SPACE), mode="bicubic", align_corners=True
        )
        assert omega.shape == (n_ens, 1, N_SPACE, N_SPACE)

        lst.append(omega)

        if len(lst) == N_FRAME:
            break

    alls = torch.cat(lst, dim=1)
    assert alls.shape == (n_ens, N_FRAME, N_SPACE, N_SPACE)

    return alls


def make_dataarray(out_data: torch.Tensor) -> xr.DataArray:

    bs = np.arange(0, out_data.shape[0])
    ts = np.arange(0.0, out_data.shape[1]) * DT * N_OUT_STEPS
    xs = np.linspace(0, LX, out_data.shape[2], endpoint=False)
    ys = np.linspace(0, LY, out_data.shape[3], endpoint=False)

    return xr.DataArray(
        out_data.numpy().astype(np.float32),
        dims=["batch", "t", "x", "y"],
        coords={
            "batch": bs.astype(np.int32),
            "t": ts.astype(np.float32),
            "x": xs.astype(np.float32),
            "y": ys.astype(np.float32),
        },
    )


if __name__ == "__main__":
    set_seeds(42)
    seeds = np.random.randint(low=100_000, high=999_999, size=999)
    n_ens = 100
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"{device=}\n")

    for _i in range(70):
        i = _i + 1
        p_npy = (
            f"{out_dir}/kf_{n_ens:03}x{N_FRAME:02}x{N_SPACE:02}x{N_SPACE:02}_{i:03}.npy"
        )
        if os.path.exists(p_npy):
            logger.info(f"Files {p_npy} already exists, skipping generation.")
            continue

        data = generate_data(n_ens=n_ens, seed=int(seeds[i]), device=device)
        np.save(p_npy, data.numpy().astype(np.float32))
        logger.info(f"Data saved to {p_npy}")

    logger.info("\nAll data generation completed successfully.")
