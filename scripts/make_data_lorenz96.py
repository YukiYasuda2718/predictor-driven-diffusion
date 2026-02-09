import os
import pathlib
import sys
from logging import INFO, StreamHandler, getLogger
from typing import Optional

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

from src.models.dynamics.lorenz96_model import simulate_lorenz96
from src.util.random_seed_helper import set_seeds

set_seeds(42)

logger = getLogger()
logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)

ROOT_DIR = pathlib.Path(os.environ["PYTHONPATH"].split(":")[0]).resolve()

out_dir = ROOT_DIR / "data" / "DL_data" / "lorenz96"
os.makedirs(out_dir, exist_ok=True)


all_batches = 4_000
n_mini_batches = 1_000

h = 1.0
dt = 0.0005

n_channels = 2
out_n_times = 64
out_time_interval = 100
steps = 60_000 + out_time_interval * out_n_times

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float32
seed = 42

assert all_batches % n_mini_batches == 0


def generate_data(
    sigma: Optional[float], K: int, J: int, F: float, b: float, c: float
) -> torch.Tensor:

    n_spaces = K * J

    X = torch.rand(n_mini_batches, n_spaces, dtype=dtype, device=device)
    Y = torch.rand_like(X)

    if sigma == 0.0:
        sigma = None
        logger.info("Sigma is set to None (no noise).")

    alls = simulate_lorenz96(
        x0=torch.stack([X, Y], dim=1),  # stack along channel dim
        K=K,
        J=J,
        F=F,
        h=h,
        b=b,
        c=c,
        dt=dt,
        steps=steps,
    )

    assert alls.shape == (n_mini_batches, n_channels, steps + 1, n_spaces)

    return (
        alls[:, :, -(out_n_times * out_time_interval) :: out_time_interval, :]
        .clone()
        .detach()
        .cpu()
    )


def make_dataarray(
    out_data: torch.Tensor,
    sigma: Optional[float],
    K: int,
    J: int,
    F: float,
    b: float,
    c: float,
) -> xr.DataArray:

    n_spaces = K * J

    ts = np.arange(0.0, out_n_times) * (dt * out_time_interval)
    xs = np.linspace(0, 2 * np.pi, n_spaces, endpoint=False)

    return xr.DataArray(
        out_data.numpy(),
        dims=["batch", "channel", "time", "space"],
        coords={
            "batch": np.arange(all_batches, dtype=np.int32),
            "channel": np.array([0, 1], dtype=np.int32),
            "time": ts.astype(np.float32),
            "space": xs.astype(np.float32),
        },
        name="lorenz96_trajectory",
        attrs={
            "K": K,
            "J": J,
            "N": K * J,
            "F": F,
            "h": h,
            "b": b,
            "c": c,
            "dt": dt,
            "steps": steps,
            "seed": seed,
            "sigma": sigma,
        },
    )


if __name__ == "__main__":
    sigma = 0.0
    b = 10.0
    c = 10.0
    F = 10.0

    for K, J in [(32, 4)]:
        _s = str(sigma).replace(".", "p")
        _b = str(b).replace(".", "p")
        _c = str(c).replace(".", "p")
        _F = str(F).replace(".", "p")

        p = f"{out_dir}/lorenz96_K{K:02}J{J:02}_b{_b}_c{_c}_F{_F}_sigma{_s}.nc"
        if os.path.exists(p):
            logger.info(f"File {p} already exists, skipping generation.")
            continue

        logger.info(f"Generating data with sigma={sigma}")
        assert (
            isinstance(sigma, float) and sigma >= 0.0
        ), "Sigma must be a positive float or zero."
        set_seeds(seed)

        results = []
        for _ in tqdm(range(all_batches // n_mini_batches)):
            result = generate_data(K=K, J=J, F=F, b=b, c=c, sigma=sigma)
            assert torch.all(~torch.isnan(result)).item()
            assert torch.all(torch.isfinite(result)).item()
            results.append(result)

        out_data = torch.cat(results, dim=0)
        assert out_data.shape == (all_batches, n_channels, out_n_times, K * J)

        da = make_dataarray(out_data, K=K, J=J, F=F, b=b, c=c, sigma=sigma)

        da.to_netcdf(path=p)

        logger.info(f"Data saved to {p}")

    logger.info("\nAll data generation completed successfully.")
