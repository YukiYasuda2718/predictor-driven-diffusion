import argparse
import copy
import gc
import math
import os
import pathlib
import sys
import time
import traceback
from itertools import product
from logging import INFO, FileHandler, StreamHandler, getLogger
from typing import Literal

import numpy as np
import torch

from scripts.make_dl_data_v11 import DT as dt
from scripts.make_dl_data_v11 import LX, LY, N_OUT_STEPS
from src.configs.experiment40_config import (
    Experiment40FnoConfig,
    Experiment40UnetConfig,
)
from src.datasets.dataset_kolmogorov_flow import DatasetKolmogorovFlow
from src.models.dynamics.surrogate_kolmogorov_flow_real_space import (
    integrate_surrogate_kolmogorov_flow_in_real_space,
)
from src.models.ml.diffusion.for_deterministic_systems.gaussian_diffusion_2d_cross_entropy import (
    GaussianDiffusion2dCrossEntropy as GaussianDiffusion2D,
)
from src.models.ml.networks.fno_2d import FNO2D
from src.models.ml.networks.sliding_window_wrapper import SlidingWindowWrapper
from src.models.ml.networks.unet_2d_periodic import Unet2D
from src.models.ml.score.for_deterministic_systems.score_surrogate_kolmogorov_flow import (
    ScoreSurrogateKolmogorovFlow,
)
from src.tools.psd import compute_3d_psd
from src.training.diffusion_trainer import Trainer
from src.training.loss_logger import LossLogger
from src.util.io_pickle import write_pickle
from src.util.random_seed_helper import set_seeds
from src.util.variance_helper import dimensionalize_vars

# These constants are passed from scripts.make_dl_data_v11
DT = N_OUT_STEPS * dt
assert LX == LY
L_SPACE = LX
CHANNELS = 1
del dt

os.environ["CUBLAS_WORKSPACE_CONFIG"] = r":4096:8"  # to make calculations deterministic

ROOT_DIR = pathlib.Path(os.environ["PYTHONPATH"].split(":")[0]).resolve()

DL_EXPERIMENT_DIR_PATH = f"{ROOT_DIR}/data/DL_model/experiment40/"

logger = getLogger()
logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda:0")


def make_dataset(
    config: Experiment40UnetConfig | Experiment40FnoConfig,
    root_dir: str,
    kind: Literal["train", "valid", "test"],
):
    assert kind in ["train", "valid", "test"]
    assert config.data_ver == "v11"

    data_dir_path = f"{root_dir}/data/DL_data/{config.data_ver}"
    logger.info(f"make_dataset: {kind=}, {data_dir_path=}")

    return DatasetKolmogorovFlow(
        data_dir_path=data_dir_path,
        min_data_idx=config.data_min_indices[kind],
        max_data_idx=config.data_max_indices[kind],
        nt=config.nt,
        ny=config.ny,
        nx=config.nx,
        mean=config.mean,
        std=config.std,
    )


def initialize_trainer(
    config: Experiment40UnetConfig | Experiment40FnoConfig,
    device: str,
    root_dir: str,
    result_dir: str,
    kind: Literal["train", "valid", "test"],
):

    dataset = make_dataset(config, root_dir, kind)
    logger.info(f"DatasetKolmogorovFlow size: {len(dataset):,}")

    if isinstance(config, Experiment40UnetConfig):
        model = SlidingWindowWrapper(
            window_size=config.window_size,
            missing_value=config.missing_value,
            model=Unet2D(
                dim=config.dim,
                nx=config.nx,
                ny=config.ny,
                padding_mode="circular",
                in_channels=CHANNELS * config.window_size,
                out_channels=CHANNELS,
                dim_mults=tuple(config.dim_mults),
                att_block_indices=tuple(config.att_block_indices),
                time_base=config.time_base,
                has_last_bias=True,
                init_kernel_size=config.init_kernel_size,
            ),
        )
    elif isinstance(config, Experiment40FnoConfig):
        model = SlidingWindowWrapper(
            window_size=config.window_size,
            missing_value=config.missing_value,
            model=FNO2D(
                in_channels=CHANNELS * config.window_size,
                out_channels=CHANNELS,
                ny=config.ny,
                nx=config.nx,
                width=config.width_fno,
                modes_y=config.modes_y_fno,
                modes_x=config.modes_x_fno,
                num_layers=config.num_layers_fno,
                time_base=config.time_base,
                has_last_bias=True,
            ),
        )
    else:
        raise ValueError(f"Unknown config type: {type(config)}")

    noise_estimate_fn = ScoreSurrogateKolmogorovFlow(
        trainable_closure=model.to(device),
        #
        mean=config.mean,
        std=config.std,
        n_channels=CHANNELS,
        n_spaces=int(config.ny * config.nx),
        dt=DT,
        #
        use_ratio=config.use_ratio,
        ratio_value=config.ratio_value,
        #
        device=torch.device(device),
        dtype=torch.float32,
    )

    diffusion = GaussianDiffusion2D(
        noise_estimate_fn=noise_estimate_fn.to(device),
        #
        channels=CHANNELS,
        num_frames=config.nt,
        image_size=0,  # not used
        #
        spatial_dimension="2d",
        image_size_x=config.nx,
        image_size_y=config.ny,
        x_length=L_SPACE,
        #
        num_timesteps=config.num_timesteps,
        laplacian_factor=config.laplacian_factor,
        mass_factor=config.mass_factor,
        skip_A0=config.skip_A0,
        is_always_t0=config.is_always_t0,
        noise_amplitude_squared=config.noise_amplitude_squared,
        substitute_t1_to_t0_for_var=config.substitute_t1_to_t0_for_var,
        #
        loss_mode=config.loss_mode,
        loss_type=config.loss_type,
        noise_type=config.noise_type,
        std_ratio=config.std_ratio,
        min_eigen_value_for_variance=config.min_eigen_value_for_variance,
        use_etd_for_back_sde=True,
        #
        device=torch.device(device),
        dtype=torch.float32,
    )

    trainer = Trainer(
        diffusion_model=diffusion,
        dataset=dataset,
        train_batch_size=config.train_batch_size,
        train_lr=config.train_lr,
        train_num_steps=config.train_num_steps,
        save_and_sample_every=config.save_and_sample_every,
        step_start_ema=config.step_start_ema,
        update_ema_every=config.update_ema_every,
        results_folder=result_dir,
        device=torch.device(device),
    )

    return trainer, dataset


def run_simulation(
    trainer: Trainer,
    dataset: DatasetKolmogorovFlow,
    config: Experiment40UnetConfig | Experiment40FnoConfig,
    n_batches: int,
    diffusion_time: int,
    n_steps: int,
    device: torch.device,
    is_noise_off: bool,
    return_gt: bool,
    disable_tqdm: bool,
):
    logger.info(
        f"{config.data_ver=}, {n_batches=}, {diffusion_time=}, {n_steps=}, {is_noise_off=}"
    )
    assert 0 <= diffusion_time < trainer.model.num_timesteps

    diffusion: GaussianDiffusion2D = trainer.model
    score: ScoreSurrogateKolmogorovFlow = diffusion.noise_estimate_fn

    n_space = int(config.ny * config.nx)
    batched = torch.stack([dataset[i] for i in range(n_batches)], dim=0).contiguous()
    x_0 = copy.deepcopy(batched)
    assert x_0.shape == (n_batches, CHANNELS, config.nt, config.ny, config.nx)

    time = torch.ones((n_batches,), dtype=torch.long, device=device) * diffusion_time

    x_0 = x_0.to(device).view(n_batches, CHANNELS, config.nt, n_space)
    zeros = torch.zeros_like(x_0)
    if not is_noise_off:
        print("Noise is injected into calc_q_samples")
        zeros = diffusion._noise(shape=zeros.shape)
    x_t, _, _ = diffusion._calc_q_samples(x_0, time, noise=zeros)

    samples_t = score._dimensionalize(x_t)
    samples_t = samples_t.cpu().numpy()
    assert samples_t.shape == (n_batches, CHANNELS, config.nt, n_space)

    init_cond = samples_t[:, :, : config.window_size, :]
    assert init_cond.shape == (n_batches, CHANNELS, config.window_size, n_space)

    # non-dimensionalized simulation variances -> dimensionalized variances
    sigma = None
    if diffusion_time >= 0:
        var = float(diffusion.var_t[diffusion_time].item())
        assert var > 0.0
        var = dimensionalize_vars(vars=var, var_scale=config.std**2, dt=DT)
        r = (
            float(score.ratio.item())
            if isinstance(score.ratio, torch.nn.Parameter)
            else float(score.ratio)
        )
        sigma = math.sqrt(var) * r

    _ = score.closure.eval()

    set_seeds(config.seed)
    results = integrate_surrogate_kolmogorov_flow_in_real_space(
        x0=torch.from_numpy(init_cond).to(device=device, dtype=score.dtype),
        dt=DT,
        mean=config.mean,
        std=config.std,
        steps=n_steps,
        closure=score.closure,
        diffusion_times=time,
        sigma=(None if is_noise_off else sigma),
        disable_tqdm=disable_tqdm,
    )

    results = results.detach().clone().cpu()
    nt = config.window_size + n_steps
    s = (n_batches, CHANNELS, nt, n_space)
    assert results.shape == s

    ny = config.ny
    nx = config.nx

    ret = results.view(n_batches, CHANNELS, nt, ny, nx).numpy()

    if not return_gt:
        return ret

    assert samples_t.shape == (n_batches, CHANNELS, config.nt, n_space)
    gt = samples_t.reshape(n_batches, CHANNELS, config.nt, ny, nx)

    return gt, ret


def calc_lsd(psd1: np.ndarray, psd2: np.ndarray) -> float:
    assert psd1.shape == psd2.shape
    return float(np.mean(np.abs(np.log10(psd1) - np.log10(psd2))))


def calc_renormalized_gt(
    trainer: Trainer,
    gt: np.ndarray,
    diffusion_time: int,
    device: torch.device,
) -> np.ndarray:

    n_batches, n_channels, nt, ny, nx = gt.shape
    assert n_channels == CHANNELS

    diffusion: GaussianDiffusion2D = trainer.model
    score: ScoreSurrogateKolmogorovFlow = diffusion.noise_estimate_fn

    x_0 = torch.from_numpy(gt)
    x_0 = score._nondimensionalize(x_0.to(device=device, dtype=torch.float32))

    n_space = ny * nx
    x_0 = x_0.view(n_batches, n_channels, nt, n_space)

    zeros = torch.zeros_like(x_0)
    time = torch.ones((n_batches,), dtype=torch.long, device=device) * diffusion_time
    x_t, _, _ = diffusion._calc_q_samples(x_0, time, noise=zeros)

    samples_t = score._dimensionalize(x_t).view(n_batches, n_channels, nt, ny, nx)

    return samples_t.cpu().numpy()


if __name__ == "__main__":
    try:
        set_seeds(42)

        device = parser.parse_args().device
        config_path: str = parser.parse_args().config_path

        config_name = os.path.basename(config_path).replace(".yml", "")

        if "unet" in config_name:
            config = Experiment40UnetConfig.load(config_path)
        elif "fno" in config_name:
            config = Experiment40FnoConfig.load(config_path)
        else:
            raise ValueError(f"Unknown config name: {config_name}")
        set_seeds(config.seed)

        result_dir = f"{DL_EXPERIMENT_DIR_PATH}/{config_name}"
        os.makedirs(result_dir, exist_ok=False)

        logger.addHandler(FileHandler(f"{result_dir}/log.txt"))
        logger.info(f"{ROOT_DIR=}")
        logger.info(f"{CHANNELS=}, {DT=}, {L_SPACE=}")
        logger.info(f"Config: {config.to_json_str()}")

        trainer, dataset = initialize_trainer(
            config, device, str(ROOT_DIR), result_dir, kind="train"
        )
        # logger.info(
        #     f"Initial weights = {trainer.model.noise_estimate_fn.closure.model.init_conv.weight[0, 0, 0]}"
        # )

        start_time = time.time()
        trainer.train(log_fn=LossLogger(log_file=f"{result_dir}/loss.csv"))
        end_time = time.time()

        logger.info(f"\nTraining time: {(end_time - start_time) / 60.0:.2f} minutes")
        logger.info(f"Training completed. Model saved to {result_dir}")

        del trainer, dataset
        torch.cuda.empty_cache()
        gc.collect()

        logger.info("\nStarting inference...")
        start_time = time.time()

        dict_results = {}
        n_batches = config.n_generated_samples
        pickle_path = f"{result_dir}/generated_samples.pkl"

        trainer, dataset = initialize_trainer(
            config, device, str(ROOT_DIR), result_dir, kind="valid"
        )

        for _diff_time, milestone in product(
            [1, 2, 5, 10, 20, 50, 100, 200, 500],
            range(0, config.train_num_steps + 1, config.save_and_sample_every),
        ):
            if config.is_always_t0 and _diff_time > 1:
                logger.info(f"Skip {_diff_time=} because of {config.is_always_t0=}")
                continue

            diffusion_time = _diff_time - 1
            p = f"{result_dir}/model-{milestone:07}.pt"
            if not os.path.exists(p):
                logger.warning(f"Model {p} not found. Skipping...")
                continue

            logger.info(
                f"Running simulation for model at diffusion_time = {_diff_time}, epoch = {milestone:07}"
            )
            key = f"Epoch{milestone:07}_DiffTime{_diff_time:05}"
            assert key not in dict_results

            trainer.load_only_model(milestone=milestone)

            diffused_gt, results = run_simulation(
                trainer=trainer,
                dataset=dataset,
                config=config,
                n_batches=n_batches,
                diffusion_time=diffusion_time,
                n_steps=config.nt,
                device=device,
                is_noise_off=True,
                disable_tqdm=True,
                return_gt=True,
            )

            _nt = config.window_size + config.nt
            s = (n_batches, CHANNELS, _nt, config.ny, config.nx)
            assert results.shape == s

            results = results[:, :, -config.nt :, :, :]
            s = (n_batches, CHANNELS, config.nt, config.ny, config.nx)
            assert results.shape == diffused_gt.shape == s

            gt_psd = compute_3d_psd(torch.from_numpy(diffused_gt)).cpu().numpy()
            assert gt_psd.shape == (CHANNELS, config.nt, config.ny, config.nx)

            lsd, ratio = np.nan, 0.0

            is_valids = (np.isnan(results).sum(axis=(1, 2, 3, 4)) == 0).astype(bool)
            assert is_valids.shape == (n_batches,)

            if np.sum(is_valids) > 0:
                valids = results[is_valids]
                psd = compute_3d_psd(torch.from_numpy(valids)).cpu().numpy()
                assert psd.shape == (CHANNELS, config.nt, config.ny, config.nx)
                lsd = calc_lsd(gt_psd, psd)
                ratio = np.mean(is_valids)

            dict_results[key] = {"LSD": lsd, "ValidRatio": ratio}
            write_pickle(dict_results, pickle_path)

        end_time = time.time()
        logger.info(f"\nInference time: {(end_time - start_time) / 60.0:.2f} min")
        logger.info(f"Inference completed. Results saved to {pickle_path}")

        logger.info("\nStarting generation...")

        epoch = 30_000
        snr_min, snr_max, snr_cnt = 0.0, 1.0, 21
        n_corrector_steps = 3
        batch_size = 12

        trainer.load_only_model(milestone=epoch)
        _ = trainer.model.noise_estimate_fn.closure.eval()

        for snr in np.linspace(snr_min, snr_max, snr_cnt):
            assert isinstance(trainer.model, GaussianDiffusion2D)
            set_seeds(config.seed)

            snr = float(snr)
            s = str(snr).replace(".", "p")[:5]
            if len(s) < 5:
                s = s + "0" * (5 - len(s))
            path = f"{result_dir}/s{s}_c{n_corrector_steps:01}_b{batch_size:03}.pkl"
            if os.path.exists(path):
                continue

            start_time = time.time()

            results = trainer.model.sample(
                batch_size=batch_size,
                save_interval=1,
                extrapolate_endpoints=True,
                estimate_score_directly=True,
                project_to_no_mean_space=True,
                clamp_min=-2.5,
                clamp_max=2.5,
                use_empirical_prior=False,
                prior_strength=0.0,
                corrector_snr=snr,
                num_corrector_steps=n_corrector_steps,
                use_ETD=True,
            )

            ts = set(map(lambda x: int(x), [5, 4, 3, 2, 1, 0]))

            saved = {}
            for t, items in results.items():
                if t in ts:
                    saved[t] = items
            write_pickle(saved, path)

            end_time = time.time()
            logger.info(
                f"Result path: {path}, Inference time: {(end_time - start_time) / 60.0:.2f} min\n"
            )

        logger.info("Generation completed.")

    except Exception as e:
        logger.error("\n" + "*" * 50)
        logger.error("Error")
        logger.error("*" * 50)
        logger.error(e)
        logger.error(traceback.format_exc())
