import argparse
import math
import os
import pathlib
import sys
import time
import traceback
from logging import INFO, FileHandler, StreamHandler, getLogger
from typing import Literal

import torch

from scripts.make_data_lorenz96 import dt, h, out_n_times, out_time_interval
from src.configs.lorenz96_config import Lorenz96UnetConfig
from src.datasets.dataset_lorenz96 import DatasetLorenz96
from src.models.ml.diffusion.gaussian_diffusion import GaussianDiffusion
from src.models.ml.networks.sliding_window_wrapper import SlidingWindowWrapper
from src.models.ml.networks.unet_1d import Unet1D
from src.models.ml.score.score_lorenz96 import ScoreLorenz96
from src.training.loss_logger import LossLogger
from src.training.trainer import Trainer
from src.util.random_seed_helper import set_seeds

# These constants are passed from scripts.make_dl_data_v09
N_FRAMES = out_n_times
BARE_DT = dt
DT = dt * out_time_interval
H = h
CHANNELS = 2
X_LENGTH = 2.0 * math.pi
del dt, h, out_n_times, out_time_interval

os.environ["CUBLAS_WORKSPACE_CONFIG"] = r":4096:8"  # to make calculations deterministic

ROOT_DIR = pathlib.Path(os.environ["PYTHONPATH"].split(":")[0]).resolve()

DL_EXPERIMENT_DIR_PATH = f"{ROOT_DIR}/data/DL_model/lorenz96"

logger = getLogger()
logger.setLevel(INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda:0")


def make_dataset(
    config: Lorenz96UnetConfig,
    root_dir: str,
    kind: Literal["train", "valid", "test"],
):
    assert kind in ["train", "valid", "test"]

    _b = str(config.b).replace(".", "p")
    _c = str(config.c).replace(".", "p")
    _F = str(config.F).replace(".", "p")
    _name = f"K{config.K:02}J{config.J:02}_b{_b}_c{_c}_F{_F}"
    assert _name in config.data_file_name

    dl_data_file_path = (
        f"{root_dir}/data/DL_data/{config.data_dir_name}/{config.data_file_name}"
    )
    logger.info(f"make_dataset: {kind=}, {dl_data_file_path=}")

    return DatasetLorenz96(
        path_to_dataarray=dl_data_file_path,
        means=config.means,
        stds=config.stds,
        min_data_idx=config.data_min_indices[kind],
        max_data_idx=config.data_max_indices[kind],
    )


def initialize_trainer(
    config: Lorenz96UnetConfig,
    device: str,
    root_dir: str,
    result_dir: str,
    kind: Literal["train", "valid", "test"],
):

    dataset = make_dataset(config, root_dir=root_dir, kind=kind)

    if isinstance(config, Lorenz96UnetConfig):
        model = SlidingWindowWrapper(
            window_size=config.window_size,
            missing_value=config.missing_value,
            model=Unet1D(
                dim=config.dim,
                padding_mode="circular",
                in_channels=CHANNELS * config.window_size,
                out_channels=CHANNELS,
                dim_mults=tuple(config.dim_mults),
                att_block_indices=config.att_block_indices,
                time_base=config.time_base,
                has_last_bias=True,
            ),
        )
    else:
        raise ValueError(f"Unknown config type: {type(config)}")

    noise_estimate_fn = ScoreLorenz96(
        surrogate_model=model.to(device),
        #
        mean=torch.tensor(config.means)[:, None, None],  # add time and space dims
        std=torch.tensor(config.stds)[:, None, None],
        n_channels=CHANNELS,
        n_spaces=config.n_spaces,
        dt=DT,
        #
        use_ratio=config.use_ratio,
        ratio_value=config.ratio_value,
        #
        device=torch.device(device),
        dtype=torch.float32,
    )

    diffusion = GaussianDiffusion(
        noise_estimate_fn=noise_estimate_fn.to(device),
        #
        channels=CHANNELS,
        num_frames=N_FRAMES,
        image_size=config.n_spaces,
        x_length=X_LENGTH,
        #
        num_timesteps=config.num_timesteps,
        laplacian_factor=config.laplacian_factor,
        mass_factor=config.mass_factor,
        #
        noise_type=config.noise_type,
        std_ratio=config.std_ratio,
        noise_amplitude_squared=config.noise_amplitude_squared,
        #
        spatial_dimension="1d",
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


if __name__ == "__main__":
    try:

        device = parser.parse_args().device
        config_path: str = parser.parse_args().config_path

        config_name = os.path.basename(config_path).replace(".yml", "")

        if "unet" in config_name:
            config = Lorenz96UnetConfig.load(config_path)
        else:
            raise ValueError(f"Unknown config name: {config_name}")
        set_seeds(config.seed)

        result_dir = f"{DL_EXPERIMENT_DIR_PATH}/{config_name}"
        os.makedirs(result_dir, exist_ok=True)

        logger.addHandler(FileHandler(f"{result_dir}/log.txt"))
        logger.info(f"Config: {config.to_json_str()}")

        trainer, dataset = initialize_trainer(
            config, device, str(ROOT_DIR), result_dir, kind="train"
        )

        start_time = time.time()
        trainer.train(log_fn=LossLogger(log_file=f"{result_dir}/loss.csv"))
        end_time = time.time()

        logger.info(f"\nTraining time: {(end_time - start_time) / 60.0:.2f} minutes")
        logger.info(f"Training completed. Model saved to {result_dir}")

    except Exception as e:
        logger.error("\n" + "*" * 50)
        logger.error("Error")
        logger.error("*" * 50)
        logger.error(e)
        logger.error(traceback.format_exc())
