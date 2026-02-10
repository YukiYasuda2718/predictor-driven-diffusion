import argparse
import os
import pathlib
import time
import traceback
from logging import INFO, FileHandler, getLogger
from typing import Literal

import torch

from scripts.make_data_kolmogorov_flow import DT as dt
from scripts.make_data_kolmogorov_flow import LX, LY, N_OUT_STEPS
from src.configs.kolmogorov_flow_config import KolmogorovFlowUnetConfig
from src.datasets.dataset_kolmogorov_flow import DatasetKolmogorovFlow
from src.models.ml.diffusion.gaussian_diffusion import GaussianDiffusion
from src.models.ml.networks.sliding_window_wrapper import SlidingWindowWrapper
from src.models.ml.networks.unet_2d import Unet2D
from src.models.ml.score.score_kolmogorov_flow import ScoreKolmogorovFlow
from src.training.loss_logger import LossLogger
from src.training.trainer import Trainer
from src.util.random_seed_helper import set_seeds

DT = N_OUT_STEPS * dt
assert LX == LY
L_SPACE = LX
CHANNELS = 1
del dt

os.environ["CUBLAS_WORKSPACE_CONFIG"] = r":4096:8"  # to make calculations deterministic

ROOT_DIR = pathlib.Path(os.environ["PYTHONPATH"].split(":")[0]).resolve()

DL_EXPERIMENT_DIR_PATH = f"{ROOT_DIR}/data/DL_model/kolmogorov_flow"

logger = getLogger()
logger.setLevel(INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda:0")


def make_dataset(
    config: KolmogorovFlowUnetConfig,
    root_dir: str,
    kind: Literal["train", "valid", "test"],
):
    assert kind in ["train", "valid", "test"]

    data_dir_path = f"{root_dir}/data/DL_data/{config.data_dir_name}"
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
    config: KolmogorovFlowUnetConfig,
    device: str,
    root_dir: str,
    result_dir: str,
    kind: Literal["train", "valid", "test"],
):

    dataset = make_dataset(config, root_dir, kind)

    if isinstance(config, KolmogorovFlowUnetConfig):
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
    else:
        raise ValueError(f"Unknown config type: {type(config)}")

    noise_estimate_fn = ScoreKolmogorovFlow(
        surrogate_model=model.to(device),
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

    diffusion = GaussianDiffusion(
        noise_estimate_fn=noise_estimate_fn.to(device),
        #
        channels=CHANNELS,
        num_frames=config.nt,
        image_size=0,  # not used
        #
        image_size_x=config.nx,
        image_size_y=config.ny,
        x_length=L_SPACE,
        #
        num_timesteps=config.num_timesteps,
        laplacian_factor=config.laplacian_factor,
        mass_factor=config.mass_factor,
        #
        noise_type=config.noise_type,
        std_ratio=config.std_ratio,
        noise_amplitude_squared=config.noise_amplitude_squared,
        #
        spatial_dimension="2d",
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
            config = KolmogorovFlowUnetConfig.load(config_path)
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
