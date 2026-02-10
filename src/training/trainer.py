import copy
import sys
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset

from src.models.ml.diffusion.gaussian_diffusion import GaussianDiffusion
from src.util.random_seed_helper import get_torch_generator, seed_worker

from .ema import EMA

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

logger = getLogger(__name__)


def noop(*args: Any, **kwargs: Any) -> None:
    pass


def cycle(dl: Any) -> Iterator[Any]:
    while True:
        for data in dl:
            yield data


class Trainer:

    def __init__(
        self,
        diffusion_model: GaussianDiffusion,
        dataset: Dataset,
        *,
        ema_decay: float = 0.995,
        train_batch_size: int = 32,
        train_lr: float = 1e-4,
        train_num_steps: int = 100_000,
        gradient_accumulate_every: int = 2,
        amp: bool = False,
        step_start_ema: int = 2_000,
        update_ema_every: int = 10,
        save_and_sample_every: int = 1_000,
        results_folder: str = "./results",
        max_grad_norm: Optional[float] = None,
        device: Optional[torch.device] = torch.device("cpu"),
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(beta=ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.device = device

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.ds = dataset
        self.dl = cycle(
            torch.utils.data.DataLoader(
                self.ds,
                batch_size=train_batch_size,
                drop_last=True,
                shuffle=True,
                pin_memory=True,
                num_workers=4,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(),
            )
        )

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 1

        self.amp = amp
        self.scaler = torch.GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        self.ema_model.load_state_dict(self.model.state_dict())

    def _step_ema(self) -> None:
        if self.step < self.step_start_ema:
            self._reset_parameters()
            return
        self.ema.update_model_average(ma_model=self.ema_model, current_model=self.model)

    def save(self, milestone: int) -> None:
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "scaler": self.scaler.state_dict(),
            "optimizer": self.opt.state_dict(),
        }
        path = str(self.results_folder / f"model-{milestone:07}.pt")
        torch.save(data, path)
        logger.info(f"Saved model to {path} (step = {self.step})")

    def save_only_model(self, milestone: int) -> None:
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
        }
        path = str(self.results_folder / f"model-{milestone:07}.pt")
        torch.save(data, path)
        logger.info(f"Saved model to {path} (step = {self.step})")

    def load(self, milestone: int, **kwargs: Any) -> None:
        if milestone == -1:
            all_milestones = [
                int(p.stem.split("-")[-1])
                for p in Path(self.results_folder).glob("**/*.pt")
            ]
            assert (
                len(all_milestones) > 0
            ), "need to have at least one milestone to load from latest checkpoint (milestone == -1)"
            milestone = max(all_milestones)

        path = str(self.results_folder / f"model-{milestone:07}.pt")
        data = torch.load(path, weights_only=False, map_location=self.device)

        self.step = data["step"]
        self.model.load_state_dict(data["model"], **kwargs)
        self.ema_model.load_state_dict(data["ema"], **kwargs)
        self.scaler.load_state_dict(data["scaler"])
        self.opt.load_state_dict(data["optimizer"])

        logger.info(f"Loaded model from {path} (step = {self.step})")

    def load_only_model(self, milestone: int, **kwargs: Any) -> None:
        if milestone == -1:
            all_milestones = [
                int(p.stem.split("-")[-1])
                for p in Path(self.results_folder).glob("**/*.pt")
            ]
            assert (
                len(all_milestones) > 0
            ), "need to have at least one milestone to load from latest checkpoint (milestone == -1)"
            milestone = max(all_milestones)

        path = str(self.results_folder / f"model-{milestone:07}.pt")
        data = torch.load(path, weights_only=False, map_location=self.device)

        self.step = data["step"]
        self.model.load_state_dict(data["model"], **kwargs)
        self._reset_parameters()

        logger.info(f"Loaded model from {path} (step = {self.step})")

    def train(
        self,
        prob_focus_present: Optional[float] = None,
        focus_present_mask: Optional[torch.Tensor] = None,
        log_fn: Callable = noop,
    ) -> None:
        assert callable(log_fn)

        _ = self.model.noise_estimate_fn.train()

        with tqdm(
            total=self.train_num_steps,
            desc="Training Progress",
            unit="step",
            initial=self.step,
        ) as pbar:
            while self.step <= self.train_num_steps:
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(self.device)

                    with torch.autocast(
                        device_type="cuda", dtype=torch.float16, enabled=self.amp
                    ):
                        loss = self.model(
                            data,
                            prob_focus_present=prob_focus_present,
                            focus_present_mask=focus_present_mask,
                        )
                        self.scaler.scale(
                            loss / self.gradient_accumulate_every
                        ).backward()

                log: Dict[str, Any] = {"step": self.step, "loss": loss.item()}

                if self.max_grad_norm is not None:
                    self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.max_grad_norm
                    )

                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

                if self.step % self.update_ema_every == 0:
                    self._step_ema()

                if self.step > 1 and self.step % self.save_and_sample_every == 0:
                    self.save_only_model(milestone=self.step)

                log_fn(log)
                self.step += 1
                pbar.set_postfix({"loss": loss.item()})
                pbar.update(1)

        logger.info("Training completed")
