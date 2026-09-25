import dataclasses

from .lorenz96_config import Lorenz96UnetConfig


@dataclasses.dataclass()
class Lorenz96SaPddUnetConfig(Lorenz96UnetConfig):
    drift_loss_weight: float = 0.1
    score_loss_weight: float = 1.0
    num_head_blocks: int = 0
