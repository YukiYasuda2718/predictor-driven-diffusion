import dataclasses

from .kolmogorov_flow_config import KolmogorovFlowUnetConfig


@dataclasses.dataclass()
class KolmogorovFlowSaPddUnetConfig(KolmogorovFlowUnetConfig):
    drift_loss_weight: float = 0.1
    score_loss_weight: float = 1.0
    num_head_blocks: int = 0
