from typing import Any, Dict

import torch
from einops import rearrange
from torch import nn


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops: str, to_einops: str, fn: nn.Module):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        shape = x.shape

        from_einops_parts = self.from_einops.split(" ")
        if len(from_einops_parts) != len(shape):
            raise ValueError()
        reconstitute_kwargs: Dict[str, int] = dict(zip(from_einops_parts, shape))

        x = rearrange(x, f"{self.from_einops} -> {self.to_einops}")
        x = self.fn(x, **kwargs)
        x = rearrange(
            x, f"{self.to_einops} -> {self.from_einops}", **reconstitute_kwargs
        )
        return x
