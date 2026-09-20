from logging import getLogger
from typing import Literal, Optional

import torch
import torch.nn as nn

logger = getLogger(__name__)


class SlidingWindowWrapper(nn.Module):
    def __init__(self, model: nn.Module, window_size: int, missing_value: float = 0.0):
        super().__init__()
        self.model = model
        self.window_size = window_size
        self.missing_value = float(missing_value)
        logger.info(f"{self.window_size=}, {self.missing_value=}")

    def _sliding_window_stack(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, L = x.shape
        w = self.window_size
        x_padded = torch.full(
            (B, C, T + w - 1, L), self.missing_value, dtype=x.dtype, device=x.device
        )
        x_padded[:, :, w - 1 :, :] = x
        x_unfold = x_padded.unfold(dimension=2, size=w, step=1)  # (B, C, T, L, w)
        x_unfold = x_unfold.permute(0, 1, 4, 2, 3)  # (B, C, w, T, L)
        x_windowed = x_unfold.contiguous().view(B, C * w, T, L)
        return x_windowed

    def _permute_x(self, x_windowed: torch.Tensor):
        B, Cw, T, L = x_windowed.shape

        y = x_windowed.permute(0, 2, 1, 3)  # (B, T, Cw, L)
        y = y.contiguous().view(B * T, Cw, L)  # (B*T, Cw, L)

        return y

    def _broadcast_time(self, time: torch.Tensor, T: int) -> torch.Tensor:
        t = time[:, None].expand(-1, T)  # (B, T)
        t = t.contiguous().view(-1)  # (B*T,)
        return t

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        output_head: Literal["default", "closure", "noise", "both"] = "default",
        cond: Optional[torch.Tensor] = None,
        diffusion_std: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        x_windowed = self._sliding_window_stack(x)  # B, Cw, T, L (Cw = C * window_size)

        B, Cw, T, L = x_windowed.shape

        y = self._permute_x(x_windowed)  # (B*T, Cw, L)
        t = self._broadcast_time(time, T)

        y = self.model(x=y, time=t, output_head=output_head)  # (B*T, C, L)

        if isinstance(y, dict):
            out: dict[str, torch.Tensor] = {}
            for key, val in y.items():
                C_out = val.shape[1]
                val = val.view(B, T, C_out, L)
                out[key] = val.permute(0, 2, 1, 3).contiguous()
            return out

        C_out = y.shape[1]
        y = y.view(B, T, C_out, L)
        y = y.permute(0, 2, 1, 3)

        return y.contiguous()  # (B, C, T, L)
