from logging import getLogger
from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from .normalization import RMSNorm1D, RMSNorm2D

logger = getLogger(__name__)


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim: int, time_base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.time_base = time_base
        logger.info(f"{self.time_base=}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(self.time_base, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample2D(dim: int) -> nn.Module:
    return nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1)


def Upsample1D(dim: int) -> nn.Module:
    return nn.ConvTranspose1d(dim, dim, kernel_size=(4,), stride=(2,), padding=(1,))


def Downsample2D(dim: int) -> nn.Module:
    return nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)


def Downsample1D(dim: int, padding_mode: str = "zeros") -> nn.Module:
    return nn.Conv1d(
        dim, dim, kernel_size=(4,), stride=(2,), padding=(1,), padding_mode=padding_mode
    )


class Block2D(nn.Module):

    def __init__(self, dim: int, dim_out: int, padding_mode: str = "zeros"):
        super().__init__()
        self.proj = nn.Conv2d(
            dim, dim_out, kernel_size=3, padding=1, padding_mode=padding_mode
        )
        self.norm = RMSNorm2D(dim_out)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        scale_shift: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class Block1D(nn.Module):

    def __init__(self, dim: int, dim_out: int, padding_mode: str = "zeros"):
        super().__init__()
        self.proj = nn.Conv1d(
            dim, dim_out, kernel_size=3, padding=1, padding_mode=padding_mode
        )
        self.norm = RMSNorm1D(dim_out)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        scale_shift: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock2D(nn.Module):

    def __init__(
        self,
        dim: int,
        dim_out: int,
        *,
        time_emb_dim: Optional[int] = None,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block2D(dim, dim_out, padding_mode=padding_mode)
        self.block2 = Block2D(dim_out, dim_out, padding_mode=padding_mode)
        self.res_conv = (
            nn.Conv2d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        scale_shift = None
        if self.mlp is not None:
            assert time_emb is not None
            emb: torch.Tensor = self.mlp(time_emb)
            scale_shift = rearrange(emb, "b c -> b c 1 1").chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class ResnetBlock1D(nn.Module):

    def __init__(
        self,
        dim: int,
        dim_out: int,
        *,
        time_emb_dim: Optional[int] = None,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block1D(dim, dim_out, padding_mode=padding_mode)
        self.block2 = Block1D(dim_out, dim_out, padding_mode=padding_mode)
        self.res_conv = (
            nn.Conv1d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        scale_shift = None
        if self.mlp is not None:
            assert time_emb is not None
            emb: torch.Tensor = self.mlp(time_emb)
            scale_shift = rearrange(emb, "b c -> b c 1").chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)
