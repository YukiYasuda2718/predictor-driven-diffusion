from functools import partial
from logging import getLogger
from typing import Callable, Iterable, List, Literal, Optional, Sequence

import torch
from torch import nn

from src.models.ml.networks.util.attention import Attention, Spatial2DLinearAttention
from src.models.ml.networks.util.basics import default, is_odd
from src.models.ml.networks.util.blocks import (
    Downsample2D,
    Residual,
    ResnetBlock2D,
    SinusoidalPosEmb,
    Upsample2D,
)
from src.models.ml.networks.util.einops_helpers import EinopsToAndFrom
from src.models.ml.networks.util.normalization import PreNorm1D, PreNorm2D
from src.models.ml.networks.util.periodic_conv import (
    PeriodicDownsample2D,
    PeriodicUpsampleConv2d,
)

logger = getLogger(__name__)


class Unet2D(nn.Module):
    def __init__(
        self,
        dim: int,
        nx: int,
        ny: int,
        padding_mode: Literal["zeros", "circular"] = "zeros",
        dim_mults: Sequence[int] = (1, 2, 4, 8),
        att_block_indices: Sequence[int] = (2, 3, 4),
        in_channels: int = 1,
        out_channels: int = 1,
        attn_heads: int = 8,
        init_dim: Optional[int] = None,
        init_kernel_size: int = 5,
        use_sparse_linear_attn: bool = True,
        time_base: float = 10000.0,
        has_last_bias: bool = True,
    ):
        super().__init__()
        logger.info(
            f"UNet 2D: {dim=}\n{nx=}, {ny=}\n{padding_mode=},{dim_mults=}, {att_block_indices=}\n{in_channels=}, {out_channels=},\n{attn_heads=},\n{init_dim=}, {init_kernel_size=},\n{use_sparse_linear_attn=},\n{time_base=}, {has_last_bias=}\n"
        )

        self.nx, self.ny = nx, ny
        self.in_channels = in_channels
        self.out_channels = out_channels

        init_dim = default(init_dim, dim)
        assert isinstance(init_dim, int)
        assert is_odd(init_kernel_size), "init kernel size must be odd"
        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv2d(
            self.in_channels,
            init_dim,
            kernel_size=init_kernel_size,
            padding=init_padding,
            padding_mode=padding_mode,
        )

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim, time_base),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs: Iterable[nn.Module] = nn.ModuleList([])
        self.ups: Iterable[nn.Module] = nn.ModuleList([])

        num_resolutions = len(in_out)
        block_class = ResnetBlock2D
        block_class_cond = partial(
            block_class, time_emb_dim=time_dim, padding_mode=padding_mode
        )

        Downsample: Callable[[int], nn.Module] = Downsample2D
        if padding_mode == "circular":
            logger.info("PeriodicDownsample2D is used.")
            Downsample = partial(PeriodicDownsample2D, kernel_size=5)

        use_attns = []
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            use_attn = False
            if ind in att_block_indices and use_sparse_linear_attn:
                use_attn = True
            use_attns.append(use_attn)
            logger.info(
                f"Downblock: {ind=}, {is_last=}, {dim_in=}, {dim_out=}, {use_attn=}"
            )

            down_spatial_attn: nn.Module = nn.Identity()
            if use_attn:
                down_spatial_attn = Residual(
                    PreNorm2D(
                        dim=dim_out,
                        fn=Spatial2DLinearAttention(dim_out, heads=attn_heads),
                    )
                )
                logger.info("Attention is added")

            self.downs.append(
                nn.ModuleList(
                    [
                        block_class_cond(dim_in, dim_out),
                        block_class_cond(dim_out, dim_out),
                        down_spatial_attn,
                        (Downsample(dim_out) if not is_last else nn.Identity()),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_class_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom(
            "b c h", "b h c", Attention(mid_dim, heads=attn_heads)
        )
        self.mid_spatial_attn = Residual(PreNorm1D(dim=mid_dim, fn=spatial_attn))
        self.mid_block2 = block_class_cond(mid_dim, mid_dim)

        Upsample: nn.Module | Callable[[int], nn.Module] = Upsample2D
        if padding_mode == "circular":
            logger.info("PeriodicUpsampleConv2d is used.")
            Upsample = partial(PeriodicUpsampleConv2d, kernel_size=5)

        for ind, ((dim_in, dim_out), use_attn) in enumerate(
            zip(reversed(in_out), reversed(use_attns))
        ):
            is_last = ind >= (num_resolutions - 1)
            logger.info(
                f"Upblock: {ind=}, {is_last=}, {dim_in=}, {dim_out=}, {use_attn=}"
            )

            up_spatial_attn: nn.Module = nn.Identity()
            if use_attn:
                up_spatial_attn = Residual(
                    PreNorm2D(
                        dim=dim_in,
                        fn=Spatial2DLinearAttention(dim_in, heads=attn_heads),
                    )
                )
                logger.info("Attention is added")

            self.ups.append(
                nn.ModuleList(
                    [
                        block_class_cond(dim_out * 2, dim_in),
                        block_class_cond(dim_in, dim_in),
                        up_spatial_attn,
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            block_class(dim * 2, dim, padding_mode=padding_mode),
            nn.Conv2d(dim, self.out_channels, kernel_size=1, bias=has_last_bias),
        )

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        **kwargs: Optional[dict],
    ) -> torch.Tensor:
        # x shape = b c (ny * nx)
        # time shape = b

        b, c, nyx = x.shape
        assert c == self.in_channels and nyx == self.ny * self.nx
        x_view = x.view(b, c, self.ny, self.nx)

        x = self.init_conv(x_view)
        r = x.clone()
        t = self.time_mlp(time)

        h: List[torch.Tensor] = []

        for downs in self.downs:
            assert isinstance(downs, nn.ModuleList)
            block1, block2, spatial_attn, downsample = downs
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        _b, _c, _y, _x = x.shape
        x = self.mid_spatial_attn(x.view(_b, _c, _y * _x))
        x = x.view(_b, _c, _y, _x)
        x = self.mid_block2(x, t)

        for ups in self.ups:
            assert isinstance(ups, nn.ModuleList)
            block1, block2, spatial_attn, upsample = ups
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        y = self.final_conv(x)
        assert y.shape == (b, self.out_channels, self.ny, self.nx)

        return y.view(b, self.out_channels, self.ny * self.nx)
