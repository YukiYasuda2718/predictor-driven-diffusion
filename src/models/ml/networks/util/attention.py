from typing import Optional

import torch
from einops import rearrange
from torch import einsum, nn


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        rotary_emb: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        pos_bias: Optional[torch.Tensor] = None,
        focus_present_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n, device = x.shape[-2], x.device

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        if (focus_present_mask is not None) and focus_present_mask.all():
            # If all batch samples are focusing on the present token,
            # it is equivalent to directly passing the token's values (v) through to the output.
            return self.to_out(v)

        q = rearrange(q, "... n (h d) -> ... h n d", h=self.heads)
        k = rearrange(k, "... n (h d) -> ... h n d", h=self.heads)
        v = rearrange(v, "... n (h d) -> ... h n d", h=self.heads)

        q = q * self.scale

        # Rotate positions into queries and keys for time attention
        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # Compute similarity between queries and keys
        sim = einsum("... h i d, ... h j d -> ... h i j", q, k)

        if pos_bias is not None:
            sim = sim + pos_bias

        if (focus_present_mask is not None) and (not (~focus_present_mask).all()):
            # If some batch samples are focusing only on the present token:
            # - Create a mask to restrict attention to the present token for those samples.
            # - For other samples, allow attention to all tokens.

            attend_all_mask = torch.ones((n, n), device=device, dtype=torch.bool)
            # Full attention mask (all tokens attendable)

            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)
            # Self-attention mask (only the current token)

            # Combine the masks based on the focus_present_mask:
            # - If focus_present_mask is True, use attend_self_mask.
            # - Otherwise, use attend_all_mask.
            mask = torch.where(
                rearrange(focus_present_mask, "b -> b 1 1 1"),
                rearrange(attend_self_mask, "i j -> 1 1 i j"),
                rearrange(attend_all_mask, "i j -> 1 1 i j"),
            )

            # Mask out positions that should not be attended to by setting them to a very large negative value.
            # This ensures that the softmax operation effectively ignores these positions.
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # For numerical stability, subtract the maximum value in each row before applying softmax.
        # This prevents potential overflow issues during the exponential operation in softmax.
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # Aggregate values using the attention weights
        out = einsum("... h i j, ... h j d -> ... h i d", attn, v)
        out = rearrange(out, "... h n d -> ... n (h d)")
        return self.to_out(out)


class Spatial2DLinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, ny, nx = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q = rearrange(q, "b (h c) y x -> b h c (y x)", h=self.heads, y=ny, x=nx)
        k = rearrange(k, "b (h c) y x -> b h c (y x)", h=self.heads, y=ny, x=nx)
        v = rearrange(v, "b (h c) y x -> b h c (y x)", h=self.heads, y=ny, x=nx)

        q = q.softmax(dim=-2)  # along the channel dim
        k = k.softmax(dim=-1)  # along the spatial dim

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        # inner product among the spatial dims

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        # inner product among the channel dims

        out = rearrange(out, "b h c (y x) -> b (h c) y x", h=self.heads, y=ny, x=nx)
        return self.to_out(out)


class Spatial1DLinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q = rearrange(q, "b (h c) x -> b h c x", h=self.heads)
        k = rearrange(k, "b (h c) x -> b h c x", h=self.heads)
        v = rearrange(v, "b (h c) x -> b h c x", h=self.heads)

        q = q.softmax(dim=-2)  # along the channel dim
        k = k.softmax(dim=-1)  # along the spatial dim

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        # inner product among the spatial dims

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        # inner product among the channel dims

        out = rearrange(out, "b h c x -> b (h c) x", h=self.heads, x=h)
        return self.to_out(out)
