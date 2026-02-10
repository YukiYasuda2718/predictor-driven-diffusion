import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm2D(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize along the channel dimension
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class LayerNorm1D(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize along the channel dimension
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class RMSNorm2D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize along the channel dimension
        return F.normalize(x, dim=1) * self.scale * self.gamma


class RMSNorm1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize along the channel dimension
        return F.normalize(x, dim=1) * self.scale * self.gamma


class PreNorm2D(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm2D(dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.norm(x)
        return self.fn(x, **kwargs)


class PreNorm1D(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm1D(dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.norm(x)
        return self.fn(x, **kwargs)
