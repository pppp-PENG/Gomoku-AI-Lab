from dataclasses import asdict, dataclass
import torch
from torch import nn


@dataclass
class ModelConfig:
    architecture: str = "cnn"
    size: int = 9
    channels: int = 48
    blocks: int = 3
    dim: int = 64
    layers: int = 3
    heads: int = 4
    mlp_ratio: int = 2

    def __post_init__(self):
        if self.architecture not in ("cnn", "vit") or not 5 <= self.size <= 19:
            raise ValueError("invalid architecture or board size")
        if not 8 <= self.channels <= 128 or self.channels % 4 or not 1 <= self.blocks <= 8:
            raise ValueError("CNN resource budget exceeded")
        if not 8 <= self.dim <= 128 or not 1 <= self.layers <= 8 or not 1 <= self.heads <= 8 or self.dim % self.heads:
            raise ValueError("invalid transformer dimensions")
        if not 1 <= self.mlp_ratio <= 4:
            raise ValueError("invalid mlp_ratio")


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                 nn.GroupNorm(4, channels), nn.ReLU(),
                                 nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                 nn.GroupNorm(4, channels))

    def forward(self, x):
        return torch.relu(x + self.net(x))


class CNN(nn.Module):
    def __init__(self, c):
        super().__init__()
        if c.channels % 4:
            raise ValueError("channels must be divisible by 4")
        self.body = nn.Sequential(nn.Conv2d(3, c.channels, 3, padding=1), nn.ReLU(),
                                  *[Residual(c.channels) for _ in range(c.blocks)])
        self.policy = nn.Conv2d(c.channels, 1, 1)
        self.value = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                   nn.Linear(c.channels, c.channels), nn.ReLU(), nn.Linear(c.channels, 3))

    def forward(self, x):
        x = self.body(x)
        return self.policy(x).flatten(1), self.value(x)


class ViT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embed = nn.Linear(3, c.dim)
        self.position = nn.Parameter(torch.randn(1, c.size**2, c.dim) * .02)
        layer = nn.TransformerEncoderLayer(c.dim, c.heads, c.dim*c.mlp_ratio,
                                           dropout=0, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, c.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(c.dim)
        self.policy = nn.Linear(c.dim, 1)
        self.value = nn.Sequential(nn.Linear(c.dim, c.dim), nn.GELU(), nn.Linear(c.dim, 3))

    def forward(self, x):
        x = self.embed(x.flatten(2).transpose(1, 2)) + self.position
        x = self.norm(self.encoder(x))
        return self.policy(x).squeeze(-1), self.value(x.mean(1))


def build_model(config):
    model = CNN(config) if config.architecture == "cnn" else ViT(config)
    model.config = asdict(config)
    return model


def masked_policy(logits, legal):
    if not torch.all(legal.any(-1)):
        raise ValueError("policy requested for terminal board")
    return torch.softmax(logits.masked_fill(~legal.bool(), -torch.inf), -1)


def estimated_flops(c):
    """Analytic multiply-add=2 estimate; excludes norms, activations, softmax."""
    n = c.size**2
    if c.architecture == "cnn":
        macs = n*(3*c.channels*9 + c.blocks*2*c.channels*c.channels*9 + c.channels)
        macs += c.channels*c.channels + c.channels*3
    else:
        d = c.dim
        macs = n*3*d + c.layers*(4*n*d*d + 2*n*n*d + 2*n*d*d*c.mlp_ratio)
        macs += n*d + d*d + d*3
    return 2*macs
