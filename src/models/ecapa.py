"""
ECAPA-TDNN multi-task backbone for VocalMorph.

Input:
  features: (B, T, D) frame-level acoustic features
Output:
  dict with keys: height, weight, age, gender_logits
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TDNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    def __init__(self, channels: int, se_channels: int = 128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, se_channels, kernel_size=1)
        self.fc2 = nn.Conv1d(se_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class SERes2Block(nn.Module):
    """
    Lightweight SE-Res2 block inspired by ECAPA-TDNN.
    """

    def __init__(self, channels: int, scale: int = 8, kernel_size: int = 3, dilation: int = 2):
        super().__init__()
        if channels % scale != 0:
            raise ValueError(f"channels ({channels}) must be divisible by scale ({scale})")

        self.scale = scale
        width = channels // scale

        self.pre = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            [TDNNBlock(width, width, kernel_size=kernel_size, dilation=dilation) for _ in range(scale - 1)]
        )
        self.post = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.se = SEBlock(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.pre(x)
        splits = torch.chunk(x, self.scale, dim=1)

        out = [splits[0]]
        running = splits[0]
        for i in range(1, self.scale):
            running = self.blocks[i - 1](splits[i] + running)
            out.append(running)

        x = torch.cat(out, dim=1)
        x = self.post(x)
        x = self.se(x)
        x = x + identity
        return self.act(x)


class AttentiveStatsPooling(nn.Module):
    def __init__(self, in_channels: int, bottleneck_channels: int = 128):
        super().__init__()
        self.tdnn = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1)
        self.attn = nn.Conv1d(bottleneck_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, C, T)
        padding_mask: (B, T), True for padded frames
        """
        e = torch.tanh(self.tdnn(x))
        a = self.attn(e)  # (B, C, T)

        if padding_mask is not None:
            invalid = padding_mask.unsqueeze(1)  # (B, 1, T)
            a = a.masked_fill(invalid, -1e9)

        alpha = torch.softmax(a, dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x - mean.unsqueeze(2)) ** 2, dim=2).clamp(min=1e-9)
        std = torch.sqrt(var)
        return torch.cat([mean, std], dim=1)  # (B, 2C)


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, n_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ECAPAMultiTask(nn.Module):
    """
    ECAPA-TDNN backbone with 3 regression heads + 1 classification head.
    """

    def __init__(
        self,
        input_dim: int,
        channels: int = 512,
        emb_dim: int = 192,
        scale: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.front = TDNNBlock(input_dim, channels, kernel_size=5, dilation=1)
        self.block1 = SERes2Block(channels, scale=scale, dilation=2)
        self.block2 = SERes2Block(channels, scale=scale, dilation=3)
        self.block3 = SERes2Block(channels, scale=scale, dilation=4)

        self.mfa = nn.Sequential(
            nn.Conv1d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.pool = AttentiveStatsPooling(channels)
        self.bn = nn.BatchNorm1d(channels * 2)
        self.embedding = nn.Linear(channels * 2, emb_dim)

        self.height_head = RegressionHead(emb_dim, hidden_dim=256, dropout=dropout)
        self.weight_head = RegressionHead(emb_dim, hidden_dim=256, dropout=dropout)
        self.age_head = RegressionHead(emb_dim, hidden_dim=256, dropout=dropout)
        self.gender_head = ClassificationHead(emb_dim, hidden_dim=256, n_classes=2, dropout=dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        features: (B, T, D)
        padding_mask: (B, T), True for padded frames
        """
        x = features.transpose(1, 2)  # (B, D, T)
        x = self.front(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        x = self.mfa(torch.cat([x1, x2, x3], dim=1))
        x = self.pool(x, padding_mask=padding_mask)
        x = self.bn(x)
        emb = self.embedding(x)

        return {
            "height": self.height_head(emb),
            "weight": self.weight_head(emb),
            "age": self.age_head(emb),
            "gender_logits": self.gender_head(emb),
        }
