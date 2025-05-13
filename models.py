# models.py
"""Нейронные архитектуры для оценки обстановки БМЗ.

Содержит:
    • MLPV1   – быстрая модель интегрального индекса риска.
    • UNetV2  – сверточный энкодер‑декодер для карт пригодности.

Все размеры каналов и гиперпараметры соответствуют главе 2.
Код совместим с PyTorch ≥ 2.0.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. MLP‑v1
# ---------------------------------------------------------

class MLPV1(nn.Module):
    """Полносвязный регрессор для индекса риска R и грубой карты S̄."""

    def __init__(self, in_features: int, hidden: tuple[int, ...] = (256, 128, 64),
                 dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head_r = nn.Linear(prev, 1)           # интегральный индекс R
        self.head_s = nn.Linear(prev, 1)           # грубая пригодность (скаляр)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, d]
        h = self.backbone(x)
        r = self.head_r(h)          # [B,1] – без активации (MAE)
        s_bar = torch.sigmoid(self.head_s(h))  # [B,1]
        return r.squeeze(-1), s_bar.squeeze(-1)

# ---------------------------------------------------------
# 2. U‑Net‑v2
# ---------------------------------------------------------

def _conv_block(in_c: int, out_c: int, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

class UNetV2(nn.Module):
    """U‑Net: 5 down / 4 up, dilated bottleneck."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        filters = [32, 64, 128, 256, 512]

        self.conv_down1 = _conv_block(in_channels, filters[0])
        self.conv_down2 = _conv_block(filters[0], filters[1])
        self.conv_down3 = _conv_block(filters[1], filters[2])
        self.conv_down4 = _conv_block(filters[2], filters[3])
        self.conv_down5 = _conv_block(filters[3], filters[4], dilation=2)  # bottleneck

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)
        self.conv_up4 = _conv_block(filters[4], filters[3])
        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.conv_up3 = _conv_block(filters[3], filters[2])
        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.conv_up2 = _conv_block(filters[2], filters[1])
        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.conv_up1 = _conv_block(filters[1], filters[0])

        self.out_map = nn.Conv2d(filters[0], 1, 1)  # карта S (1 канал, сигмоид)
        self.out_r = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(filters[4], 1),
        )  # индекс R из bottleneck

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Down path
        c1 = self.conv_down1(x)
        p1 = self.pool(c1)
        c2 = self.conv_down2(p1)
        p2 = self.pool(c2)
        c3 = self.conv_down3(p2)
        p3 = self.pool(c3)
        c4 = self.conv_down4(p3)
        p4 = self.pool(c4)
        c5 = self.conv_down5(p4)  # bottleneck

        # Up path
        u4 = self.up4(c5)
        u4 = torch.cat([u4, c4], dim=1)
        c6 = self.conv_up4(u4)
        u3 = self.up3(c6)
        u3 = torch.cat([u3, c3], dim=1)
        c7 = self.conv_up3(u3)
        u2 = self.up2(c7)
        u2 = torch.cat([u2, c2], dim=1)
        c8 = self.conv_up2(u2)
        u1 = self.up1(c8)
        u1 = torch.cat([u1, c1], dim=1)
        c9 = self.conv_up1(u1)

        s_map = torch.sigmoid(self.out_map(c9))  # [B,1,H,W]
        r_idx = self.out_r(c5).squeeze(-1)       # [B]
        return s_map, r_idx

# ---------------------------------------------------------
if __name__ == "__main__":
    B, C, H, W = 2, 3, 128, 256
    x = torch.randn(B, C, H, W)
    model = UNetV2(C)
    s, r = model(x)
    print("map", s.shape, "idx", r.shape)

    mlp = MLPV1(120)
    y_r, y_s = mlp(torch.randn(B, 120))
    print(y_r.shape, y_s.shape)
