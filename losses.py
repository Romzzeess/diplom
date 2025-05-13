# losses.py
"""Функции потерь и вспомогательные метрики.

Включает:
    * dice_loss            — сегментационный штраф (1 − Dice).
    * tv_loss              — total variation для сглаживания карты.
    * combined_unet_loss   — MAE + Dice + TV (для U‑Net).
    * mlp_loss             — MSE + Focal (для MLP).

Все функции возвращают скаляр `torch.Tensor`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

# -----------------------------------------
# 1. Dice loss (binary)
# -----------------------------------------

def dice_loss(pred: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """1 − Dice коэффициент для бинарных карт.
    pred, target: [B,1,H,W] или broadcast совместимые.
    """
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()

# -----------------------------------------
# 2. Total Variation loss
# -----------------------------------------

def tv_loss(img: Tensor) -> Tensor:
    """Isotropic total variation loss for smoothing (batch‑averaged)."""
    dh = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
    dw = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
    return dh + dw

# -----------------------------------------
# 3. Combined loss for U‑Net
# -----------------------------------------

def combined_unet_loss(pred_map: Tensor, true_map: Tensor,
                        pred_r: Tensor, true_r: Tensor,
                        w_mae: float = 1.0,
                        w_dice: float = 2.0,
                        w_tv: float = 0.05,
                        w_r: float = 0.5) -> Tensor:
    mae = F.l1_loss(pred_map, true_map)
    dice = dice_loss(pred_map, true_map)
    tv = tv_loss(pred_map)
    mae_r = F.l1_loss(pred_r, true_r)
    return w_mae * mae + w_dice * dice + w_tv * tv + w_r * mae_r

# -----------------------------------------
# 4. Focal loss (binary scalar) для MLP‑выхода S̄
# -----------------------------------------

def focal_loss(pred: Tensor, target: Tensor, gamma: float = 2.0, eps: float = 1e-6) -> Tensor:
    """Focal loss для скалярной пригодности (B,). pred — после sigmoid."""
    pred = torch.clamp(pred, eps, 1.0 - eps)
    loss = - target * (1 - pred) ** gamma * torch.log(pred) - \
           (1 - target) * pred ** gamma * torch.log(1 - pred)
    return loss.mean()

# -----------------------------------------
# 5. Loss для MLP‑v1
# -----------------------------------------

def mlp_loss(pred_r: Tensor, true_r: Tensor,
             pred_s: Tensor, true_s: Tensor,
             w_r: float = 1.0, w_s: float = 2.0) -> Tensor:
    mae_r = F.l1_loss(pred_r, true_r)
    foc_s = focal_loss(pred_s, true_s)
    return w_r * mae_r + w_s * foc_s

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    b, h, w = 4, 64, 128
    out = torch.sigmoid(torch.randn(b, 1, h, w))
    tgt = torch.randint(0, 2, (b, 1, h, w)).float()
    print("Dice Loss:", dice_loss(out, tgt).item())
    print("TV Loss:", tv_loss(out).item())
