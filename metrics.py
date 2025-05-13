# metrics.py
"""Набор метрик для оценки моделей обстановки БМЗ.

Используются в evaluate.py и в раннем останове обучения.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Tuple

# -----------------------------------------------------------
# Полевая RMSE в метрах (Hs, S-map и т. д.)
# -----------------------------------------------------------

def rmse_field(pred: torch.Tensor | np.ndarray,
               target: torch.Tensor | np.ndarray) -> float:
    """Root-Mean-Square Error для карт [B,1,H,W] или [H,W]."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    mse = np.mean((pred - target) ** 2)
    return float(np.sqrt(mse))

# -----------------------------------------------------------
# Dice Score (IoU-подобная метрика) для бинарных карт
# -----------------------------------------------------------

def dice_score(pred_bin: torch.Tensor | np.ndarray,
               target_bin: torch.Tensor | np.ndarray,
               eps: float = 1e-6) -> float:
    if isinstance(pred_bin, torch.Tensor):
        pred_bin = pred_bin.cpu().numpy()
    if isinstance(target_bin, torch.Tensor):
        target_bin = target_bin.cpu().numpy()
    pred_bin = pred_bin.astype(bool)
    target_bin = target_bin.astype(bool)
    inter = np.logical_and(pred_bin, target_bin).sum()
    union = pred_bin.sum() + target_bin.sum()
    return float((2 * inter + eps) / (union + eps))

# -----------------------------------------------------------
# MAE для скалярного индекса риска R
# -----------------------------------------------------------

def mae_risk(pred: torch.Tensor | np.ndarray,
             target: torch.Tensor | np.ndarray) -> float:
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    return float(np.mean(np.abs(pred - target)))

# -----------------------------------------------------------
# AU-ROC / AU-PR для бинарной пригодности карты
# -----------------------------------------------------------

def auroc(pred: torch.Tensor | np.ndarray,
          target: torch.Tensor | np.ndarray) -> Tuple[float, float]:
    """Возвращает (AUROC, AUPR). pred – вероятность, target – 0/1."""
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy().ravel()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy().ravel()
    return (
        roc_auc_score(target, pred),
        average_precision_score(target, pred),
    )

# -----------------------------------------------------------
# Бутстреп-доверительный интервал
# -----------------------------------------------------------

def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval."""
    rng = np.random.default_rng(42)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = boots.mean(axis=1)
    lo = np.percentile(means, 100 * (alpha / 2))
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    a = torch.randn(10, 1, 64, 64)
    b = torch.randn(10, 1, 64, 64)
    print("RMSE field:", rmse_field(a, b))
