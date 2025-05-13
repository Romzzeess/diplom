# train.py
"""Обучение нейронных моделей для оценки обстановки БМЗ.

Поддерживает две архитектуры:
    * mlp  – MLPV1 (интегральный риск + грубая пригодность)
    * unet – UNetV2 (карта пригодности + риск)

Пример запуска:
    python train.py --model mlp --tensor data/processed/mbz_tensor.zarr --out models/mlp_v1.pt
    python train.py --model unet --tensor data/processed/mbz_tensor.zarr --out models/unet_v2.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from data_loader import DGGSDataModule
from models import MLPV1, UNetV2
from losses import mlp_loss, combined_unet_loss
from metrics import rmse_field

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_mlp(dm: DGGSDataModule, out_path: Path, epochs: int = 15):
    in_dim = len(dm.ds_train[0][0].view(-1))  # C*H*W ->, но мы храним агрегаты (B,d)
    model = MLPV1(in_dim).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=3e-4)
    best_rmse, patience, wait = float("inf"), 3, 0
    scaler = GradScaler()

    for epoch in range(1, epochs + 1):
        model.train()
        for X, Y in dm.train_dataloader():
            X = X.view(X.size(0), -1).to(DEVICE)
            true_r, true_s = Y["r_idx"].to(DEVICE), Y["s_map"].mean(dim=[2, 3]).to(DEVICE)
            opt.zero_grad()
            with autocast():
                pr_r, pr_s = model(X)
                loss = mlp_loss(pr_r, true_r, pr_s, true_s)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        # валидация
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, Y in dm.val_dataloader():
                X = X.view(X.size(0), -1).to(DEVICE)
                pr_r, pr_s = model(X)
                loss = mlp_loss(pr_r, Y["r_idx"].to(DEVICE), pr_s, Y["s_map"].mean(dim=[2, 3]).to(DEVICE))
                val_losses.append(loss.item())
        val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch}: val loss {val_loss:.4f}")
        if val_loss < best_rmse:
            best_rmse = val_loss
            torch.save(model.state_dict(), out_path)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break


def train_unet(dm: DGGSDataModule, out_path: Path, epochs: int = 30):
    in_c = dm.ds_train[0][0].shape[1]  # C
    model = UNetV2(in_c).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = GradScaler()
    best_metric, patience, wait = float("inf"), 5, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for X, Y in dm.train_dataloader():
            X = X.to(DEVICE)
            true_s, true_r = Y["s_map"].to(DEVICE), Y["r_idx"].to(DEVICE)
            opt.zero_grad()
            with autocast():
                pr_s, pr_r = model(X)
                loss = combined_unet_loss(pr_s, true_s, pr_r, true_r)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        # валидация: используем RMSE поля как ключевую метрику
        model.eval()
        rmses = []
        with torch.no_grad():
            for X, Y in dm.val_dataloader():
                X = X.to(DEVICE)
                pr_s, _ = model(X)
                rmse = rmse_field(pr_s, Y["s_map"])
                rmses.append(rmse)
        val_rmse = sum(rmses) / len(rmses)
        print(f"Epoch {epoch}: val RMSE {val_rmse:.3f}")
        if val_rmse < best_metric:
            best_metric = val_rmse
            torch.save(model.state_dict(), out_path)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP or UNet on MBZ tensor")
    parser.add_argument("--model", choices=["mlp", "unet"], required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train_years", nargs="*", type=int, default=[2023])
    parser.add_argument("--test_years", nargs="*", type=int, default=[2024])
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    dm = DGGSDataModule(args.tensor, train_years=args.train_years, test_years=args.test_years,
                        batch_size=args.batch)
    dm.setup()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    if args.model == "mlp":
        train_mlp(dm, out_path)
    else:
        train_unet(dm, out_path)
    print("Finished in", (time.time() - start) / 60, "min")
