# evaluate.py
"""Оценка обученных моделей на тестовой выборке.

Пример:
    python evaluate.py --tensor data/processed/mbz_tensor.zarr \
                       --mlp models/mlp_v1.pt \
                       --unet models/unet_v2.pt \
                       --out reports/eval_report.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from data_loader import DGGSDataModule
from models import MLPV1, UNetV2
from metrics import (rmse_field, dice_score, mae_risk, auroc,
                     bootstrap_ci)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models(mlp_path: Path | None, unet_path: Path | None, dm: DGGSDataModule):
    mlp, unet = None, None
    if mlp_path is not None:
        in_dim = len(dm.ds_test[0][0].view(-1))
        mlp = MLPV1(in_dim).to(DEVICE)
        mlp.load_state_dict(torch.load(mlp_path, map_location=DEVICE))
        mlp.eval()
    if unet_path is not None:
        in_c = dm.ds_test[0][0].shape[1]
        unet = UNetV2(in_c).to(DEVICE)
        unet.load_state_dict(torch.load(unet_path, map_location=DEVICE))
        unet.eval()
    return mlp, unet


def evaluate(dm: DGGSDataModule, mlp: MLPV1 | None, unet: UNetV2 | None,
             threshold: float = 0.7) -> Dict[str, float]:
    rmse_list, dice_list, risk_list = [], [], []
    auroc_scores, aupr_scores = [], []

    with torch.no_grad():
        for X, Y in dm.test_dataloader():
            X_device = X.to(DEVICE)
            true_s = Y["s_map"].to(DEVICE)
            true_r = Y["r_idx"].to(DEVICE)

            if unet is not None:
                pred_s, pred_r_u = unet(X_device)
            else:
                pred_s = torch.zeros_like(true_s)
                pred_r_u = torch.zeros_like(true_r)

            if mlp is not None:
                X_flat = X_device.view(X_device.size(0), -1)
                pred_r_mlp, pred_s_scalar = mlp(X_flat)
                # добавляем скалярную пригодность в карту как однородное поле
                pred_s += pred_s_scalar.view(-1, 1, 1, 1)
                pred_r = 0.5 * (pred_r_u + pred_r_mlp)
            else:
                pred_r = pred_r_u

            rmse_list.append(rmse_field(pred_s, true_s))
            mae_r = mae_risk(pred_r, true_r)
            risk_list.append(mae_r)

            # Dice и ROC
            dice_list.append(dice_score((pred_s > threshold).cpu(), (true_s > threshold).cpu()))
            au, ap = auroc(pred_s.cpu(), (true_s > threshold).cpu())
            auroc_scores.append(au)
            aupr_scores.append(ap)

    # средние значения
    metrics = {
        "rmse_S": float(np.mean(rmse_list)),
        "rmse_S_ci": bootstrap_ci(np.array(rmse_list)),
        "mae_R": float(np.mean(risk_list)),
        "mae_R_ci": bootstrap_ci(np.array(risk_list)),
        "dice": float(np.mean(dice_list)),
        "auroc": float(np.mean(auroc_scores)),
        "aupr": float(np.mean(aupr_scores)),
    }
    return metrics


def make_pdf_report(metrics: Dict[str, float | tuple], out_pdf: Path):
    with PdfPages(out_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(6, 3))
        keys = ["rmse_S", "mae_R", "dice", "auroc", "aupr"]
        values = [metrics[k] if not isinstance(metrics[k], tuple) else metrics[k][0] for k in keys]
        ax.bar(keys, values)
        ax.set_ylabel("value")
        ax.set_title("Summary metrics (test set)")
        pdf.savefig(fig)
        plt.close(fig)

        # вывод текста
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis("off")
        lines = [f"{k}: {v if not isinstance(v, tuple) else f'{v[0]:.3f} [{v[0]-v[1]:.3f},{v[2]-v[0]:.3f}]'}"
                 for k, v in metrics.items()]
        txt = "\n".join(lines)
        ax.text(0.01, 0.99, txt, va="top", ha="left", fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models on test subset")
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--mlp", default=None)
    parser.add_argument("--unet", default=None)
    parser.add_argument("--out", default="reports/eval_report.pdf")
    parser.add_argument("--train_years", nargs="*", type=int, default=[2023])
    parser.add_argument("--test_years", nargs="*", type=int, default=[2024])
    args = parser.parse_args()

    dm = DGGSDataModule(args.tensor, train_years=args.train_years, test_years=args.test_years, batch_size=4)
    dm.setup()

    mlp_path = Path(args.mlp) if args.mlp else None
    unet_path = Path(args.unet) if args.unet else None
    mlp_model, unet_model = load_models(mlp_path, unet_path, dm)

    metrics = evaluate(dm, mlp_model, unet_model)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    make_pdf_report(metrics, Path(args.out))

    # также сохраняем JSON
    with Path(args.out).with_suffix(".json").open("w") as fp:
        json.dump(metrics, fp, indent=2)
    print("Evaluation complete. Report stored in", args.out)
