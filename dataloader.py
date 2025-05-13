# data_loader.py
"""DGGSDataModule: загрузка Zarr‑тензора и генерация PyTorch‑датасетов
с почасовой дискретизацией и temporal block‑split.

Формат входного Zarr, созданного dataset_pipeline.py:
    dims:  time × lat × lon
    vars:  u10, swh, ais_density, h3 (индекс),
           (опционально) s_mask (карта пригодности), r_index (скалярный риск)

Выход `__getitem__` → tuple(X, Y):
    X : torch.Tensor [C, H, W]
    Y : dict {"s_map":[1,H,W], "r_idx":[1]}

Использование:
    dm = DGGSDataModule('data/processed/mbz_tensor.zarr',
                        train_years=[2023], test_years=[2024])
    dm.setup()
    train_loader = dm.train_dataloader(batch_size=8)
"""
from __future__ import annotations

import datetime as dt
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr

# -------------------- Dataset -------------------------------------------------

class HourlyDGGSSet(Dataset):
    """Torch‑Dataset для одной выборки (train / val / test)."""

    def __init__(self, ds: xr.Dataset, indices: np.ndarray, aug: bool = False):
        self.ds = ds
        self.idxs = indices  # индексы времени
        self.aug = aug
        # каналы входа (порядок фиксируем)
        self.in_vars = [v for v in ["u10", "swh", "ais_density"] if v in ds]

    def __len__(self):
        return len(self.idxs)

    def _augment(self, x: torch.Tensor, s_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # простые произвольные отражения (дид, хор) и повороты 90°
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])  # горизонтальное зеркало
            s_map = torch.flip(s_map, dims=[2])
        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])  # вертикальное зеркало
            s_map = torch.flip(s_map, dims=[1])
        if random.random() < 0.5:
            x = torch.rot90(x, k=1, dims=[1, 2])
            s_map = torch.rot90(s_map, k=1, dims=[1, 2])
        return x, s_map

    def __getitem__(self, idx: int):
        t_idx = self.idxs[idx]
        sample = self.ds.isel(time=t_idx)
        # --- X: нормализуем в [0,1] или Z‑score
        x_stack = []
        for v in self.in_vars:
            arr = sample[v].values.astype(np.float32)
            if v == "u10":
                arr /= 30.0  # примитивная н‑ция (максимум 30 м/с)
            elif v == "swh":
                arr /= 10.0
            elif v == "ais_density":
                arr = np.log1p(arr) / 6.0  # плотность обычно до ~400
            x_stack.append(arr)
        X = torch.from_numpy(np.stack(x_stack))  # [C,H,W]

        # --- Y
        if "s_mask" in sample:
            y_map = torch.from_numpy(sample["s_mask"].values.astype(np.float32)[None, ...])
        else:
            y_map = torch.zeros(1, *X.shape[1:], dtype=torch.float32)
        if "r_index" in sample:
            r_idx = torch.tensor([sample["r_index"].item()], dtype=torch.float32)
        else:
            r_idx = torch.tensor([0.0], dtype=torch.float32)

        if self.aug:
            X, y_map = self._augment(X, y_map)
        return X, {"s_map": y_map, "r_idx": r_idx}

# -------------------- DataModule ---------------------------------------------

class DGGSDataModule:
    """Простой аналог PyTorch‑Lightning DataModule для загрузки DGGS Zarr."""

    def __init__(self,
                 tensor_path: str | Path,
                 train_years: List[int],
                 test_years: List[int],
                 batch_size: int = 8,
                 num_workers: int = 2):
        self.tensor_path = Path(tensor_path)
        self.train_years = train_years
        self.test_years = test_years
        self.val_years = []  # можно задать отдельно, иначе 20 % от train
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self):
        ds = xr.open_zarr(self.tensor_path, consolidated=False)
        self.ds = ds
        times = pd.to_datetime(ds.time.values)
        year = times.year
        train_mask = np.isin(year, self.train_years)
        test_mask = np.isin(year, self.test_years)
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        # 20 % последних train — на валидацию
        if len(train_idx) > 0:
            cut = int(len(train_idx) * 0.8)
            self.idx_train = train_idx[:cut]
            self.idx_val = train_idx[cut:]
        else:
            self.idx_train, self.idx_val = np.array([], int), np.array([], int)
        self.idx_test = test_idx

        self.ds_train = HourlyDGGSSet(ds, self.idx_train, aug=True)
        self.ds_val = HourlyDGGSSet(ds, self.idx_val, aug=False)
        self.ds_test = HourlyDGGSSet(ds, self.idx_test, aug=False)

    # --- loaders
    def train_dataloader(self, shuffle: bool = True):
        return DataLoader(self.ds_train, batch_size=self.batch_size,
                          shuffle=shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    dm = DGGSDataModule("data/processed/mbz_tensor.zarr", train_years=[2023], test_years=[2024])
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    X, Y = batch
    print("X", X.shape, "s_map", Y["s_map"].shape, "r", Y["r_idx"].shape)
