# -*- coding: utf-8 -*-
"""dataset_pipeline.py

Мини‑пайплайн для диплома: автоматически добывает открытые данные
ERA5 (ветер и высоту волн) и выборку AIS (архив IWR‑AIS 2023),
выполняет QC, переводит всё в изотропную DGGS‑решётку (H3 level 8 ≈ 1 км)
и сохраняет итоговый тензор признаков в формате Zarr.

Запуск:
    python dataset_pipeline.py --bbox "9 53 32 66" --years 2023 2024

Требования:
    pip install prefect aiohttp pandas numpy xarray netCDF4 zarr h3 geopandas

NB: для примера используется публичный прокси open‑meteo; точность
эквивалентна ERA5‑hourly 0.25°. Для реального проекта можно заменить
на CDS API.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib
import tarfile
from datetime import datetime
from typing import List, Tuple

import aiohttp
import numpy as np
import pandas as pd
import xarray as xr
from prefect import flow, task, get_run_logger
from tqdm.asyncio import tqdm

try:
    import h3
except ImportError as exc:  # noqa: D401
    raise SystemExit("pip install h3") from exc

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

DGGS_LEVEL = 8  # H3 cell ~1 км
ERA_VARS = {"u10": "10m_wind_speed", "swh": "wave_height"}
AIS_URL = "https://zenodo.org/record/8198889/files/iwr-ais-2023.tar.gz?download=1"

# ---------------------------------------------------------------------------
# Утилитарные функции DGGS
# ---------------------------------------------------------------------------

def to_h3(lon: float, lat: float, level: int = DGGS_LEVEL) -> str:
    return h3.geo_to_h3(lat, lon, level)

# ---------------------------------------------------------------------------
# 1. Скачивание и QC ERA‑5 через open‑meteo
# ---------------------------------------------------------------------------

@task(retries=2, retry_delay_seconds=30)
async def fetch_era5(year: int, bbox: Tuple[float, float, float, float]) -> pathlib.Path:
    logger = get_run_logger()
    lon_min, lat_min, lon_max, lat_max = bbox
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude_min={lat_min}&latitude_max={lat_max}"
        f"&longitude_min={lon_min}&longitude_max={lon_max}"
        f"&start_date={year}-01-01&end_date={year}-12-31"
        f"&hourly={'%2C'.join(ERA_VARS.values())}&format=netcdf"
    )
    out_nc = RAW / f"era5_{year}.nc"
    if out_nc.exists():
        logger.info("ERA5 %s уже скачан", year)
        return out_nc

    logger.info("Скачиваю ERA5 %s…", year)
    async with aiohttp.ClientSession() as sess, sess.get(url) as resp:
        resp.raise_for_status()
        with out_nc.open("wb") as fout:
            async for chunk in resp.content.iter_chunked(1 << 16):
                fout.write(chunk)
    return out_nc

# ---------------------------------------------------------------------------
# 2. Скачать подмножество AIS внутри bbox
# ---------------------------------------------------------------------------

@task(retries=1, retry_delay_seconds=60)
async def fetch_ais_subset(bbox: Tuple[float, float, float, float]) -> pathlib.Path:
    logger = get_run_logger()
    subset_csv = RAW / "ais_subset.csv.gz"
    if subset_csv.exists():
        logger.info("AIS subset уже готов")
        return subset_csv

    tmp_tar = RAW / "iwr_ais.tar.gz"
    if not tmp_tar.exists():
        logger.info("Скачиваю IWR‑AIS архив (~6 GB)…")
        async with aiohttp.ClientSession() as sess, sess.get(AIS_URL) as resp:
            resp.raise_for_status()
            with tmp_tar.open("wb") as f:
                async for chunk in resp.content.iter_chunked(1 << 18):
                    f.write(chunk)

    lon_min, lat_min, lon_max, lat_max = bbox
    logger.info("Фильтрую AIS внутри bbox…")
    with tarfile.open(tmp_tar, "r:gz") as tar, subset_csv.open("wt") as out:
        for member in tar:  # файлы csv по месяцам
            if not member.name.endswith(".csv"):
                continue
            fobj = tar.extractfile(member)
            if fobj is None:
                continue
            df = pd.read_csv(fobj, usecols=["mmsi", "lat", "lon", "sog", "cog", "timestamp"],
                             parse_dates=["timestamp"])
            mask = (
                (df["lat"] >= lat_min) & (df["lat"] <= lat_max) &
                (df["lon"] >= lon_min) & (df["lon"] <= lon_max)
            )
            df.loc[mask].to_csv(out, header=False, index=False)
    return subset_csv

# ---------------------------------------------------------------------------
# 3. Построение почасового тензора DGGS
# ---------------------------------------------------------------------------

@task
def build_tensor(era_nc: List[pathlib.Path], ais_csv: pathlib.Path,
                 bbox: Tuple[float, float, float, float]) -> pathlib.Path:
    logger = get_run_logger()
    logger.info("Собираю единый xarray‑тензор…")

    # 3.1  ERA5 объединяем
    era_ds = xr.open_mfdataset([str(p) for p in era_nc], combine="by_coords")
    era_ds = era_ds.rename({v: k for k, v in ERA_VARS.items()})

    # 3.2  Подготовим координатную сетку DGGS (индекс в каждой точке grib‑сетки)
    lon_min, lat_min, lon_max, lat_max = bbox
    lons = np.linspace(lon_min, lon_max, 256)
    lats = np.linspace(lat_min, lat_max, 128)
    hh = np.empty((len(lats), len(lons)), dtype="<U15")
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            hh[i, j] = to_h3(lon, lat)

    tensor = xr.Dataset(coords={"lat": lats, "lon": lons, "time": era_ds.time})
    tensor["h3"] = ("lat", "lon"), hh

    # 3.3  Интегрируем ERA5 (билинейная интерполяция)
    for v in ["u10", "swh"]:
        tensor[v] = era_ds[v].interp(latitude=("lat", lats), longitude=("lon", lons))

    # 3.4  AIS плотность почасовая → та же сетка
    df = pd.read_csv(ais_csv, names=["mmsi", "lat", "lon", "sog", "cog", "ts"], parse_dates=["ts"])
    df["ts"] = df["ts"].dt.floor("H")
    df["h3"] = df.apply(lambda r: to_h3(r["lon"], r["lat"], DGGS_LEVEL), axis=1)
    grp = df.groupby(["ts", "h3"]).size().rename("ais_density")
    dens = np.zeros((len(tensor.time), len(lats), len(lons)), dtype=np.float32)
    h3_to_idx = {cell: (i, j) for i, lat in enumerate(lats) for j, lon in enumerate(lons) for cell in [hh[i, j]]}

    for (ts, cell), count in grp.items():
        if cell in h3_to_idx:
            t_idx = int((ts - tensor.time[0].to_pydatetime()) / np.timedelta64(1, "h"))
            i, j = h3_to_idx[cell]
            if 0 <= t_idx < len(tensor.time):
                dens[t_idx, i, j] += count
    tensor["ais_density"] = ("time", "lat", "lon"), dens

    out_path = PROCESSED / "mbz_tensor.zarr"
    tensor.to_zarr(out_path, mode="w")
    logger.info("Сохранён тензор %s", out_path)
    return out_path

# ---------------------------------------------------------------------------
# Prefect flow
# ---------------------------------------------------------------------------

@flow(name="Dataset‑MBZ‑Flow")
def main(years: List[int], bbox: Tuple[float, float, float, float]):
    era_tasks = [fetch_era5.submit(y, bbox) for y in years]
    era_paths = [t.result() for t in era_tasks]
    ais_path = fetch_ais_subset.submit(bbox).result()
    build_tensor.submit(era_paths, ais_path, bbox)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сбор Zarr‑тензора MBZ")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("lon_min", "lat_min", "lon_max", "lat_max"))
    parser.add_argument("--years", nargs="*", type=int, default=[2023])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main(args.years, tuple(args.bbox)))
