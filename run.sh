#!/usr/bin/env bash
# run.sh – end‑to‑end сценарий: сбор → обучение → тест → отчёт
#
# usage: bash run.sh [lon_min lat_min lon_max lat_max]
# если bbox не указан, используется дефолт из config.yaml (Балтика)

set -e

# --- директории ---
RAW=data/raw
PROC=data/processed
MODELS=models
REPORTS=reports
mkdir -p "$RAW" "$PROC" "$MODELS" "$REPORTS"

# --- BBOX из аргументов или по умолчанию ---
if [ "$#" -eq 4 ]; then
  BBOX=("$1" "$2" "$3" "$4")
else
  BBOX=(9 53 32 66)
fi

# --- 1. Сбор датасета -------------------------------------------------------
python dataset_pipeline.py --bbox "${BBOX[*]}" --years 2023

# --- 2. Обучение ------------------------------------------------------------
python train.py --model mlp \
                --tensor $PROC/mbz_tensor.zarr \
                --out $MODELS/mlp_v1.pt

python train.py --model unet \
                --tensor $PROC/mbz_tensor.zarr \
                --out $MODELS/unet_v2.pt \
                --batch 4

# --- 3. Оценка --------------------------------------------------------------
python evaluate.py --tensor $PROC/mbz_tensor.zarr \
                   --mlp $MODELS/mlp_v1.pt \
                   --unet $MODELS/unet_v2.pt \
                   --out $REPORTS/eval_report.pdf

echo "\nВся цепочка завершена. Итоговый отчёт: $REPORTS/eval_report.pdf"
