#!/usr/bin/env bash
# sweep_supcon_temp.sh
# ====================
# Sweep the SupCon temperature for one model and evaluate each run on test.
# Lower temperature = sharper/harder contrast (more separation, can over-tighten);
# higher = softer. We scan a sensible band around the 0.1 default.
#
# Usage:
#   ./sweep_supcon_temp.sh g2a 0          # model, CUDA device
#   ./sweep_supcon_temp.sh m2a 2
#
# Edit TEMPS / paths as needed. Checkpoints + per-run JSON land in sweep_supcon/.

set -euo pipefail

MODEL="${1:-g2a}"
GPU="${2:-0}"
DATA_DIRS="${DATA_DIRS:-dataset}"
CACHE="${CACHE:-cache_verification}"
SPLIT_FILE="${SPLIT_FILE:-split_ids.json}"
EPOCHS="${EPOCHS:-150}"
OUTDIR="sweep_supcon/${MODEL}"
mkdir -p "$OUTDIR"

TEMPS=(0.03 0.05 0.07 0.09 0.10 0.12 0.15)

for T in "${TEMPS[@]}"; do
  TAG="${MODEL}_t${T}"
  echo "=== train $TAG (temperature=$T) ==="
  CUDA_VISIBLE_DEVICES="$GPU" python train_verification.py \
    --model "$MODEL" --loss supcon --temperature "$T" \
    --data_dirs $DATA_DIRS --split_file "$SPLIT_FILE" --cache_dir "$CACHE" \
    --epochs "$EPOCHS" --out "$OUTDIR/${TAG}.pt" --device cuda

  echo "=== eval $TAG ==="
  CUDA_VISIBLE_DEVICES="$GPU" python eval_verification.py \
    --model "$MODEL" --checkpoint "$OUTDIR/${TAG}.best.pt" \
    --split test --data_dirs $DATA_DIRS --cache_dir "$CACHE" \
    --out_json "$OUTDIR/${TAG}_test.json"
done

echo "Done. Per-temperature results in $OUTDIR/*_test.json"
