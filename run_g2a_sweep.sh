#!/usr/bin/env bash
# ============================================================================
# run_g2a_sweep.sh — G2A: MULTI-SEED, train-per-protocol × temperature sweep
# ============================================================================
# For every (seed × temperature × enroll/verify protocol) cell we:
#   1. TRAIN a fresh deterministic G2A (SupCon) at that temperature/seed. The
#      VALIDATION set is partitioned with THAT protocol; the lowest-val-EER
#      checkpoint is kept (.best.pt).
#   2. EVALUATE that checkpoint on TEST under the SAME protocol.
# Multiple seeds let you report mean ± std (see aggregate_g2a_sweep.py), which is
# what a top-tier venue expects given residual GPU/GRU nondeterminism.
#
#   cells = #SEEDS × #TEMPS × #PROTOCOLS
#   default = 10 × 7 × 6 = 420 trainings + 420 evals.
#
# COST: ~30 min/cell single-GPU → 420 cells ≈ 8–9 GPU-days. Use GPUS="0 1 2 3"
# to shard across GPUs (wall-clock ÷ #GPUs), and/or shrink SEEDS/TEMPS/PROTOS.
# RESUMABLE: a finished cell (its result JSON exists) is skipped unless FORCE=1.
#
# Examples:
#   GPUS="0 1 2 3" nohup ./run_g2a_sweep.sh > logs/master.out 2>&1 &   # 4-GPU
#   SEEDS="42 123 7" TEMPS="0.09" PROTOS="e70v30 e50v50" ./run_g2a_sweep.sh
#
# Overrides (env): SEEDS TEMPS PROTOS GPUS EPOCHS DEVICE DATA_DIR SPLIT_FILE
#   CACHE_DIR CKPT_DIR RESULTS_DIR LOG_DIR SELECT_BY TRAIN_ARGS EVAL_ARGS FORCE
# ============================================================================
set -uo pipefail
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"   # determinism
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

# ---- config ----------------------------------------------------------------
SEEDS=${SEEDS:-"42 123 7 2024 99 5 31 88 17 256"}   # 10 seeds
TEMPS=${TEMPS:-"0.03 0.05 0.07 0.09 0.10 0.12 0.15"}
PROTOS=${PROTOS:-"e50v50 e70v30 e80v20 v50e50 v30e70 v20e80"}
GPUS=${GPUS:-"0"}                      # space-separated GPU ids to shard across
EPOCHS=${EPOCHS:-150}
DEVICE=${DEVICE:-cuda}
DATA_DIR=${DATA_DIR:-"/media/sharma/CE9C1E919C1E7465/APW_data/baseline models/dataset"}
SPLIT_FILE=${SPLIT_FILE:-split_ids.json}
CACHE_DIR=${CACHE_DIR:-./cache_verification}
CKPT_DIR=${CKPT_DIR:-./checkpoints_verification}
RESULTS_DIR=${RESULTS_DIR:-./results}
LOG_DIR=${LOG_DIR:-./logs}
SELECT_BY=${SELECT_BY:-eer}
TRAIN_ARGS=${TRAIN_ARGS:-""}
EVAL_ARGS=${EVAL_ARGS:-""}
FORCE=${FORCE:-0}
PY=${PY:-python}

mkdir -p "$CKPT_DIR" "$RESULTS_DIR" "$LOG_DIR"
MASTER_LOG="$LOG_DIR/sweep_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }

# label -> "enroll_ratio [--verify_first]"
proto_params() {
  case "$1" in
    e50v50) echo "0.5 " ;;   e70v30) echo "0.7 " ;;   e80v20) echo "0.8 " ;;
    v50e50) echo "0.5 --verify_first" ;;
    v30e70) echo "0.7 --verify_first" ;;
    v20e80) echo "0.8 --verify_first" ;;
    *) echo "" ;;
  esac
}

# One cell: deterministic train (if needed) + eval, pinned to one GPU.
run_cell() {
  local SEED=$1 T=$2 LBL=$3 GPU=$4
  local ER VF; read -r ER VF <<< "$(proto_params "$LBL")"
  [[ -z "$ER" ]] && { log "[gpu$GPU] unknown protocol '$LBL', skip"; return 1; }
  local TAG="g2a_t${T}_${LBL}_s${SEED}"
  local CKPT="$CKPT_DIR/${TAG}.pt" BEST="$CKPT_DIR/${TAG}.best.pt"
  local OUT_JSON="$RESULTS_DIR/${TAG}.json"
  local TLOG="$LOG_DIR/train_${TAG}.log" ELOG="$LOG_DIR/eval_${TAG}.log"

  if [[ ! -f "$BEST" || "$FORCE" == "1" ]]; then
    log "[gpu$GPU][train] $TAG (T=$T $LBL seed=$SEED)"
    CUDA_VISIBLE_DEVICES="$GPU" $PY train_verification.py \
        --model g2a --loss supcon --temperature "$T" --seed "$SEED" \
        --data_dirs "$DATA_DIR" --split_file "$SPLIT_FILE" --cache_dir "$CACHE_DIR" \
        --out "$CKPT" --epochs "$EPOCHS" --device "$DEVICE" \
        --enroll_ratio "$ER" $VF --select_by "$SELECT_BY" \
        $TRAIN_ARGS > "$TLOG" 2>&1 \
      || { log "[gpu$GPU][train] $TAG FAILED (see $TLOG)"; return 1; }
  fi

  local ECK="$BEST"; [[ -f "$ECK" ]] || ECK="$CKPT"
  [[ -f "$ECK" ]] || { log "[gpu$GPU] $TAG: no checkpoint, skip eval"; return 1; }

  CUDA_VISIBLE_DEVICES="$GPU" $PY eval_verification.py \
      --model g2a --checkpoint "$ECK" --split test --seed "$SEED" \
      --data_dirs "$DATA_DIR" --split_file "$SPLIT_FILE" --cache_dir "$CACHE_DIR" \
      --device "$DEVICE" --enroll_ratio "$ER" $VF --out_json "$OUT_JSON" \
      $EVAL_ARGS > "$ELOG" 2>&1 \
    && { local E; E=$(grep -m1 "EER  " "$ELOG" | tr -s ' '); log "[gpu$GPU][eval] $TAG done ${E}"; } \
    || log "[gpu$GPU][eval] $TAG FAILED (see $ELOG)"
}

gpus=($GPUS); ngpu=${#gpus[@]}
n_seed=$(echo $SEEDS|wc -w); n_temp=$(echo $TEMPS|wc -w); n_proto=$(echo $PROTOS|wc -w)
total_cells=$((n_seed*n_temp*n_proto))

log "======== G2A multi-seed × train-per-protocol × temperature ========"
log "SEEDS  : $SEEDS  (${n_seed})"
log "TEMPS  : $TEMPS  (${n_temp})"
log "PROTOS : $PROTOS  (${n_proto})"
log "cells  : $total_cells   GPUs: ${gpus[*]} (${ngpu} parallel)   EPOCHS=$EPOCHS"
log "determinism: CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG  select=$SELECT_BY"
log "est. wall-clock ≈ $(( total_cells * 30 / (ngpu*60) )) h  (@~30 min/cell)"
log "FORCE=$FORCE   master log: $MASTER_LOG"
log "==================================================================="

# Build the cache ONCE (single process) so parallel workers never race on it.
log "Warming cache (train/val/test, single process)…"
$PY - "$CACHE_DIR" "$SPLIT_FILE" "$DATA_DIR" <<'PYEOF' >> "$MASTER_LOG" 2>&1 || log "cache warmup warning (continuing)"
import sys
from pathlib import Path
from verification import VerificationData
d = VerificationData(sys.argv[1], sys.argv[2], [Path(sys.argv[3])])
for k in ("train", "val", "test"):
    d.prepare_cache(d.split[k])
PYEOF

# Enumerate PENDING cells (skip finished ones up front so no GPU slot is wasted).
cells=(); n_skipped=0
for SEED in $SEEDS; do for T in $TEMPS; do for LBL in $PROTOS; do
  OUT_JSON="$RESULTS_DIR/g2a_t${T}_${LBL}_s${SEED}.json"
  if [[ -f "$OUT_JSON" && "$FORCE" != "1" ]]; then n_skipped=$((n_skipped+1)); continue; fi
  cells+=("$SEED|$T|$LBL")
done; done; done
log "Pending cells: ${#cells[@]}  (skipping $n_skipped already done)"

start=$(date +%s); idx=0; total=${#cells[@]}
while (( idx < total )); do                       # dispatch ngpu at a time
  for (( g=0; g<ngpu && idx<total; g++ )); do
    IFS='|' read -r SEED T LBL <<< "${cells[$idx]}"
    run_cell "$SEED" "$T" "$LBL" "${gpus[$g]}" &
    idx=$((idx+1))
  done
  wait
  log "progress: $idx/$total cells dispatched"
done

# ---- aggregate (mean ± std + significance tests) ---------------------------
log "Aggregating -> $RESULTS_DIR/summary.csv (+ stats report)"
$PY aggregate_g2a_sweep.py --results_dir "$RESULTS_DIR" \
    --out "$RESULTS_DIR/summary.csv" 2>&1 | tee -a "$MASTER_LOG"

dur=$(( $(date +%s) - start ))
log "==================================================================="
log "DONE in $((dur/3600))h $(((dur%3600)/60))m.  ran $total cells (skipped $n_skipped)."
log "summary: $RESULTS_DIR/summary.csv   per-cell stats: $RESULTS_DIR/summary_by_cell.csv"
log "==================================================================="
