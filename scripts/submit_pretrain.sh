#!/bin/bash
# Halo8 NequIP pretraining — auto-resubmitting SLURM driver (Spec v5 §6).

#SBATCH --job-name=halo8_nequip
#SBATCH --partition=gpu1
#SBATCH --nodelist=n007
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=output/halo8_nequip_v1/logs/slurm/job-%j.out
#SBATCH --error=output/halo8_nequip_v1/logs/slurm/job-%j.err
#SBATCH --signal=B:USR1@600

set -euo pipefail

PROJECT_DIR="/gpfs/home1/yeseo1ee/projects/halo8-nequip-pretrain"
WORK_DIR="$PROJECT_DIR/output/halo8_nequip_v1"
CONFIG="$PROJECT_DIR/configs/halo8_nequip_v1.yaml"
DONE_FLAG="$WORK_DIR/training_complete.flag"
COUNT_FILE="$WORK_DIR/resubmit_count.txt"
MAX_RESUBMIT=50
RESUBMIT_DELAY_ON_CRASH=120

mkdir -p "$WORK_DIR/logs/slurm" "$WORK_DIR/checkpoints"
cd "$PROJECT_DIR"

[ -f "$COUNT_FILE" ] || echo "0" > "$COUNT_FILE"
COUNT=$(cat "$COUNT_FILE")
echo "[$(date)] === halo8_nequip job start (resubmit #$COUNT / $MAX_RESUBMIT) ==="

if [ -f "$DONE_FLAG" ]; then
    echo "[$(date)] DONE_FLAG present — training already complete. Exiting."
    exit 0
fi
if [ "$COUNT" -ge "$MAX_RESUBMIT" ]; then
    echo "[$(date)] MAX_RESUBMIT reached. Investigate manually."
    exit 1
fi

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
echo "[$(date)] python=$(which python)  nequip-train=$(which nequip-train)"

# TF32 — propagated to the launcher
export NEQUIP_ENABLE_TF32=1

TRAIN_PID=""
graceful_exit() {
    echo "[$(date)] === SIGUSR1 received — graceful shutdown ==="
    if [ -n "$TRAIN_PID" ]; then
        kill -SIGTERM "$TRAIN_PID" 2>/dev/null || true
        wc=0
        while kill -0 "$TRAIN_PID" 2>/dev/null; do
            sleep 5
            wc=$((wc + 1))
            if [ $wc -ge 96 ]; then
                echo "[$(date)] WARNING: nequip did not exit within 8min, force kill"
                kill -SIGKILL "$TRAIN_PID" 2>/dev/null || true
                break
            fi
        done
    fi
    if [ -f "$WORK_DIR/checkpoints/last.ckpt" ]; then
        AGE=$(($(date +%s) - $(stat -c %Y "$WORK_DIR/checkpoints/last.ckpt")))
        echo "[$(date)] last.ckpt age=${AGE}s"
    else
        echo "[$(date)] WARNING: no last.ckpt to resume from"
    fi
    NEW_COUNT=$((COUNT + 1))
    echo "$NEW_COUNT" > "$COUNT_FILE"
    echo "[$(date)] submitting next job (will be #$NEW_COUNT)"
    sbatch "$PROJECT_DIR/scripts/submit_pretrain.sh"
    exit 0
}
trap graceful_exit SIGUSR1

echo "[$(date)] preflight..."
python "$PROJECT_DIR/scripts/preflight.py" || {
    echo "[$(date)] preflight failed — NOT resubmitting"
    exit 2
}

echo "[$(date)] prepare_data (idempotent)..."
python "$PROJECT_DIR/scripts/prepare_data.py" || {
    echo "[$(date)] prepare_data failed — NOT resubmitting"
    exit 3
}

CHECKPOINT="$WORK_DIR/checkpoints/last.ckpt"
RESTART_FLAG=""
if [ -f "$CHECKPOINT" ]; then
    echo "[$(date)] resuming from $CHECKPOINT"
    RESTART_FLAG="--restart $CHECKPOINT"
else
    echo "[$(date)] starting fresh"
    # record git SHA at fresh start for provenance
    git -C "$PROJECT_DIR" rev-parse HEAD > "$WORK_DIR/code_sha.txt" 2>/dev/null \
        || echo "unknown" > "$WORK_DIR/code_sha.txt"
fi

nvidia-smi || true

echo "[$(date)] launching nequip_launcher.py..."
cd "$WORK_DIR"
# shellcheck disable=SC2086
python "$PROJECT_DIR/scripts/nequip_launcher.py" $RESTART_FLAG "$CONFIG" \
    > "$WORK_DIR/logs/training.log" 2>&1 &
TRAIN_PID=$!
echo "[$(date)] nequip launcher pid=$TRAIN_PID"

wait $TRAIN_PID
EXIT_CODE=$?
echo "[$(date)] nequip exited with code $EXIT_CODE"

if [ -f "$DONE_FLAG" ]; then
    echo "[$(date)] DONE_FLAG present — stopping chain"
    exit 0
fi
if grep -q "Training complete" "$WORK_DIR/logs/training.log" 2>/dev/null; then
    echo "[$(date)] training-complete marker found — stopping chain"
    touch "$DONE_FLAG"
    exit 0
fi

EPOCH=$(python -c "
import torch
try:
    t = torch.load('$WORK_DIR/checkpoints/last.ckpt', map_location='cpu')
    print(t.get('epoch', 0))
except Exception:
    print(0)
" 2>/dev/null || echo 0)
MAX_EPOCHS=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['max_epochs'])")
echo "[$(date)] epoch=$EPOCH / $MAX_EPOCHS"
if [ "$EPOCH" -ge "$MAX_EPOCHS" ]; then
    echo "[$(date)] reached max_epochs — marking complete"
    touch "$DONE_FLAG"
    exit 0
fi

if [ "$EXIT_CODE" -ne 0 ]; then
    echo "[$(date)] crash (exit=$EXIT_CODE) — sleeping ${RESUBMIT_DELAY_ON_CRASH}s before resubmit"
    sleep "$RESUBMIT_DELAY_ON_CRASH"
fi

NEW_COUNT=$((COUNT + 1))
echo "$NEW_COUNT" > "$COUNT_FILE"
echo "[$(date)] resubmitting (will be #$NEW_COUNT)"
sbatch "$PROJECT_DIR/scripts/submit_pretrain.sh"
