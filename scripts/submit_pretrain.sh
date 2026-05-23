#!/bin/bash
# Halo8 NequIP 0.9.1 pre-training — multi-GPU DDP SLURM driver.
#
# Lightning auto-spawns one process per GPU via DDP; we request 4 GPUs on a
# single gpu1 (RTX 3090) node and a single SLURM task. Lightning handles
# inter-rank coordination via torch.distributed.

#SBATCH --job-name=halo8_nequip
# Try multiple GPU partitions — SLURM picks the one with earliest start.
# gpu4/gpu5 (A6000 48GB) and gpu3 (A6000ada 48GB) are preferred over gpu1
# (RTX 3090 24GB) for memory headroom; gpu6 (A10 24GB) is a fallback.
#SBATCH --partition=gpu1,gpu3,gpu4,gpu5,gpu6
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
# Exclude n021 — both 653760 and 653789 hung during startup (DDP init OK,
# then complete silence until watchdog SIGKILL at 16min). Same node twice
# suggests node-specific InfiniBand or NCCL state issue.
#SBATCH --exclude=n021
# 1 SLURM task per GPU — Lightning auto-detects SLURM_NTASKS as the
# DDP world size. Earlier `--ntasks=1` reduced the DDP world size to 1
# (job 649813 ran on a single GPU despite 4 being allocated).
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=output/halo8_nequip_v1/logs/slurm/job-%j.out
#SBATCH --error=output/halo8_nequip_v1/logs/slurm/job-%j.err
#SBATCH --signal=B:USR1@600

set -euo pipefail

PROJECT_DIR="/gpfs/home1/yeseo1ee/projects/halo8-nequip-pretrain"
WORK_DIR="$PROJECT_DIR/output/halo8_nequip_v1"
CONFIG_DIR="$PROJECT_DIR/configs"
CONFIG_NAME="halo8_nequip_v1.yaml"
DONE_FLAG="$WORK_DIR/training_complete.flag"
COUNT_FILE="$WORK_DIR/resubmit_count.txt"
MAX_RESUBMIT=50
RESUBMIT_DELAY_ON_CRASH=120

mkdir -p "$WORK_DIR/logs/slurm" "$WORK_DIR/training_run"
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

# Both 649909 and 650533 froze silently mid-training (NCCL allreduce deadlock,
# no Python-level error, no exit). Without a timeout NCCL waits forever and
# our auto-resubmit chain never fires. Force NCCL to error out when a
# collective stalls so SLURM sees a non-zero exit and resubmits.
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600   # 10 min before collective is killed
export NCCL_TIMEOUT=600
# Verbose NCCL logging so we can identify which collective hung.
export NCCL_DEBUG=WARN
echo "[$(date)] NCCL timeout/heartbeat env exported"

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
                kill -SIGKILL "$TRAIN_PID" 2>/dev/null || true
                break
            fi
        done
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

# Lightning's ModelCheckpoint(save_last=True) writes <dirpath>/last.ckpt.
# NequIP resumes by passing the path via the hydra `ckpt_path` override.
LAST_CKPT="$WORK_DIR/training_run/last.ckpt"
HYDRA_CKPT_ARG=""
if [ -f "$LAST_CKPT" ]; then
    echo "[$(date)] resuming from $LAST_CKPT"
    HYDRA_CKPT_ARG="++ckpt_path=$LAST_CKPT"
else
    echo "[$(date)] starting fresh"
    git -C "$PROJECT_DIR" rev-parse HEAD > "$WORK_DIR/code_sha.txt" 2>/dev/null \
        || echo "unknown" > "$WORK_DIR/code_sha.txt"
fi

# Preserve previous attempt's stdout/stderr so a failing restart doesn't
# overwrite the original first-attempt traceback.
if [ -f "$WORK_DIR/logs/training.log" ]; then
    cp "$WORK_DIR/logs/training.log" "$WORK_DIR/logs/training.prev.log" 2>/dev/null || true
fi

# Hydra's @main has the default config_path set to os.getcwd(). We can either
# cd into $CONFIG_DIR or pass --config-path explicitly. Use the explicit
# form so cwd remains at $PROJECT_DIR (paths in the config are absolute
# anyway, but this keeps SLURM working-dir semantics predictable).
HYDRA_RUN_DIR="$WORK_DIR/training_run/hydra/${SLURM_JOB_ID:-local}"
mkdir -p "$HYDRA_RUN_DIR"

nvidia-smi || true

echo "[$(date)] launching nequip-train via srun (4-GPU DDP, ntasks-per-node=$SLURM_NTASKS_PER_NODE)..."
# srun spawns SLURM_NTASKS_PER_NODE processes; Lightning detects this and
# uses them as DDP ranks. Without srun, Lightning + ntasks=1 collapses to
# rank 0 only — see comments in the SBATCH header.
# shellcheck disable=SC2086
srun --ntasks-per-node=$SLURM_NTASKS_PER_NODE --cpus-per-task=$SLURM_CPUS_PER_TASK \
    nequip-train \
        --config-path "$CONFIG_DIR" \
        --config-name "$CONFIG_NAME" \
        hydra.run.dir="$HYDRA_RUN_DIR" \
        $HYDRA_CKPT_ARG \
    > "$WORK_DIR/logs/training.log" 2>&1 &
TRAIN_PID=$!
echo "[$(date)] nequip-train pid=$TRAIN_PID"

# In-job hang watchdog: silent NCCL deadlocks have repeatedly hung the
# chain for hours (NCCL_TIMEOUT/TORCH_NCCL_* did not fire, neither did
# SimpleDDPStrategy's manual all_reduce). Detect by training.log mtime
# staleness and force-kill srun + ranks; the main `wait` then returns
# non-zero and the auto-resubmit block at the end of this script fires.
STALE_THRESHOLD=900   # 15 min without log update => hang
(
    sleep 600  # grace period for setup / dataset stats
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        sleep 60
        if [ ! -f "$WORK_DIR/logs/training.log" ]; then continue; fi
        AGE=$(( $(date +%s) - $(stat -c %Y "$WORK_DIR/logs/training.log" 2>/dev/null || date +%s) ))
        if [ "$AGE" -gt "$STALE_THRESHOLD" ]; then
            echo "[$(date)] HANG WATCHDOG: training.log stale for ${AGE}s — force-killing srun (PID $TRAIN_PID)"
            kill -SIGTERM "$TRAIN_PID" 2>/dev/null || true
            sleep 30
            kill -SIGKILL "$TRAIN_PID" 2>/dev/null || true
            pkill -9 -u "$USER" -f "nequip-train" 2>/dev/null || true
            break
        fi
    done
) &
WATCHDOG_PID=$!

EXIT_CODE=0
wait "$TRAIN_PID" || EXIT_CODE=$?
echo "[$(date)] nequip-train exited with code $EXIT_CODE"

# Stop watchdog regardless of exit path
kill "$WATCHDOG_PID" 2>/dev/null || true

if [ -f "$DONE_FLAG" ]; then
    echo "[$(date)] DONE_FLAG present — stopping chain"
    exit 0
fi
if grep -q "Trainer.fit stopped" "$WORK_DIR/logs/training.log" 2>/dev/null; then
    echo "[$(date)] Lightning early-stop marker detected — marking complete"
    touch "$DONE_FLAG"
    exit 0
fi

# Try to read the last completed epoch from the most recent metrics csv.
EPOCH=$(find "$WORK_DIR/training_run/lightning_logs" -name "metrics.csv" -print 2>/dev/null \
    | xargs -I {} bash -c 'tail -1 "$1" 2>/dev/null' _ {} \
    | awk -F',' 'NR==1 {print $1}' 2>/dev/null)
EPOCH=${EPOCH:-0}
MAX_EPOCHS=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG_DIR/$CONFIG_NAME'));print(c['trainer']['max_epochs'])")
echo "[$(date)] last_epoch=$EPOCH / $MAX_EPOCHS"
if [ -n "$EPOCH" ] && [ "$EPOCH" -ge "$MAX_EPOCHS" ] 2>/dev/null; then
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
