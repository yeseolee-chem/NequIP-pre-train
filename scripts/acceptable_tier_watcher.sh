#!/bin/bash
# Polls metrics_epoch.csv every 60s. When validation reaches the 합격
# (acceptable) tier — energy MAE < 15 meV/atom AND force MAE < 0.08 eV/Å —
# touches training_complete.flag, cancels any halo8_nequip jobs in queue,
# and copies best_model.pth to backbone_acceptable_epoch_<N>.pth as the
# final preserved backbone for downstream fine-tuning.
set -u

WORK=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/NequIP/output/halo8_nequip_v1
EP_CSV=$WORK/training_run/metrics_epoch.csv
DONE_FLAG=$WORK/training_complete.flag
THRESH_F=0.08
THRESH_E=0.015

echo "[$(date)] [acceptable_watcher] armed — threshold f_mae<$THRESH_F AND e/N_mae<$THRESH_E"

while true; do
    sleep 60
    if [ -f "$DONE_FLAG" ]; then
        echo "[$(date)] [acceptable_watcher] DONE_FLAG already present — exiting"
        break
    fi
    if [ ! -f "$EP_CSV" ]; then continue; fi
    # Latest data row (skip header)
    last_row=$(grep -v "^epoch" "$EP_CSV" 2>/dev/null | tail -1)
    [ -z "$last_row" ] && continue

    # CSV columns: 1=epoch, 14=validation_f_mae, 16=validation_e/N_mae
    epoch=$(echo "$last_row" | awk -F',' '{print $1}' | tr -d ' ')
    f_mae=$(echo "$last_row" | awk -F',' '{print $14}' | tr -d ' ')
    e_mae=$(echo "$last_row" | awk -F',' '{print $16}' | tr -d ' ')

    [ -z "$epoch" ] && continue
    [ -z "$f_mae" ] && continue

    # Only log when epoch changes
    if [ "${PREV_EPOCH:-}" != "$epoch" ]; then
        echo "[$(date)] [acceptable_watcher] epoch=$epoch val_f_mae=$f_mae val_e/N_mae=$e_mae"
        PREV_EPOCH=$epoch
    fi

    # Float compare via awk
    ok=$(awk -v f="$f_mae" -v e="$e_mae" -v tf="$THRESH_F" -v te="$THRESH_E" \
        'BEGIN { print (f<tf && e<te) ? 1 : 0 }')
    if [ "$ok" = "1" ]; then
        echo "[$(date)] [acceptable_watcher] 합격 REACHED at epoch $epoch (f=$f_mae, e=$e_mae)"
        echo "  → touching DONE_FLAG, cancelling jobs, archiving backbone"

        touch "$DONE_FLAG"

        # Save backbone snapshot
        BACKBONE=$WORK/backbone_acceptable_epoch_${epoch}.pth
        if [ -f "$WORK/training_run/best_model.pth" ]; then
            cp "$WORK/training_run/best_model.pth" "$BACKBONE"
            echo "  → backbone saved: $BACKBONE ($(du -h $BACKBONE | awk '{print $1}'))"
        fi

        # Cancel the running job(s)
        QUEUED=$(squeue -u "$USER" -n halo8_nequip -h -o "%i" | tr '\n' ' ')
        if [ -n "$QUEUED" ]; then
            echo "  → scancel $QUEUED"
            scancel $QUEUED
        fi
        break
    fi
done

echo "[$(date)] [acceptable_watcher] exit"
