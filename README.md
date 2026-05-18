# Halo8 NequIP Pre-training

Pre-training pipeline for a NequIP MLIP backbone on the Halo8 reaction
trajectory dataset (≈17k reactions, millions of frames). Built per the
operational spec in [`docs/Halo8_NequIP_Pretrain_Spec.md`](docs/Halo8_NequIP_Pretrain_Spec.md).

Auto-resubmitting SLURM driver, every-epoch checkpointing, idempotent data
prep, and graceful SIGUSR1 shutdown so no more than 1 epoch is ever lost.

## Layout

```
halo8-nequip-pretrain/
├── configs/
│   └── halo8_nequip_v1.yaml      # NequIP config (Spec §4)
├── data/                          # F0-passed / V2-holdout JSONs (gitignored if generated)
├── scripts/
│   ├── preflight.py               # hard env + data checks (Spec §1)
│   ├── prepare_data.py            # reaction selection + extxyz writer (Spec §3)
│   └── submit_pretrain.sh         # SLURM driver with auto-resubmit (Spec §6)
├── output/
│   └── halo8_nequip_v1/           # working dir, gitignored
│       ├── checkpoints/           # best/last/epoch_NNN
│       ├── data/                  # halo8_{train,val}.extxyz, hashes, split JSONs
│       └── logs/                  # training.log + slurm/job-*.out
└── docs/
    └── Halo8_NequIP_Pretrain_Spec.md
```

## Quick start

```bash
# 1. Activate env (nequip 0.6.2 + torch 2.2.1+cu118 already installed in reactot)
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

# 2. Verify preflight
python scripts/preflight.py

# 3. Submit (auto-resubmits up to MAX_RESUBMIT=50)
sbatch scripts/submit_pretrain.sh
```

## SLURM settings (defaults)

| Setting | Value |
|---|---|
| `--partition` | `gpu1` (RTX3090) |
| `--nodelist` | `n007` (or any idle gpu1 node) |
| `--gres` | `gpu:rtx3090:1` |
| `--cpus-per-task` | 8 |
| `--mem` | 64G |
| `--time` | 48:00:00 (max per single submission) |
| `--signal` | `B:USR1@600` (T-10 min grace) |

## Reaction selection

1. If `data/halo8_F0_passed.json` exists, the file's reaction IDs are used.
2. Otherwise the script reads `data/halo8_index/index.parquet` from the
   sibling `eda-asm-prediction` project and filters reactions by
   `interior_ts & !short_traj & n_components_R == 1`, then **deterministically
   caps to 17,574 reactions (seed=42)** to match the spec's target count.
3. If `data/v2_holdout_ids.json` exists, those reactions are removed.

The exact set used is written to `output/halo8_nequip_v1/data/used_reaction_ids.json`
with provenance metadata.

## Monitoring

```bash
squeue -u $USER -n halo8_nequip
tail -f output/halo8_nequip_v1/logs/training.log
cat output/halo8_nequip_v1/resubmit_count.txt
ls output/halo8_nequip_v1/training_complete.flag 2>/dev/null && echo "COMPLETE"
```

See [`docs/SERVER_ACCESS_GUIDE.md`](docs/SERVER_ACCESS_GUIDE.md) for full
shell-level monitoring commands.

## Stopping criteria

Training halts (and `training_complete.flag` is created) when ANY of:
- epoch ≥ 200 (`config.max_epochs`)
- LR ≤ 1e-6 (`early_stopping_lower_bounds`)
- 100 epochs without val_loss improvement
- target metrics met (val energy MAE < 1 kcal/mol AND val force MAE < 0.05 eV/Å)
- `touch training_complete.flag` (manual)

To halt the auto-resubmit chain at any time:

```bash
touch output/halo8_nequip_v1/training_complete.flag
scancel -u $USER -n halo8_nequip
```
