# Halo8 NequIP Pre-training (Spec v5)

Pre-training pipeline for a NequIP MLIP backbone on the Halo8 reaction
trajectory dataset. Implements the v5 operational spec in
[`docs/Halo8_NequIP_Pretrain_Spec_v5.md`](docs/Halo8_NequIP_Pretrain_Spec_v5.md).

Key v5 features:
- Schema probe with public ASE API (`row.key_value_pairs.keys()`).
- **`np.linspace` endpoint-preserving** validation-frame sampling — covers
  reactant, TS, AND product regions.
- Streaming extxyz write (peak RAM < 1 GB regardless of dataset size).
- `n_train` / `n_val` injection into the NequIP config via ruamel.yaml
  (comments preserved).
- Python launcher (`scripts/nequip_launcher.py`) that flips TF32 ON
  BEFORE importing nequip.
- 48-hour-per-job SLURM driver with SIGUSR1 graceful resubmit, every-epoch
  checkpoint.
- Tiered tier check (`scripts/check_target_metrics.py`) in meV/atom.

## Layout

```
halo8-nequip-pretrain/
├── configs/
│   └── halo8_nequip_v1.yaml      # NequIP config (Spec §4)
├── data/                          # F0-passed JSON (when provided)
├── scripts/
│   ├── preflight.py
│   ├── prepare_data.py            # v5 schema probe + linspace + stream + ruamel
│   ├── nequip_launcher.py         # TF32-before-import wrapper
│   ├── submit_pretrain.sh         # 48h SLURM, SIGUSR1 auto-resubmit
│   ├── check_target_metrics.py    # 우수/합격/실패 tier classification
│   ├── validate_output.py         # post-training manifest + asserts
│   └── daily_check.py             # cron monitoring
├── output/
│   └── halo8_nequip_v1/           # working dir, gitignored
└── docs/
    ├── Halo8_NequIP_Pretrain_Spec_v5.md
    └── SERVER_ACCESS_GUIDE.md
```

## Quick start

```bash
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot              # nequip 0.6.2 + torch 2.2.1 + ruamel.yaml 0.19

python scripts/prepare_data.py      # ≈30–60 min (one-time)
python scripts/preflight.py
sbatch scripts/submit_pretrain.sh   # chains via MAX_RESUBMIT=50
```

## Monitoring

```bash
squeue -u $USER -n halo8_nequip
tail -f output/halo8_nequip_v1/logs/training.log
cat output/halo8_nequip_v1/resubmit_count.txt
ls output/halo8_nequip_v1/training_complete.flag 2>/dev/null && echo COMPLETE
```

Full guide: [`docs/SERVER_ACCESS_GUIDE.md`](docs/SERVER_ACCESS_GUIDE.md).

## Local notes

- Halo8 shards under `/gpfs/home1/yeseo1ee/projects/ts_prediction_project/data/`
  use the `row.data["dand_id"]` encoding for reaction_id + frame_idx;
  `prepare_data.py` autodetects.
- F0-passed list not present yet → fallback to a deterministic 17,574-reaction
  subset from `eda-asm-prediction/data/halo8_index/index.parquet`
  (seed=42, `interior_ts & !short_traj & n_components_R==1`). Drop the
  real list at `data/halo8_F0_passed.json` and rerun with `--force` to
  switch.
- HPC walltime cap is 48 h per submission → relies on auto-resubmit.
