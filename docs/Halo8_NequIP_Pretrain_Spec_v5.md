# Halo8 NequIP Pre-training — Operational Specification v5 (FINAL)

> **Status**: Production-ready after 4 iterative review cycles (v1 → v5).
> **Hard rule 1**: Training must never lose more than 1 epoch of progress to any interruption.
> **Hard rule 2**: Jobs must auto-resubmit on timeout/crash until training reaches stop criterion or `MAX_RESUBMIT` is hit.
> **Hard rule 3**: All hyperparameter, data, and code versions must be reproducibly logged.
> **Hard rule 4**: Data preparation must NOT load full dataset into RAM. Streaming write only.
> **Hard rule 5**: Validation set must uniformly cover each reaction's full trajectory, especially TS regions. Reaction-uniform stride sampling, never head sampling.
> **Hard rule 6** (new in v5): Validation sampling must include both first AND last frames of each trajectory (endpoint-preserving via `np.linspace`).

## Changelog v4 → v5 (polish-only; no critical issues remaining)

| Item | v4 | **v5** | Severity | Source |
|---|---|---|---|---|
| Val stride sampling | `frame_list[::stride][:N]` → truncates last 6% of trajectory | **`np.linspace(0, total-1, n_keep)` → endpoint-preserving** | 🟡 Medium — paper reviewer Q | review §2.1 |
| DDP module path | hardcoded `nequip.scripts.train_dist` import | **deferred discovery; run `find` after ddp branch install** | 🟡 Medium — DDP only | review §2.2 |
| Schema probe API | `row._keys` (private) | **`row.key_value_pairs.keys()` (public)** | 🟢 Low — forward-compat | review §2.3 |
| Schema hard-fail | only `energy`/`forces` checked | **`frame_idx` also hard-fails** | 🟢 Low — defensive | review §2.4 |
| `toatoms()` calls | called twice per row in fallback path | **merged into `extract_data()` single call** | 🟢 Low — ~5h saved (fallback only) | review §2.5 |

## Project evolution summary (v1 → v5)

| Version | Critical issue resolved | Reviewer cycle finding |
|---|---|---|
| v1 | (initial) | `n_train`/`n_val` missing → would not start |
| v2 | Auto-injection of n_train/n_val | RAM OOM → 264 GB required (would not finish data prep) |
| v3 | Streaming write fix | Val sampling biased toward reactant region → scientific validity |
| v4 | Reaction-uniform stride sampling | Polish only — endpoint truncation, private API usage |
| **v5** | **Endpoint-preserving linspace + final polish** | **No critical issues remain** |

Pattern: *bug → design → science → polish*. v5 is production-ready.

---

## Local deviation note

This repo implements v5 with one adaptation: the on-disk Halo8 shards
(`/gpfs/home1/yeseo1ee/projects/ts_prediction_project/data/Halo_{1..10}.db`)
store the reaction-trajectory key as `row.data["dand_id"]`
(e.g. `T1x_C2H2N2O_rxn00001_99`) instead of top-level `row.reaction_id` /
`row.frame_idx` attributes. `scripts/prepare_data.py` probes the schema at
startup and routes to either path. The hard-fail in §1 ("schema:
rxn/frame path") triggers only if **neither** is available.

The 17,574-reaction `F0_passed` list is also not yet on disk; until it is
provided, `prepare_data.py` builds a reproducible (seed=42) 17,574-reaction
subset from `eda-asm-prediction/data/halo8_index/index.parquet` after
filtering `interior_ts & !short_traj & n_components_R == 1`. Provenance
is recorded in `output/halo8_nequip_v1/data/used_reaction_ids.json`.

Drop the real list at `data/halo8_F0_passed.json` and re-run
`python scripts/prepare_data.py --force` to switch over.

---

## 0. Inputs & Outputs

### Inputs
- **Halo8 ASE database** (Zenodo DOI: `10.5281/zenodo.16737590`)
- **F0-passed reaction list**: `data/halo8_F0_passed.json` — exactly 17,574 reaction IDs
- **V2 holdout**: not needed (≥50 heavy atoms vs Halo8's 3–8 → disjoint)

### Outputs
```
output/halo8_nequip_v1/
├── best_model.pth
├── last_model.pth
├── trainer.pth
├── config_used.yaml
├── training_log.csv
├── manifest.json
├── training_complete.flag
├── checkpoints/{last,best,epoch_NNN}.ckpt
├── logs/{training.log, slurm/, tensorboard/}
├── data/
│   ├── halo8_train.extxyz       # ~16.7M frames
│   ├── halo8_val.extxyz         # 50K frames (linspace-sampled, v5)
│   ├── dataset_hash.txt
│   ├── .prepare_done
│   ├── halo8_schema_probe.txt
│   ├── val_frame_distribution.txt
│   └── used_reaction_ids.json
├── code_sha.txt
└── resubmit_count.txt
```

---

## 1. Pre-flight Checks

| Check | Threshold | Failure action |
|---|---|---|
| GPU available | ≥ 1 GPU with ≥ 24 GB memory | Abort |
| CUDA version | ≥ 11.8 | Abort |
| PyTorch | ≥ 2.0 | Abort |
| NequIP version | ≥ 0.6.2 | Abort |
| For multi-GPU: NequIP `ddp` branch (v5: discovery-based) | discovered module path verified | Abort with §7 instructions |
| ruamel.yaml | importable | `pip install ruamel.yaml` |
| Disk space | ≥ 200 GB free | Abort |
| Halo8 ASE DB | file present | Abort |
| F0-passed list | exactly 17,574 entries | Abort if mismatch |
| **Halo8 row schema** | `energy`, `forces`, **`frame_idx`** (v5: now required) | Abort with schema info |
| RAM available | ≥ 8 GB free | Warn |
| WandB API key | env var set | Set `wandb: false` if missing |
| SLURM | `sbatch` available | Switch to local mode if absent |

---

## 2. Environment Setup

```text
torch>=2.0.0,<2.5.0
nequip>=0.6.2,<0.7.0
e3nn>=0.5.0
ase>=3.22.0
pyyaml>=6.0
ruamel.yaml>=0.17.0
wandb>=0.15.0
h5py>=3.8.0
numpy>=1.24.0,<2.0.0
```

```bash
conda create -n nequip-halo8 python=3.10 -y
conda activate nequip-halo8
pip install -r requirements.txt
pip install 'nequip[wandb]>=0.6.2' ruamel.yaml
```

*In this repo we reuse the existing `reactot` env on `gate1.hpc`, which already has the same versions installed.*

---

## 3. Data Preparation (v5: 4 polish fixes)

See [`scripts/prepare_data.py`](../scripts/prepare_data.py) for the
implementation. The v5 steps are:

1. **Schema probe** (`halo8_schema_probe.txt`) — record where energy /
   forces / reaction_id / frame_idx live; pick access paths.
2. **F0-passed reactions** — exactly 17,574, hard-asserted (or
   deterministic 17,574 fallback from index.parquet, recorded as provenance).
3. **Reaction-level 95/5 split** (seed = 42).
4. **First sweep** — collect val-reaction `frame_idx` lists.
5. **`np.linspace` stride** per reaction — endpoint-preserving, no
   trailing truncation. `val_frame_distribution.txt` records p1/p25/p50/p75/p99
   so the user can verify `p99 ≈ pool_max`.
6. **Second sweep, streaming write** — `CHUNK = 10_000` frames flushed via
   `ase.io.write(..., append=True)`. Peak RAM < 1 GB.
7. **Inject `n_train` / `n_val`** into `configs/halo8_nequip_v1.yaml`
   via ruamel.yaml (preserves comments).
8. **SHA-256** hashes of `halo8_train.extxyz` and `halo8_val.extxyz`.
9. **`.prepare_done` sentinel** with `spec_version: v5`, sampling provenance.

The script is idempotent — re-running with `.prepare_done` present is a no-op.
Use `--force` to overwrite.

---

## 4. NequIP Configuration

See [`configs/halo8_nequip_v1.yaml`](../configs/halo8_nequip_v1.yaml).

Key v5 settings:

- `default_dtype: float32`
- `invariant_layers: 3` (Batzner 2022 Methods)
- `optimizer_amsgrad: true`
- `metrics_components` block (`forces_{mae,rmse}` + `total_energy_{mae,rmse}` per-atom)
- `batch_size: 5`, `validation_batch_size: 16`
- `root: output/` (FLAT — NequIP writes into `output/halo8_nequip_v1/`)
- `n_train` / `n_val` placeholders overwritten by `prepare_data.py`.

After first epoch, verify metric keys:

```bash
python -c "
import torch
t = torch.load('output/halo8_nequip_v1/checkpoints/last.ckpt', map_location='cpu')
print(sorted(t.get('best_val_metrics', {}).keys()))
"
```

Expected: `forces_mae`, `forces_rmse`, `total_energy_mae_per_atom`,
`total_energy_rmse_per_atom`.

---

## 5. Checkpointing

Same as v4. Per-epoch save is the **actual** protection mechanism.

> **Clarification**: NequIP v0.6.2 does NOT install a SIGTERM handler. The
> shell trap catches SIGUSR1 (sent at T-10min by SLURM) and sends SIGTERM
> to NequIP, but NequIP exits immediately. Real protection:
> `save_checkpoint_freq: 1` means every *completed* epoch is already
> saved to `last.ckpt`. Mid-epoch SIGTERM forfeits only the current
> incomplete epoch.

---

## 6. Auto-Resubmit SLURM

See [`scripts/submit_pretrain.sh`](../scripts/submit_pretrain.sh) +
[`scripts/nequip_launcher.py`](../scripts/nequip_launcher.py).

Local cluster constraints (UBAI `gate1.hpc`):
- Maximum time per job: **48 hours** (set via `--time=48:00:00`).
- Target node: `n007` (gpu1 partition, RTX 3090). Drop the `--nodelist`
  line to allow any idle gpu1 node.
- TF32 enabled via `NEQUIP_ENABLE_TF32=1` → `nequip_launcher.py`
  flips `torch.backends.cuda.matmul.allow_tf32` BEFORE any nequip import.

---

## 7. Multi-GPU (DDP)

Not active in this run. If used later:

```bash
conda create -n nequip-halo8-ddp python=3.10 -y
conda activate nequip-halo8-ddp
pip install -r requirements.txt ruamel.yaml
git clone -b ddp https://github.com/mir-group/nequip.git nequip-ddp
cd nequip-ddp && pip install -e . && cd ..
git clone -b state-reduce https://github.com/mir-group/pytorch_runstats.git
cd pytorch_runstats && pip install -e . && cd ..
```

Then run discovery:

```bash
cd nequip-ddp
grep -rn "DistributedDataParallel\|init_process_group\|--distributed" \
  nequip/ --include="*.py" -l
ls nequip/scripts/
python -c "from nequip.scripts.train_dist import main" 2>&1 | head -3
nequip-train --help 2>&1 | grep -i distributed | head -3
```

Choose Pattern A (separate module) or Pattern B (`--distributed` flag)
based on findings, and update `scripts/preflight.py --multi-gpu` accordingly.

For 4 GPUs, ensure `batch_size`, `validation_batch_size`, `n_train`, `n_val`
are all divisible by 4 (change `batch_size: 5` → 4 or 8).

---

## 8. Timeline Estimates

| Configuration | Time per epoch | 200 epochs (worst) | Realistic (~100 ep, early stop) |
|---|---|---|---|
| 1× RTX 3090 (this repo, n007) | ~10–14 h | ~120 d | ~60 d, ~30 resubmits |
| 1× A100 40GB | ~8 h | ~67 d | ~33 d, ~12 resubmits |
| 4× A100 (DDP) | ~2 h | ~17 d | ~8 d, ~3 resubmits |
| 8× A100 (DDP) | ~1 h | ~8 d | ~4 d, ~2 resubmits |

On a single RTX 3090 expect to chain many resubmits; consider moving to
A6000 (`gpu4`/`gpu5`) or A100 if walltime matters.

---

## 9. Monitoring

See [`scripts/daily_check.py`](../scripts/daily_check.py). Run via cron
every 12 h, pipe alerts to `mail` / Slack.

---

## 10. Stopping Criteria (per-atom thresholds)

See [`scripts/check_target_metrics.py`](../scripts/check_target_metrics.py).

| Status | Energy MAE | Force MAE | Action |
|---|---|---|---|
| **우수** | < 5 meV/atom | < 0.05 eV/Å | `touch training_complete.flag` |
| **합격** | < 15 meV/atom | < 0.08 eV/Å | Continue to natural stop; proceed to ASR head |
| **실패** | > 30 meV/atom OR | > 0.15 eV/Å | Halt chain, investigate |

Auto-stop conditions are unchanged: max_epochs / LR floor / val patience /
manual / MAX_RESUBMIT.

---

## 11. Failure Mode Handling

| Symptom | Cause | Action |
|---|---|---|
| `prepare_data.py` OOM | Non-streaming code | Verify §3 Step 6 uses `append=True` + `CHUNK=10_000` |
| Val frames all clustered at low frame_idx | Pre-v4 head sampling | Use v5 linspace (already in `prepare_data.py`) |
| Val p99 << pool max | v4 tail truncation | Switch to linspace (v5) |
| `KeyError: 'forces_mae'` | Missing `metrics_components` | Verify config block |
| `RuntimeError: missing frame_idx` | Halo8 schema mismatch | Inspect `halo8_schema_probe.txt`; adapt `derive_rxn_and_frame()` |
| `RuntimeError: Cannot extract energy/forces` | Halo8 schema mismatch | Check probe `energy_forces_path` |
| `CUDA OOM` | bs=5 too large | Reduce `batch_size: 3` or use larger GPU |
| `NaN loss` | LR too high | Reduce `learning_rate: 1e-3`, restart |
| Loss flat >50 epochs | LR stuck | `lr_scheduler_factor: 0.1`, restart |
| `KeyError: 'n_train'` | Skipped prepare_data | Run `python scripts/prepare_data.py` first |
| Comments lost in config | Used `yaml.safe_dump` | Use ruamel.yaml |
| Resubmit count > 30 | Stuck loop | Halt, investigate |

---

## 12. Output Validation

See [`scripts/validate_output.py`](../scripts/validate_output.py).

---

## 13. Quick-Reference Commands

```bash
# Initial setup
conda activate reactot
python scripts/prepare_data.py             # v5: schema probe + linspace stride + stream
python scripts/preflight.py
sbatch scripts/submit_pretrain.sh

# Verify val coverage (v5 sanity)
cat output/halo8_nequip_v1/data/val_frame_distribution.txt

# Monitor
squeue -u $USER -n halo8_nequip
tail -f output/halo8_nequip_v1/logs/training.log
cat output/halo8_nequip_v1/resubmit_count.txt

# After first epoch — verify keys
python -c "
import torch
t = torch.load('output/halo8_nequip_v1/checkpoints/last.ckpt', map_location='cpu')
print(sorted(t.get('best_val_metrics', {}).keys()))
"

# Tier check
python scripts/check_target_metrics.py

# Final validation
python scripts/validate_output.py

# Emergency stop
touch output/halo8_nequip_v1/training_complete.flag
scancel -u $USER -n halo8_nequip

# Resume after fix
echo "0" > output/halo8_nequip_v1/resubmit_count.txt
sbatch scripts/submit_pretrain.sh
```

---

## 14. Pre-launch Checklist (v5)

- [ ] `data/halo8_F0_passed.json` has exactly 17,574 entries — *or* fallback path acknowledged
- [ ] `Halo_{1..10}.db` readable
- [ ] `reactot` env has `nequip>=0.6.2` AND `ruamel.yaml`
- [ ] **For DDP**: separate env with ddp branch + state-reduce; §7 discovery completed
- [ ] `python scripts/prepare_data.py` ran:
  - [ ] `halo8_schema_probe.txt` shows `rxn_frame_path` not None
  - [ ] `.prepare_done` has `val_sampling_method: numpy_linspace_endpoint_preserving`
  - [ ] `val_frame_distribution.txt`: `frame_idx_p99 ≈ pool_max`
  - [ ] peak RAM < 1 GB
- [ ] `configs/halo8_nequip_v1.yaml`:
  - [ ] `n_train`, `n_val` injected
  - [ ] `metrics_components` block present (4 entries)
  - [ ] comments preserved
  - [ ] `root: output/` (flat)
- [ ] `head -2 output/halo8_nequip_v1/data/halo8_train.extxyz` shows `energy=...` and `forces:R:3`
- [ ] `python scripts/preflight.py` passes
- [ ] `code_sha.txt` is the working-tree HEAD
- [ ] `export NEQUIP_ENABLE_TF32=1` in submit script
- [ ] After first epoch: metric keys match `check_target_metrics.py`

---

## Non-negotiable rules (final)

1. Checkpoints saved every epoch. Loss of >1 epoch never acceptable.
2. Auto-resubmit on incomplete exit up to `MAX_RESUBMIT = 50`.
3. No silent overwrites of `last.ckpt`, `best.ckpt`, `epoch_NNN.ckpt`.
4. Config matches NequIP paper (Batzner 2022 Methods).
5. `n_train` / `n_val` injected by `prepare_data.py` via ruamel.yaml.
6. Streaming data write. Never accumulate >10K Atoms in RAM.
7. Path: `root: output/`, `run_name: halo8_nequip_v1`. No nesting.
8. Val sampling: `np.linspace` endpoint-preserving uniform stride.
9. TF32 enabled via Python launcher (`NEQUIP_ENABLE_TF32=1`).
10. DDP requires `ddp` branch + `state-reduce`; module path discovered at install time.
11. Schema probe with public API; `frame_idx` is hard-required (or local `dand_id` adapter).
12. Single `extract_data()` call per row.
13. Per-atom energy MAE in meV/atom for tier classification.
14. `metrics_components` explicit in config.
15. All versions/hashes logged to `manifest.json`.
16. Resubmit count > 30 triggers alert.
17. `training_complete.flag` is one-way.
