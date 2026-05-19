#!/usr/bin/env python
"""Pre-flight checks for Halo8 NequIP 0.9.x pretraining.

Updated from the 0.6.2-era preflight: now validates the 0.9.x stack
(Lightning, torchmetrics, hydra-core, matscipy) and a 4-GPU DDP topology.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WORK = PROJECT / "output" / "halo8_nequip_v1"
HALO8_DB_DIR = Path(os.environ.get(
    "HALO8_DB_DIR",
    "/gpfs/home1/yeseo1ee/projects/ts_prediction_project/data",
))
F0_PATH = PROJECT / "data" / "halo8_F0_passed.json"
SCHEMA_PROBE = WORK / "data" / "halo8_schema_probe.txt"

errors: list[str] = []
warnings: list[str] = []


def ok(name: str, cond: bool, msg: str) -> None:
    print(f"[{'OK   ' if cond else 'FAIL '}] {name}: {msg}")
    if not cond:
        errors.append(f"{name}: {msg}")


def warn(name: str, msg: str) -> None:
    print(f"[WARN ] {name}: {msg}")
    warnings.append(f"{name}: {msg}")


parser = argparse.ArgumentParser()
parser.add_argument("--min-gpus", type=int, default=1,
                    help="Minimum number of GPUs to require (default 1).")
args = parser.parse_args()

# --- Core deps ---
try:
    import torch
    ok("torch", True, torch.__version__)
    cuda_avail = torch.cuda.is_available()
    ok("cuda available", cuda_avail, "torch.cuda.is_available()")
    if cuda_avail:
        n = torch.cuda.device_count()
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        ok(f"gpu count >= {args.min_gpus}", n >= args.min_gpus,
           f"{n} GPU(s), dev0 has {mem_gb:.1f} GB")
        ok("cuda>=11.8",
           torch.version.cuda is not None
           and tuple(int(x) for x in torch.version.cuda.split(".")[:2]) >= (11, 8),
           f"torch.version.cuda={torch.version.cuda}")
except ImportError as e:
    ok("torch", False, str(e))

# NequIP 0.9.x stack
try:
    import nequip
    nv = tuple(int(x) for x in nequip.__version__.split(".")[:3])
    ok("nequip>=0.9.0", nv >= (0, 9, 0), nequip.__version__)
except ImportError as e:
    ok("nequip", False, str(e))

try:
    import lightning
    lv = tuple(int(x) for x in lightning.__version__.split(".")[:2])
    ok("lightning>=2.0", lv >= (2, 0), lightning.__version__)
except ImportError as e:
    ok("lightning", False, str(e))

for mod, label in [
    ("torchmetrics", "torchmetrics"),
    ("hydra", "hydra-core"),
    ("matscipy", "matscipy"),
    ("ase", "ase"),
    ("e3nn", "e3nn"),
    ("numpy", "numpy"),
    ("yaml", "pyyaml"),
    ("h5py", "h5py"),
]:
    try:
        m = __import__(mod)
        ok(label, True, getattr(m, "__version__", "(unknown)"))
    except ImportError as e:
        ok(label, False, str(e))

try:
    import ruamel.yaml  # noqa: F401
    ok("ruamel.yaml", True, "available")
except ImportError as e:
    ok("ruamel.yaml", False, str(e))

try:
    from nequip.train import EMALightningModule, SimpleDDPStrategy  # noqa: F401
    from nequip.model import NequIPGNNModel  # noqa: F401
    from nequip.data.datamodule import ASEDataModule  # noqa: F401
    ok("nequip 0.9.x targets", True, "EMALightningModule / SimpleDDPStrategy / NequIPGNNModel / ASEDataModule")
except ImportError as e:
    ok("nequip 0.9.x targets", False, str(e))

# --- Disk ---
free_gb = shutil.disk_usage(PROJECT).free / 1e9
ok("disk space>=200GB", free_gb >= 200, f"{free_gb:.1f} GB free")

# --- Halo8 shards ---
expected = [HALO8_DB_DIR / f"Halo_{i}.db" for i in range(1, 11)]
missing = [p.name for p in expected if not p.exists()]
ok("halo8 db shards", not missing, f"missing: {missing}" if missing else "10 shards present")

# --- F0 list ---
if F0_PATH.exists():
    try:
        with F0_PATH.open() as f:
            n = len(json.load(f))
        ok("F0 list non-empty", n > 0, f"{n} entries")
    except Exception as e:
        ok("F0 list parse", False, str(e))
else:
    warn("F0 list", f"{F0_PATH} absent — prepare_data.py will use fallback")

# --- extxyz outputs ---
for sub in ("halo8_train.extxyz", "halo8_val.extxyz"):
    p = WORK / "data" / sub
    if p.exists():
        sz = p.stat().st_size / (1024 ** 3)
        ok(f"data/{sub}", sz > 0.01, f"{sz:.2f} GB")
    else:
        warn(f"data/{sub}", "absent — first prepare_data run will create it")

# --- Schema probe ---
if SCHEMA_PROBE.exists():
    try:
        schema = json.loads(SCHEMA_PROBE.read_text())
        ef, rf = schema.get("energy_forces_path"), schema.get("rxn_frame_path")
        ok("schema: energy/forces path", ef is not None, str(ef))
        ok("schema: rxn/frame path", rf is not None, str(rf))
    except Exception as e:
        warn("schema_probe parse", str(e))
else:
    warn("schema_probe", "not yet present — first prepare_data run will create it")

# --- SLURM ---
if shutil.which("sbatch"):
    ok("slurm", True, "sbatch on PATH")
else:
    warn("slurm", "sbatch absent — local mode only")

# --- Workdir layout ---
for sub in ("training_run", "logs/slurm", "data"):
    (WORK / sub).mkdir(parents=True, exist_ok=True)
ok("workdir layout", True, str(WORK))

print(f"\nwarnings: {len(warnings)}, errors: {len(errors)}")
if errors:
    print("\nPREFLIGHT FAILED:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(2)
print("preflight OK")
