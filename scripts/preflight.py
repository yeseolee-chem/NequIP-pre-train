#!/usr/bin/env python
"""Pre-flight checks for Halo8 NequIP pretraining (Spec v5 §1).

Hard-fail if any required check fails. Schema check uses the v5 probe but
also accepts the local Halo8 ``row.data['dand_id']`` convention as a valid
``frame_idx`` source (the on-disk shards don't expose top-level
``reaction_id``/``frame_idx`` attributes).
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
HALO8_DB_DIR = Path("/gpfs/home1/yeseo1ee/projects/ts_prediction_project/data")
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
parser.add_argument("--multi-gpu", action="store_true", help="Require DDP-branch nequip")
args = parser.parse_args()

# --- Python deps ---
try:
    import torch
    ok("torch", True, torch.__version__)
    ok(
        "torch>=2.0",
        tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 0),
        torch.__version__,
    )
    cuda_avail = torch.cuda.is_available()
    ok("cuda available", cuda_avail, "torch.cuda.is_available()")
    if cuda_avail:
        n = torch.cuda.device_count()
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        ok("gpu memory >=24GB", mem_gb >= 23.5, f"{n} GPU(s), dev0 has {mem_gb:.1f} GB")
        ok(
            "cuda>=11.8",
            torch.version.cuda is not None
            and tuple(int(x) for x in torch.version.cuda.split(".")[:2]) >= (11, 8),
            f"torch.version.cuda={torch.version.cuda}",
        )
except ImportError as e:
    ok("torch", False, str(e))

try:
    import nequip
    nequip_ver = tuple(int(x) for x in nequip.__version__.split(".")[:3])
    ok("nequip>=0.6.2", nequip_ver >= (0, 6, 2), nequip.__version__)
except ImportError as e:
    ok("nequip", False, str(e))

for mod in ("ase", "e3nn", "yaml", "h5py", "numpy"):
    try:
        m = __import__(mod)
        ok(mod, True, getattr(m, "__version__", "(unknown)"))
    except ImportError as e:
        ok(mod, False, str(e))

# ruamel.yaml — used by prepare_data.py to inject n_train/n_val without losing comments
try:
    import ruamel.yaml  # noqa: F401
    ok("ruamel.yaml", True, "available")
except ImportError as e:
    ok("ruamel.yaml", False, f"{e} (pip install 'ruamel.yaml>=0.17.0')")

# Optional packages
try:
    import wandb  # noqa: F401
    ok("wandb", True, wandb.__version__)
except ImportError:
    warn("wandb", "not installed (config has wandb: false so this is fine)")

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
        ok("F0 list == 17574", n == 17_574, f"{n} entries")
    except Exception as e:
        ok("F0 list parse", False, str(e))
else:
    warn(
        "F0 list",
        f"{F0_PATH} absent — prepare_data.py will fall back to index.parquet "
        "(deterministic 17,574-reaction subset, seed=42)",
    )

# --- Schema probe (only available after a first prepare_data run) ---
if SCHEMA_PROBE.exists():
    try:
        schema = json.loads(SCHEMA_PROBE.read_text())
        ef = schema.get("energy_forces_path")
        rf = schema.get("rxn_frame_path")
        ok("schema: energy/forces path", ef is not None, str(ef))
        ok("schema: rxn/frame path", rf is not None, str(rf))
    except Exception as e:
        warn("schema_probe parse", str(e))
else:
    warn("schema_probe", f"{SCHEMA_PROBE} not yet present — first prepare_data run will create it")

# --- WandB key ---
if not os.environ.get("WANDB_API_KEY"):
    warn("wandb", "WANDB_API_KEY not set; continuing with wandb disabled")
else:
    ok("wandb key", True, "WANDB_API_KEY set")

# --- SLURM ---
if shutil.which("sbatch"):
    ok("slurm", True, "sbatch on PATH")
else:
    warn("slurm", "sbatch absent — local mode only")

# --- DDP requirement when multi-gpu ---
if args.multi_gpu:
    pattern = None
    try:
        from nequip.scripts.train_dist import main as _dist_main  # noqa: F401
        pattern = "module_import"
    except ImportError:
        try:
            import subprocess
            r = subprocess.run(["nequip-train", "--help"], capture_output=True, text=True, timeout=15)
            if "--distributed" in r.stdout:
                pattern = "flag"
        except Exception:
            pass
    ok("ddp branch detected", pattern is not None,
       f"pattern={pattern}" if pattern else "neither train_dist module nor --distributed flag")

# --- Workdir layout ---
for sub in ("checkpoints", "logs/slurm", "logs/tensorboard", "data"):
    (WORK / sub).mkdir(parents=True, exist_ok=True)
ok("workdir layout", True, str(WORK))

print(f"\nwarnings: {len(warnings)}, errors: {len(errors)}")
if errors:
    print("\nPREFLIGHT FAILED:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(2)
print("preflight OK")
