#!/usr/bin/env python
"""Pre-flight checks for Halo8 NequIP pretraining (Spec §1).

Aborts with non-zero exit code if any hard check fails.
Warnings are printed but allowed (e.g. WandB key missing).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WORK = PROJECT / "output" / "halo8_nequip_v1"
HALO8_DB_DIR = Path("/gpfs/home1/yeseo1ee/projects/ts_prediction_project/data")
F0_PATH = PROJECT / "data" / "halo8_F0_passed.json"
V2_PATH = PROJECT / "data" / "v2_holdout_ids.json"

errors: list[str] = []
warnings: list[str] = []


def check(name: str, ok: bool, msg: str) -> None:
    status = "OK   " if ok else "FAIL "
    print(f"[{status}] {name}: {msg}")
    if not ok:
        errors.append(f"{name}: {msg}")


def warn(name: str, msg: str) -> None:
    print(f"[WARN ] {name}: {msg}")
    warnings.append(f"{name}: {msg}")


# 1. PyTorch / CUDA
try:
    import torch
    check("torch", True, f"version {torch.__version__}")
    check(
        "torch>=2.0",
        tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 0),
        torch.__version__,
    )
    cuda_ok = torch.cuda.is_available()
    check("cuda available", cuda_ok, "torch.cuda.is_available()")
    if cuda_ok:
        n = torch.cuda.device_count()
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        check("gpu memory", mem_gb >= 16, f"{n} GPU(s), {mem_gb:.1f} GB on dev0")
        check(
            "cuda version",
            torch.version.cuda is not None
            and tuple(int(x) for x in torch.version.cuda.split(".")[:2]) >= (11, 8),
            f"torch.version.cuda={torch.version.cuda}",
        )
except ImportError as e:
    check("torch", False, str(e))

# 2. NequIP
try:
    import nequip
    check("nequip", True, f"version {nequip.__version__}")
    check(
        "nequip>=0.6",
        tuple(int(x) for x in nequip.__version__.split(".")[:2]) >= (0, 6),
        nequip.__version__,
    )
except ImportError as e:
    check("nequip", False, f"{e} (pip install 'nequip>=0.6.0,<0.7.0')")

# 3. ASE / e3nn / yaml
for mod in ("ase", "e3nn", "yaml", "h5py", "numpy"):
    try:
        m = __import__(mod)
        check(mod, True, getattr(m, "__version__", "(unknown)"))
    except ImportError as e:
        check(mod, False, str(e))

# 4. Disk space
free_gb = shutil.disk_usage(PROJECT).free / 1e9
check("disk space", free_gb >= 100, f"{free_gb:.1f} GB free")

# 5. Halo8 DB files (10 shards expected)
expected = [HALO8_DB_DIR / f"Halo_{i}.db" for i in range(1, 11)]
missing = [p.name for p in expected if not p.exists()]
check("halo8 db shards", not missing, f"missing: {missing}" if missing else "10 shards present")

# 6. F0-passed list (optional — will fall back to "use all reactions" if missing)
if F0_PATH.exists():
    try:
        import json
        with F0_PATH.open() as f:
            f0 = json.load(f)
        count_ok = len(f0) == 17_574
        if not count_ok:
            warn("F0 list", f"expected 17574, got {len(f0)}")
        else:
            check("F0 list", True, f"{len(f0)} reactions (matches spec)")
    except Exception as e:
        check("F0 list", False, str(e))
else:
    warn(
        "F0 list",
        f"{F0_PATH} not present — prepare_data.py will fall back to all Halo8 reactions",
    )

# 7. V2 holdout (optional)
if V2_PATH.exists():
    try:
        import json
        with V2_PATH.open() as f:
            v2 = json.load(f)
        check("V2 holdout", True, f"{len(v2)} ids")
    except Exception as e:
        check("V2 holdout", False, str(e))
else:
    warn("V2 holdout", f"{V2_PATH} not present — no holdout will be excluded")

# 8. WandB key
if not os.environ.get("WANDB_API_KEY"):
    warn("wandb", "WANDB_API_KEY not set; continuing with wandb disabled")
else:
    check("wandb key", True, "WANDB_API_KEY set")

# 9. SLURM available
if shutil.which("sbatch"):
    check("slurm", True, "sbatch available")
else:
    warn("slurm", "sbatch not on PATH — local-mode only")

# 10. Working dir layout
for sub in ("checkpoints", "logs/slurm", "data"):
    p = WORK / sub
    p.mkdir(parents=True, exist_ok=True)
check("workdir layout", True, str(WORK))

print()
print(f"warnings: {len(warnings)}, errors: {len(errors)}")
if errors:
    print("\nPREFLIGHT FAILED:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(2)
print("preflight OK")
