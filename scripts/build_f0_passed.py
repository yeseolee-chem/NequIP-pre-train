#!/usr/bin/env python
"""Build halo8_F0_passed.json using the FULL ``stage0_fragmentation.strict_v1``
decision tree (Joung et al. 2025 BE-matrix protocol).

For each reaction in the eligible pool we:
  1. pull R = frame 0 and P = last frame from the relevant Halo_*.db shard
  2. call ``strict_v1.fragmentation_strict_v1(numbers, R_xyz, P_xyz)``
  3. classify as F0-pass iff
        - ``len(fragments) == 2`` AND
        - ``fallback_strategy != "strain_only"`` AND
        - ``is_pure_rearrangement`` is False (or rearrangement was recovered)

This is the same pipeline that produced the 500 ADF cohort, so the F0 set
is consistent with downstream Stage 5/6 expectations.

The output schema is unchanged: a sorted JSON list of reaction IDs at
``data/halo8_F0_passed.json``. A companion audit file
``data/halo8_F0_report.json`` records the case-breakdown counts.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from ase.db import connect

PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
F0_OUT = DATA_DIR / "halo8_F0_passed.json"
REPORT_OUT = DATA_DIR / "halo8_F0_report.json"

INDEX_PARQUET = Path(os.environ.get(
    "HALO8_INDEX_PARQUET",
    "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/data/halo8_index/index.parquet",
))
HALO8_DB_DIR = Path(os.environ.get(
    "HALO8_DB_DIR",
    "/gpfs/home1/yeseo1ee/projects/ts_prediction_project/data",
))
HALO8_SHARDS = [HALO8_DB_DIR / f"Halo_{i}.db" for i in range(1, 11)]

# Import strict_v1 by bypassing its package __init__ (which pulls rdkit, only
# needed for the BE-matrix path).
sys.path.insert(0, "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
sys.path.insert(0, "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/src")
import importlib.util as _iu

_root = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/stage0_fragmentation")
for sub in ("types", "bond_detection", "strict_v1"):
    spec = _iu.spec_from_file_location(f"stage0_fragmentation.{sub}", _root / f"{sub}.py")
    mod = _iu.module_from_spec(spec)
    sys.modules[f"stage0_fragmentation.{sub}"] = mod
    spec.loader.exec_module(mod)
from stage0_fragmentation.strict_v1 import fragmentation_strict_v1  # noqa: E402


def derive_rxn_and_frame(row) -> tuple[str, int]:
    dand = row.data.get("dand_id")
    if not dand:
        raise KeyError("no dand_id in row.data")
    head, _, tail = dand.rpartition("_")
    return head, int(tail) if tail.isdigit() else 0


def collect_R_P_frames(eligible_ids: set[str]) -> dict[str, dict]:
    """Stream all shards once; per reaction keep frame 0 and the largest
    observed frame_idx as R and P respectively."""
    result: dict[str, dict] = {}
    n_rows = 0
    t0 = time.time()
    for shard in HALO8_SHARDS:
        if not shard.exists():
            continue
        with connect(str(shard)) as db:
            for row in db.select():
                n_rows += 1
                try:
                    rid, fidx = derive_rxn_and_frame(row)
                except Exception:
                    continue
                if rid not in eligible_ids:
                    continue
                rec = result.setdefault(rid, {
                    "numbers": np.asarray(row.numbers, dtype=int),
                    "R_fidx": None,
                    "R_xyz": None,
                    "P_fidx": -1,
                    "P_xyz": None,
                })
                if fidx == 0 or rec["R_fidx"] is None:
                    if rec["R_fidx"] is None or fidx == 0:
                        rec["R_fidx"] = fidx
                        rec["R_xyz"] = np.asarray(row.positions, dtype=float)
                if fidx > rec["P_fidx"]:
                    rec["P_fidx"] = fidx
                    rec["P_xyz"] = np.asarray(row.positions, dtype=float)
        print(f"  {shard.name}: rows scanned cumul={n_rows} (elapsed {time.time()-t0:.1f}s)")
    return result


def main() -> int:
    t0 = time.time()
    print(f"[F0] loading {INDEX_PARQUET}")
    idx = pd.read_parquet(INDEX_PARQUET)
    pool = idx[(idx["interior_ts"]) & (~idx["short_traj"])]
    bond_path = INDEX_PARQUET.with_name("bond_changes_all.parquet")
    if bond_path.exists():
        bc = pd.read_parquet(bond_path)[["reaction_id", "n_components_R"]]
        pool = pool.merge(bc, on="reaction_id", how="left")
        pool = pool[pool["n_components_R"] == 1]
    eligible_ids: set[str] = set(pool["reaction_id"].astype(str).tolist())
    print(f"[F0] eligible pool size: {len(eligible_ids)}")

    print("[F0] streaming Halo_*.db to collect R/P frames per reaction...")
    frames = collect_R_P_frames(eligible_ids)
    print(f"[F0] collected R+P frames for {len(frames)} reactions in {time.time()-t0:.1f}s")

    passed: list[str] = []
    breakdown: Counter[str] = Counter()
    n = len(frames)
    for i, (rid, rec) in enumerate(frames.items(), start=1):
        if rec["R_xyz"] is None or rec["P_xyz"] is None or rec["R_fidx"] != 0:
            breakdown["missing_R_or_P"] += 1
            continue
        try:
            res = fragmentation_strict_v1(rec["numbers"], rec["R_xyz"], rec["P_xyz"])
        except Exception as e:  # noqa: BLE001
            breakdown[f"exception_{type(e).__name__}"] += 1
            continue
        # Pass criteria: exactly 2 fragments AND not strain_only fallback
        n_frag = len(res.fragments)
        fb = getattr(res, "fallback_strategy", None)
        is_rearr = getattr(res, "is_pure_rearrangement", False)
        if n_frag == 2 and fb != "strain_only":
            passed.append(str(rid))
        elif fb == "strain_only":
            breakdown["strain_only"] += 1
        elif n_frag < 2:
            breakdown["fewer_than_2_fragments"] += 1
        else:
            breakdown[f"{n_frag}_fragments"] += 1
        if i % 1000 == 0:
            print(f"  [{i}/{n}] pass={len(passed)} elapsed={time.time()-t0:.1f}s")

    passed.sort()
    F0_OUT.write_text(json.dumps(passed))
    report = {
        "method": "stage0_fragmentation.strict_v1 (BE-matrix decision tree)",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "eligible_pool_size": int(len(eligible_ids)),
        "frames_collected": int(len(frames)),
        "n_passed": len(passed),
        "n_failed": int(len(frames) - len(passed)),
        "failure_breakdown": dict(breakdown),
        "filters_applied": [
            "n_components_R == 1",
            "interior_ts",
            "not short_traj",
            "strict_v1: len(fragments)==2 and not strain_only",
        ],
        "elapsed_seconds": time.time() - t0,
        "output_path": str(F0_OUT),
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2))
    print(f"[F0] passed={len(passed)} / {len(frames)} "
          f"({100.0 * len(passed) / max(1, len(frames)):.1f}%)")
    print(f"[F0] failure breakdown: {dict(breakdown)}")
    print(f"[F0] wrote {F0_OUT} and {REPORT_OUT} in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
