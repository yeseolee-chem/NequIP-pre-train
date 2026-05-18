#!/usr/bin/env python
"""Check best validation metrics against Spec v5 §10 tiered targets.

Tiers (per-atom energy MAE, per-atom force MAE — converted to meV/atom and eV/Å):

    excellent (우수): energy < 5 meV/atom  AND force < 0.05 eV/Å
                       → write `training_complete.flag`
    acceptable (합격): energy < 15 meV/atom AND force < 0.08 eV/Å
                       → continue to natural stop; proceed to ASR head
    failure  (실패): energy > 30 meV/atom OR  force > 0.15 eV/Å
                       → halt chain, investigate
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
WORK = PROJECT / "output" / "halo8_nequip_v1"
BEST = WORK / "checkpoints" / "best.ckpt"


def main() -> int:
    if not BEST.exists():
        print(f"[check] {BEST} not found yet")
        return 1
    ckpt = torch.load(BEST, map_location="cpu")
    val_metrics = ckpt.get("best_val_metrics", {})
    available = sorted(val_metrics.keys())
    print(f"[check] available metric keys: {available}")

    energy_key = "total_energy_mae_per_atom"
    if energy_key not in val_metrics:
        energy_key = next((k for k in available if "energy_mae" in k.lower()), None)
        if energy_key is None:
            raise KeyError(f"No energy MAE key in {available}")
        print(f"[check] fallback energy key: {energy_key}")

    force_key = "forces_mae"
    if force_key not in val_metrics:
        force_key = next((k for k in available if "forces_mae" in k.lower()), None)
        if force_key is None:
            raise KeyError(f"No force MAE key in {available}")
        print(f"[check] fallback force key: {force_key}")

    energy_meV = float(val_metrics[energy_key]) * 1000.0
    force_evA = float(val_metrics[force_key])
    print(f"[check] Best validation:")
    print(f"[check]   Energy MAE: {energy_meV:.2f} meV/atom")
    print(f"[check]   Force  MAE: {force_evA:.4f} eV/Å")

    if energy_meV < 5.0 and force_evA < 0.05:
        print("[check] ✅ 우수 — marking complete.")
        (WORK / "training_complete.flag").touch()
        return 0
    if energy_meV < 15.0 and force_evA < 0.08:
        print("[check] ⚪ 합격 — continue to natural stop.")
        return 0
    if energy_meV > 30.0 or force_evA > 0.15:
        print("[check] ❌ 실패 — halt chain, investigate.")
        return 2
    print("[check] · between tiers — continue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
