#!/usr/bin/env python
"""Daily monitoring check (Spec v5 §9).

Run via cron every 12h. Emits ⚠ / ✅ lines to stdout — pipe through
``mail`` or ``slack-cli`` if you want alerts.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
WORK = PROJECT / "output" / "halo8_nequip_v1"


def main() -> None:
    count_file = WORK / "resubmit_count.txt"
    count = int(count_file.read_text().strip()) if count_file.exists() else 0
    if count > 30:
        print(f"⚠  ALERT: resubmit count {count} > 30 — investigate")

    ckpt = WORK / "checkpoints" / "last.ckpt"
    if ckpt.exists():
        age_hr = (time.time() - ckpt.stat().st_mtime) / 3600
        if age_hr > 12:
            print(f"⚠  ALERT: last.ckpt is {age_hr:.1f}h old — training may be stuck")
        try:
            t = torch.load(ckpt, map_location="cpu")
        except Exception as e:
            print(f"⚠  ALERT: cannot load last.ckpt ({e})")
            t = {}
        val_hist = t.get("val_loss_history", []) if isinstance(t, dict) else []
        if len(val_hist) > 10:
            recent = val_hist[-10:]
            if all(recent[i] >= recent[i - 1] for i in range(1, len(recent))):
                print("⚠  ALERT: validation loss not decreasing for 10 epochs")

    free_gb = shutil.disk_usage(WORK).free / 1e9
    if free_gb < 50:
        print(f"⚠  ALERT: only {free_gb:.1f} GB free — clean old checkpoints/logs")

    if (WORK / "training_complete.flag").exists():
        print(f"✅ training complete after {count} resubmits")


if __name__ == "__main__":
    main()
