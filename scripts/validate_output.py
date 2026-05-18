#!/usr/bin/env python
"""Post-training output validation (Spec v5 §12).

Verifies required artifacts, computes the final tier classification, and
writes ``manifest.json`` (versions, hashes, run metadata).
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import nequip
import torch

PROJECT = Path(__file__).resolve().parents[1]
WORK = PROJECT / "output" / "halo8_nequip_v1"


def must_exist(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"missing: {path}")


def main() -> int:
    for name in ("best_model.pth", "last_model.pth", "config_used.yaml",
                 "training_log.csv", "code_sha.txt"):
        must_exist(WORK / name)

    ckpt = torch.load(WORK / "best_model.pth", map_location="cpu")
    val_metrics = ckpt.get("best_val_metrics", {})
    energy_meV = float(val_metrics["total_energy_mae_per_atom"]) * 1000.0
    force_evA = float(val_metrics["forces_mae"])
    if energy_meV > 30.0 or force_evA > 0.15:
        raise SystemExit(
            f"FAILURE tier: energy={energy_meV:.2f} meV/atom, force={force_evA:.4f} eV/Å"
        )
    tier = (
        "우수" if (energy_meV < 5.0 and force_evA < 0.05)
        else "합격" if (energy_meV < 15.0 and force_evA < 0.08)
        else "between-tiers"
    )

    prep = json.loads((WORK / "data" / ".prepare_done").read_text())
    manifest = {
        "pipeline_version": "halo8_nequip_v5",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "n_reactions_used": 17_574,
        "n_train_reactions": prep["n_train_reactions"],
        "n_val_reactions": prep["n_val_reactions"],
        "n_train_frames": prep["n_train_frames"],
        "n_val_frames": prep["n_val_frames"],
        "val_sampling_method": prep["val_sampling_method"],
        "val_frame_idx_median": prep.get("val_frame_idx_median"),
        "val_frame_idx_p99": prep.get("val_frame_idx_p99"),
        "reference_level_of_theory": "omegaB97X-3c",
        "final_val_energy_mae_meV_per_atom": energy_meV,
        "final_val_force_mae_evA": force_evA,
        "tier": tier,
        "total_epochs_trained": int(ckpt.get("epoch", 0)),
        "total_resubmit_count": int((WORK / "resubmit_count.txt").read_text().strip()),
        "config_hash": hashlib.sha256(
            (WORK / "config_used.yaml").read_bytes()
        ).hexdigest(),
        "dataset_hash": (WORK / "data" / "dataset_hash.txt").read_text(),
        "code_sha": (WORK / "code_sha.txt").read_text().strip(),
        "nequip_version": nequip.__version__,
        "torch_version": torch.__version__,
    }
    (WORK / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[validate] ✅ tier={tier} — manifest written to {WORK / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
