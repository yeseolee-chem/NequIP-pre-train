#!/usr/bin/env python
"""Prepare Halo8 train/val extxyz files for NequIP pretraining (Spec §3).

Reaction selection precedence:
  1. `data/halo8_F0_passed.json` (a JSON list of reaction IDs) if present —
     this is the authoritative 17,574-reaction F0-passed set per the spec.
  2. Otherwise: read `index.parquet` from eda-asm-prediction and filter to
     reactions that pass minimum sanity checks (`interior_ts`, `not short_traj`,
     `n_components_R == 1`). Cap to exactly 17,574 with a deterministic
     RNG (seed=42) so the chosen set is reproducible. Status logged.
  3. `data/v2_holdout_ids.json` is removed from the set if present; ignored
     silently otherwise (user said it doesn't exist yet).

Always reaction-level train/val split (95/5) — frames from the same reaction
stay in the same split. Output: halo8_train.extxyz, halo8_val.extxyz,
train_rxn_ids.json, val_rxn_ids.json, used_reaction_ids.json, dataset_hash.txt.

Idempotent: skips if `.prepare_done` sentinel exists (use --force to redo).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

from ase.db import connect
from ase.io import write

PROJECT = Path(__file__).resolve().parents[1]
HALO8_DB_DIR = Path("/gpfs/home1/yeseo1ee/projects/ts_prediction_project/data")
HALO8_SHARDS = [HALO8_DB_DIR / f"Halo_{i}.db" for i in range(1, 11)]
F0_PATH = PROJECT / "data" / "halo8_F0_passed.json"
V2_PATH = PROJECT / "data" / "v2_holdout_ids.json"
INDEX_PARQUET = Path(
    "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/data/halo8_index/index.parquet"
)
OUT_DIR = PROJECT / "output" / "halo8_nequip_v1" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DONE_FLAG = OUT_DIR / ".prepare_done"
TARGET_N_RXNS = 17_574


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def derive_rxn_and_frame(row) -> tuple[str, int]:
    rid = getattr(row, "reaction_id", None)
    fidx = getattr(row, "frame_idx", None)
    if rid is not None and fidx is not None:
        return str(rid), int(fidx)
    data = getattr(row, "data", {}) or {}
    dand = data.get("dand_id")
    if dand:
        head, _, tail = dand.rpartition("_")
        return head, int(tail) if tail.isdigit() else 0
    raise KeyError("row has no reaction_id/frame_idx and no data.dand_id")


def attach_info(atoms, rid: str, fidx: int) -> None:
    atoms.info["reaction_id"] = rid
    atoms.info["frame_idx"] = fidx
    if "energy" not in atoms.info:
        try:
            atoms.info["energy"] = atoms.get_potential_energy()
        except Exception:
            pass


def select_reactions(seed: int, cap: int) -> tuple[set[str], dict]:
    """Return (selected_set, provenance) ready for filtering."""
    if F0_PATH.exists():
        with F0_PATH.open() as f:
            f0 = set(json.load(f))
        prov = {
            "source": "F0_PASSED_LIST",
            "path": str(F0_PATH),
            "n_in_list": len(f0),
        }
        print(f"[prepare_data] F0-passed list found: {len(f0)} reactions")
        return f0, prov

    if not INDEX_PARQUET.exists():
        raise SystemExit(
            f"Neither {F0_PATH} nor {INDEX_PARQUET} is available — cannot select reactions."
        )

    import pandas as pd
    df = pd.read_parquet(INDEX_PARQUET)
    print(f"[prepare_data] index.parquet: {len(df)} reactions total "
          f"(sources: {df['source'].value_counts().to_dict()})")
    import pandas as _pd
    bond_path = INDEX_PARQUET.with_name("bond_changes_all.parquet")
    if bond_path.exists():
        bc = _pd.read_parquet(bond_path)[["reaction_id", "n_components_R"]]
        df = df.merge(bc, on="reaction_id", how="left")
        before = len(df)
        df = df[df["n_components_R"] == 1]
        print(f"[prepare_data] n_components_R==1 filter: {len(df)}/{before}")
    df = df[df["interior_ts"]]
    df = df[~df["short_traj"]]
    print(f"[prepare_data] after interior_ts & !short_traj: {len(df)}")

    rxn_list = sorted(df["reaction_id"].astype(str).tolist())
    if cap and len(rxn_list) > cap:
        random.Random(seed).shuffle(rxn_list)
        rxn_list = sorted(rxn_list[:cap])
        print(f"[prepare_data] capped to exactly {cap} reactions (seed={seed})")
    selected = set(rxn_list)
    prov = {
        "source": "INDEX_PARQUET_FALLBACK",
        "path": str(INDEX_PARQUET),
        "filters": "interior_ts & !short_traj & n_components_R==1",
        "cap": cap if cap and len(selected) == cap else None,
        "n_selected": len(selected),
        "WARNING": f"True F0-passed list ({F0_PATH}) not provided; using a "
                   f"deterministic {cap}-reaction subset of the index. "
                   "Provide the real list and re-run with --force to override.",
    }
    return selected, prov


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--target-n", type=int, default=TARGET_N_RXNS,
                   help="Target number of reactions (default: 17574 per spec).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing outputs even if .prepare_done is present.")
    args = p.parse_args()

    if DONE_FLAG.exists() and not args.force:
        print(f"[prepare_data] DONE_FLAG exists at {DONE_FLAG} — skipping.")
        return 0

    t0 = time.time()

    # ----- Reaction selection -----
    selected, sel_prov = select_reactions(args.seed, args.target_n)
    v2_holdout: set[str] = set()
    if V2_PATH.exists():
        with V2_PATH.open() as f:
            v2_holdout = set(json.load(f))
        print(f"[prepare_data] V2 holdout: removing {len(v2_holdout)} reactions")
        selected -= v2_holdout
    else:
        print(f"[prepare_data] V2 holdout list absent — no exclusion applied")

    if not selected:
        print("[prepare_data] ERROR: no reactions selected.")
        return 3
    print(f"[prepare_data] FINAL selected reactions: {len(selected)}")

    # ----- Reaction-level split -----
    rxn_list = sorted(selected)
    random.Random(args.seed).shuffle(rxn_list)
    n_train = int((1 - args.val_frac) * len(rxn_list))
    train_rxns = set(rxn_list[:n_train])
    val_rxns = set(rxn_list[n_train:])
    print(f"[prepare_data] split: {len(train_rxns)} train / {len(val_rxns)} val")
    (OUT_DIR / "train_rxn_ids.json").write_text(json.dumps(sorted(train_rxns)))
    (OUT_DIR / "val_rxn_ids.json").write_text(json.dumps(sorted(val_rxns)))
    (OUT_DIR / "used_reaction_ids.json").write_text(
        json.dumps({
            "seed": args.seed,
            "val_frac": args.val_frac,
            "n_train_reactions": len(train_rxns),
            "n_val_reactions": len(val_rxns),
            "target_n": args.target_n,
            "v2_holdout_used": str(V2_PATH) if v2_holdout else None,
            "selection_provenance": sel_prov,
        }, indent=2)
    )

    # ----- Stream shards, write extxyz -----
    train_path = OUT_DIR / "halo8_train.extxyz"
    val_path = OUT_DIR / "halo8_val.extxyz"
    if train_path.exists():
        train_path.unlink()
    if val_path.exists():
        val_path.unlink()

    n_train_frames = n_val_frames = n_skip = 0
    print(f"[prepare_data] streaming {len(HALO8_SHARDS)} shards...")
    for shard in HALO8_SHARDS:
        if not shard.exists():
            print(f"  missing {shard.name}")
            continue
        s_tr, s_va = [], []
        with connect(str(shard)) as db:
            for row in db.select():
                try:
                    rid, fidx = derive_rxn_and_frame(row)
                except KeyError:
                    n_skip += 1
                    continue
                if rid in train_rxns:
                    atoms = row.toatoms()
                    attach_info(atoms, rid, fidx)
                    s_tr.append(atoms)
                elif rid in val_rxns:
                    atoms = row.toatoms()
                    attach_info(atoms, rid, fidx)
                    s_va.append(atoms)
        if s_tr:
            write(str(train_path), s_tr, format="extxyz", append=True)
            n_train_frames += len(s_tr)
        if s_va:
            write(str(val_path), s_va, format="extxyz", append=True)
            n_val_frames += len(s_va)
        print(f"  {shard.name}: +{len(s_tr)} train, +{len(s_va)} val "
              f"(running totals {n_train_frames}/{n_val_frames}) "
              f"elapsed {time.time()-t0:.0f}s")

    print(f"[prepare_data] frames written: train={n_train_frames}, val={n_val_frames}, "
          f"skipped_rows={n_skip}")
    if n_train_frames == 0:
        print("[prepare_data] ERROR: zero training frames written.")
        return 4

    # ----- Hashes -----
    print("[prepare_data] computing SHA-256 hashes...")
    train_h = file_hash(train_path)
    val_h = file_hash(val_path)
    (OUT_DIR / "dataset_hash.txt").write_text(
        f"train: {train_h}\nval:   {val_h}\n"
    )
    print(f"  train: {train_h}\n  val:   {val_h}")

    DONE_FLAG.write_text(json.dumps({
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_train_frames": n_train_frames,
        "n_val_frames": n_val_frames,
        "n_train_reactions": len(train_rxns),
        "n_val_reactions": len(val_rxns),
        "elapsed_seconds": time.time() - t0,
        "selection_provenance": sel_prov,
    }, indent=2))
    print(f"[prepare_data] DONE in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
