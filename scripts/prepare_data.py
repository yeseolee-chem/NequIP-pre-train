#!/usr/bin/env python
"""Halo8 NequIP data preparation (Spec v5 §3).

Implements the v5 spec with one local adaptation: the on-disk Halo8 shards
expose the reaction-trajectory key as ``row.data["dand_id"]`` instead of
top-level ``row.reaction_id`` / ``row.frame_idx`` attributes (see
``../CLAUDE.md`` in the sibling eda-asm-prediction repo for the encoding).
The schema probe records which path is in use and the rest of the v5
algorithm (streaming write, np.linspace val sampling, ruamel.yaml config
injection, etc.) is followed verbatim.

Run:

    python scripts/prepare_data.py            # use existing F0 list if present
    python scripts/prepare_data.py --force    # overwrite outputs

Idempotent: skips if ``.prepare_done`` already exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.io import write

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parents[1]
WORK = PROJECT / "output" / "halo8_nequip_v1"
DATA = WORK / "data"
DATA.mkdir(parents=True, exist_ok=True)

# Override either path via env vars if the data moves.
HALO8_DB_DIR = Path(os.environ.get(
    "HALO8_DB_DIR",
    "/gpfs/home1/yeseo1ee/projects/ts_prediction_project/data",
))
HALO8_SHARDS = [HALO8_DB_DIR / f"Halo_{i}.db" for i in range(1, 11)]
F0_PATH = PROJECT / "data" / "halo8_F0_passed.json"
INDEX_PARQUET = Path(os.environ.get(
    "HALO8_INDEX_PARQUET",
    "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/data/halo8_index/index.parquet",
))
BOND_PARQUET = INDEX_PARQUET.with_name("bond_changes_all.parquet")

TRAIN_PATH = DATA / "halo8_train.extxyz"
VAL_PATH = DATA / "halo8_val.extxyz"
DONE_FLAG = DATA / ".prepare_done"
SCHEMA_PROBE = DATA / "halo8_schema_probe.txt"
VAL_DIST = DATA / "val_frame_distribution.txt"

TARGET_N_RXNS = 17_574
VAL_TARGET_FRAMES = 50_000
CHUNK = 10_000
SPEC_VERSION = "v5"


# --------------------------------------------------------------------------
# Schema probe — Spec v5 §3 Step 0 with local dand_id adaptation
# --------------------------------------------------------------------------
def probe_schema() -> dict:
    """Probe the first row of the first shard and decide access patterns."""
    schema: dict = {"spec_version": SPEC_VERSION}
    shard = next((p for p in HALO8_SHARDS if p.exists()), None)
    if shard is None:
        raise RuntimeError(f"No Halo8 shards found under {HALO8_DB_DIR}")

    with connect(str(shard)) as db:
        for row in db.select(limit=1):
            schema["probe_shard"] = shard.name
            schema["has_energy_attr"] = hasattr(row, "energy")
            schema["has_forces_attr"] = hasattr(row, "forces")
            schema["has_reaction_id"] = hasattr(row, "reaction_id")
            schema["has_frame_idx"] = hasattr(row, "frame_idx")

            try:
                schema["row_kvp_keys"] = sorted(row.key_value_pairs.keys())
            except Exception as e:  # noqa: BLE001
                schema["row_kvp_keys_error"] = str(e)

            schema["row_public_attrs"] = sorted(
                a for a in dir(row) if not a.startswith("_")
            )

            data = getattr(row, "data", {}) or {}
            schema["row_data_keys"] = sorted(data.keys())
            schema["has_dand_id"] = "dand_id" in data
            if schema["has_dand_id"]:
                schema["example_dand_id"] = data["dand_id"]

            atoms = row.toatoms()
            schema["toatoms_has_calc"] = atoms.calc is not None
            if atoms.calc is not None:
                schema["calc_results"] = sorted(atoms.calc.results.keys())
            break

    # ---- Decide energy/forces access pattern ----
    if schema.get("has_energy_attr") and schema.get("has_forces_attr"):
        schema["energy_forces_path"] = "row_attr"
    elif (
        schema.get("toatoms_has_calc")
        and "energy" in schema.get("calc_results", [])
        and "forces" in schema.get("calc_results", [])
    ):
        schema["energy_forces_path"] = "atoms_calc"
    else:
        schema["energy_forces_path"] = None

    # ---- Decide reaction_id / frame_idx access pattern ----
    if schema.get("has_reaction_id") and schema.get("has_frame_idx"):
        schema["rxn_frame_path"] = "row_attr"
    elif schema.get("has_dand_id"):
        schema["rxn_frame_path"] = "dand_id_in_row_data"
    else:
        schema["rxn_frame_path"] = None

    return schema


def derive_rxn_and_frame(row, mode: str) -> tuple[str, int]:
    if mode == "row_attr":
        return str(row.reaction_id), int(row.frame_idx)
    if mode == "dand_id_in_row_data":
        dand = row.data["dand_id"]
        head, _, tail = dand.rpartition("_")
        if not tail.isdigit():
            raise ValueError(f"Cannot parse frame_idx from dand_id={dand!r}")
        return head, int(tail)
    raise RuntimeError(f"Unknown rxn_frame_path mode: {mode!r}")


def extract_data(row, ef_mode: str):
    """Return (atoms, energy, forces) — single ``toatoms()`` call per row."""
    atoms = row.toatoms()
    if ef_mode == "row_attr":
        return atoms, float(row.energy), np.asarray(row.forces)
    if ef_mode == "atoms_calc":
        return atoms, float(atoms.get_potential_energy()), np.asarray(atoms.get_forces())
    raise RuntimeError(f"Unknown energy_forces_path mode: {ef_mode!r}")


# --------------------------------------------------------------------------
# Reaction selection — F0_passed list preferred; fallback to index.parquet
# --------------------------------------------------------------------------
def select_reactions(seed: int) -> tuple[set[str], dict]:
    if F0_PATH.exists():
        with F0_PATH.open() as f:
            f0 = set(json.load(f))
        if len(f0) != TARGET_N_RXNS:
            raise RuntimeError(
                f"F0-passed list size mismatch: expected {TARGET_N_RXNS}, got {len(f0)}"
            )
        return f0, {"source": "F0_PASSED_LIST", "path": str(F0_PATH), "n": len(f0)}

    if not INDEX_PARQUET.exists():
        raise SystemExit(
            f"Neither {F0_PATH} nor {INDEX_PARQUET} is available — cannot select reactions."
        )

    import pandas as pd

    df = pd.read_parquet(INDEX_PARQUET)
    if BOND_PARQUET.exists():
        bc = pd.read_parquet(BOND_PARQUET)[["reaction_id", "n_components_R"]]
        df = df.merge(bc, on="reaction_id", how="left")
        df = df[df["n_components_R"] == 1]
    df = df[df["interior_ts"] & ~df["short_traj"]]

    rxn_list = sorted(df["reaction_id"].astype(str).tolist())
    if len(rxn_list) < TARGET_N_RXNS:
        raise RuntimeError(
            f"Index parquet has only {len(rxn_list)} eligible reactions; need {TARGET_N_RXNS}"
        )
    random.Random(seed).shuffle(rxn_list)
    selected = set(sorted(rxn_list[:TARGET_N_RXNS]))
    return selected, {
        "source": "INDEX_PARQUET_FALLBACK",
        "path": str(INDEX_PARQUET),
        "filters": "interior_ts & !short_traj & n_components_R==1",
        "n_eligible": len(rxn_list),
        "n_selected": len(selected),
        "WARNING": "True F0-passed list not provided — deterministic 17574-reaction subset",
    }


# --------------------------------------------------------------------------
# Config injection — Spec v5 §3 Step 6 (ruamel.yaml preserves comments)
# --------------------------------------------------------------------------
def inject_n_train_n_val(config_path: Path, n_train: int, n_val: int) -> None:
    from ruamel.yaml import YAML

    yaml_rt = YAML()
    yaml_rt.preserve_quotes = True
    yaml_rt.indent(mapping=2, sequence=4, offset=2)
    with config_path.open() as f:
        cfg = yaml_rt.load(f)
    cfg["n_train"] = n_train
    cfg["n_val"] = n_val
    with config_path.open("w") as f:
        yaml_rt.dump(cfg, f)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument(
        "--val-target-frames", type=int, default=VAL_TARGET_FRAMES,
        help=f"Target total validation frames (default {VAL_TARGET_FRAMES}).",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if DONE_FLAG.exists() and not args.force:
        print(f"[prepare_data] DONE_FLAG present at {DONE_FLAG} — skipping.")
        return 0
    t0 = time.time()

    # ---- Schema probe ----
    schema = probe_schema()
    SCHEMA_PROBE.write_text(json.dumps(schema, indent=2, default=str))
    print(f"[prepare_data] schema probe written to {SCHEMA_PROBE}")
    if schema["energy_forces_path"] is None:
        raise RuntimeError(
            f"Cannot extract energy/forces from Halo8 row. Schema: {schema}"
        )
    if schema["rxn_frame_path"] is None:
        raise RuntimeError(
            f"Cannot derive reaction_id/frame_idx from Halo8 row. Schema: {schema}\n"
            "Expected either top-level reaction_id+frame_idx attrs OR row.data['dand_id']."
        )
    ef_mode: str = schema["energy_forces_path"]
    rf_mode: str = schema["rxn_frame_path"]
    print(f"[prepare_data] energy/forces path: {ef_mode}  |  rxn/frame path: {rf_mode}")

    # ---- Reaction selection + train/val split ----
    f0_passed, sel_prov = select_reactions(args.seed)
    print(f"[prepare_data] selected {len(f0_passed)} reactions ({sel_prov['source']})")
    rxn_list = sorted(f0_passed)
    random.Random(args.seed).shuffle(rxn_list)
    n_train_rxns = int((1 - args.val_frac) * len(rxn_list))
    train_rxns = set(rxn_list[:n_train_rxns])
    val_rxns = set(rxn_list[n_train_rxns:])
    print(f"[prepare_data] split: {len(train_rxns)} train / {len(val_rxns)} val reactions")
    (DATA / "train_rxn_ids.json").write_text(json.dumps(sorted(train_rxns)))
    (DATA / "val_rxn_ids.json").write_text(json.dumps(sorted(val_rxns)))

    # ---- First sweep: collect frame_idx list per val reaction ----
    # Review §2.3: skip non-f0_passed rows up front so we don't pay the
    # parse cost for ~13M rows that won't be used.
    print("[prepare_data] sweep 1/2: collecting val reaction frame_idx lists...")
    val_frame_indices: dict[str, list[int]] = defaultdict(list)
    n_pairs_v1 = 0
    for shard in HALO8_SHARDS:
        if not shard.exists():
            continue
        with connect(str(shard)) as db:
            for row in db.select():
                try:
                    rid, fidx = derive_rxn_and_frame(row, rf_mode)
                except Exception:
                    continue
                if rid not in f0_passed:
                    continue
                if rid in val_rxns:
                    val_frame_indices[rid].append(fidx)
                    n_pairs_v1 += 1
        print(f"  {shard.name}: cumulative val frames discovered = {n_pairs_v1}")

    # ---- linspace endpoint-preserving stride per reaction ----
    val_frames_per_rxn = max(1, args.val_target_frames // max(1, len(val_rxns)))
    print(f"[prepare_data] target frames/val_rxn = {val_frames_per_rxn} "
          f"(target_total={args.val_target_frames}, n_val_rxns={len(val_rxns)})")
    val_keep_set: dict[str, set[int]] = {}
    for rxn_id, frames in val_frame_indices.items():
        frames.sort()
        total = len(frames)
        n_keep = min(total, val_frames_per_rxn)
        positions = np.linspace(0, total - 1, n_keep, dtype=int)
        positions = np.unique(positions)
        val_keep_set[rxn_id] = {frames[p] for p in positions}

    all_kept = [i for s in val_keep_set.values() for i in s]
    all_pool = [i for lst in val_frame_indices.values() for i in lst]
    print(f"[prepare_data] val pool total:   {len(all_pool)}")
    print(f"[prepare_data] val after select: {len(all_kept)}")
    if all_kept and all_pool:
        print(
            f"[prepare_data] frame_idx stats — min={min(all_kept)}, max={max(all_kept)}, "
            f"median={np.median(all_kept):.0f}, p99={np.percentile(all_kept, 99):.0f} "
            f"(pool_max={max(all_pool)})"
        )

    VAL_DIST.write_text(
        "\n".join([
            f"n_val_reactions: {len(val_rxns)}",
            f"target_frames_per_reaction: {val_frames_per_rxn}",
            f"actual_total_val_frames: {len(all_kept)}",
            "sampling_method: numpy_linspace_endpoint_preserving",
            f"frame_idx_min: {min(all_kept) if all_kept else 'NA'}",
            f"frame_idx_p1: {np.percentile(all_kept, 1):.0f}" if all_kept else "frame_idx_p1: NA",
            f"frame_idx_p25: {np.percentile(all_kept, 25):.0f}" if all_kept else "frame_idx_p25: NA",
            f"frame_idx_median: {np.median(all_kept):.0f}" if all_kept else "frame_idx_median: NA",
            f"frame_idx_p75: {np.percentile(all_kept, 75):.0f}" if all_kept else "frame_idx_p75: NA",
            f"frame_idx_p99: {np.percentile(all_kept, 99):.0f}" if all_kept else "frame_idx_p99: NA",
            f"frame_idx_max: {max(all_kept) if all_kept else 'NA'}",
            f"pool_max_for_reference: {max(all_pool) if all_pool else 'NA'}",
            "",
        ])
    )

    # ---- Second sweep: STREAMING extxyz write ----
    TRAIN_PATH.unlink(missing_ok=True)
    VAL_PATH.unlink(missing_ok=True)
    train_buf: list = []
    val_buf: list = []
    n_train_frames = n_val_frames = n_skip = 0

    def flush(buf, path: Path) -> None:
        if buf:
            write(str(path), buf, format="extxyz", append=True)
            buf.clear()

    print("[prepare_data] sweep 2/2: writing extxyz (streaming)...")
    for shard in HALO8_SHARDS:
        if not shard.exists():
            continue
        with connect(str(shard)) as db:
            for row in db.select():
                try:
                    rid, fidx = derive_rxn_and_frame(row, rf_mode)
                except Exception:
                    n_skip += 1
                    continue
                if rid not in f0_passed:
                    continue

                # Review §2.4: schema probe ran on the first row only; an
                # anomalous row mid-stream would otherwise crash the whole
                # sweep. Tolerate a few skips; abort if it exceeds a
                # threshold (likely a real schema regression).
                try:
                    atoms, energy, forces = extract_data(row, ef_mode)
                except Exception as e:  # noqa: BLE001
                    n_skip += 1
                    if n_skip <= 10:
                        print(f"  [skip] row id={getattr(row, 'id', '?')}: {e}")
                    if n_skip > 1000:
                        raise RuntimeError(
                            f"extract_data failed for >1000 rows — schema regression?"
                        ) from e
                    continue
                atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
                atoms.info["reaction_id"] = rid
                atoms.info["frame_idx"] = fidx

                if rid in train_rxns:
                    train_buf.append(atoms)
                    n_train_frames += 1
                    if len(train_buf) >= CHUNK:
                        flush(train_buf, TRAIN_PATH)
                elif rid in val_rxns:
                    if fidx in val_keep_set.get(rid, set()):
                        val_buf.append(atoms)
                        n_val_frames += 1
                        if len(val_buf) >= CHUNK:
                            flush(val_buf, VAL_PATH)
        flush(train_buf, TRAIN_PATH)
        flush(val_buf, VAL_PATH)
        print(f"  {shard.name}: cum train={n_train_frames}, val={n_val_frames}, "
              f"elapsed={time.time()-t0:.0f}s")
    flush(train_buf, TRAIN_PATH)
    flush(val_buf, VAL_PATH)

    print(f"[prepare_data] frames written: train={n_train_frames}, val={n_val_frames}, "
          f"skipped_rows={n_skip}")
    if n_train_frames == 0:
        raise RuntimeError("zero training frames written")

    # ---- Inject n_train / n_val into config ----
    config_path = PROJECT / "configs" / "halo8_nequip_v1.yaml"
    inject_n_train_n_val(config_path, n_train_frames, n_val_frames)
    print(f"[prepare_data] injected n_train={n_train_frames} / n_val={n_val_frames} into {config_path}")

    # ---- Hash + sentinel ----
    print("[prepare_data] computing SHA-256 hashes...")
    train_h = file_hash(TRAIN_PATH)
    val_h = file_hash(VAL_PATH)
    (DATA / "dataset_hash.txt").write_text(f"train: {train_h}\nval:   {val_h}\n")

    (DATA / "used_reaction_ids.json").write_text(json.dumps({
        "seed": args.seed,
        "val_frac": args.val_frac,
        "n_train_reactions": len(train_rxns),
        "n_val_reactions": len(val_rxns),
        "target_n_reactions": TARGET_N_RXNS,
        "selection_provenance": sel_prov,
    }, indent=2))

    DONE_FLAG.write_text(json.dumps({
        "spec_version": SPEC_VERSION,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_train_frames": n_train_frames,
        "n_val_frames": n_val_frames,
        "n_train_reactions": len(train_rxns),
        "n_val_reactions": len(val_rxns),
        "val_frames_per_rxn_target": val_frames_per_rxn,
        "val_sampling_method": "numpy_linspace_endpoint_preserving",
        "val_frame_idx_median": float(np.median(all_kept)) if all_kept else None,
        "val_frame_idx_p99": float(np.percentile(all_kept, 99)) if all_kept else None,
        "split_seed": args.seed,
        "elapsed_seconds": time.time() - t0,
        "schema_paths": {
            "energy_forces": ef_mode,
            "rxn_frame": rf_mode,
        },
    }, indent=2))
    print(f"[prepare_data] DONE in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
