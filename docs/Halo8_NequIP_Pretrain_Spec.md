# Halo8 NequIP Pre-training — Operational Specification

> **Hard rule 1**: Training must never lose more than 1 epoch of progress to any interruption (timeout, OOM, node failure).
> **Hard rule 2**: Jobs must auto-resubmit on timeout/crash until training reaches stop criterion or `MAX_RESUBMIT` is hit.
> **Hard rule 3**: All hyperparameter, data, and code versions must be reproducibly logged.

(Full spec is the document supplied by the user — checked into this repo as
`docs/Halo8_NequIP_Pretrain_Spec.md` for reference. Implementation details
live in `scripts/`, `configs/`, and this README.)

For brevity the full body is not duplicated here. The implementation pinned
to this spec is:

| Spec section | Implementation |
|---|---|
| §0 Working dir layout | `output/halo8_nequip_v1/` tree, created lazily |
| §1 Pre-flight checks | `scripts/preflight.py` |
| §2 Environment | conda env `reactot` with `nequip 0.6.2` |
| §3 Data preparation | `scripts/prepare_data.py` (idempotent) |
| §4 NequIP config | `configs/halo8_nequip_v1.yaml` |
| §5 Checkpointing | `save_checkpoint_freq: 1`, `n_save_keep: 3` |
| §6 Auto-resubmit | `scripts/submit_pretrain.sh`, SIGUSR1 @ T-10min |
| §7 Monitoring | `docs/SERVER_ACCESS_GUIDE.md` |
| §8 Stopping criteria | early stop config + manual `training_complete.flag` |
| §9 Failure modes | doc'd in `SERVER_ACCESS_GUIDE.md` |
| §10 Output validation | (to be added once a run completes) |
