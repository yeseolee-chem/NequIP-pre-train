#!/usr/bin/env python
"""NequIP entry-point wrapper (Spec v5 §6).

Why: we must flip TF32 ON in ``torch.backends`` BEFORE any ``nequip`` /
``e3nn`` import, otherwise those modules cache the unmodified flag.

Why the ``if __name__ == "__main__":`` guard is critical: NequIP's
``ASEDataset.get_data()`` parallelises raw-extxyz parsing via a
``multiprocessing.get_context("forkserver").Pool``. The forkserver
worker boots a fresh Python and reruns this file via
``runpy.run_path(..., run_name="__mp_main__")``. Without the guard,
each worker re-invokes ``nequip.scripts.train.main`` → which spawns
its own ``Pool`` → which spawns more workers, an infinite fork
recursion that silently fills ``training.log`` with thousands of
identical tracebacks while no real training happens. The
``__mp_main__`` run_name means the guarded block does not execute in
workers — only the TF32 flag flip and the function import do.
"""
from __future__ import annotations

import os
import sys

import torch

if os.environ.get("NEQUIP_ENABLE_TF32", "0") == "1":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("[launcher] TF32 enabled")

from nequip.scripts.train import main  # noqa: E402

if __name__ == "__main__":
    sys.argv[0] = "nequip-train"
    main()
