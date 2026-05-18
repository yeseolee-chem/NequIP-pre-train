#!/usr/bin/env python
"""NequIP entry-point wrapper (Spec v5 §6).

Why: we must flip TF32 ON in ``torch.backends`` BEFORE any ``nequip`` /
``e3nn`` import, otherwise those modules cache the unmodified flag.
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

sys.argv[0] = "nequip-train"
main()
