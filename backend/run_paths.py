"""
Allocate a new numbered run directory under ``outs/``.

Default layout: ``outs/run_0001/``, ``outs/run_0002/``, … (zero-padded width 4).
"""
from __future__ import annotations

import os
import re
from typing import Optional


def next_run_index_dir(outs_root: str = "outs", *, width: int = 4) -> str:
    """
    Return an absolute path ``<outs_root>/run_<nnnn>/`` where *nnnn* is one greater than
    the highest existing ``run_<digits>`` directory under *outs_root*. Creates *outs_root*
    if missing.
    """
    outs_root = os.path.abspath(outs_root)
    os.makedirs(outs_root, exist_ok=True)
    pat = re.compile(r"^run_(\d+)$")
    max_n = 0
    try:
        for name in os.listdir(outs_root):
            m = pat.match(name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    except OSError:
        pass
    n = max_n + 1
    return os.path.join(outs_root, f"run_{n:0{width}d}")


def should_auto_allocate_out_dir(out_dir: Optional[str]) -> bool:
    s = (out_dir or "").strip()
    return s == "" or s.lower() == "auto"
