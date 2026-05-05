"""
Optional Weights & Biases integration: one run, config + code snapshot, per-step scalars.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .code_snapshot import _git_info_text, write_code_snapshot_zip


def _safe_config(d: Any) -> dict:
    """W&B ``config`` must be JSON-friendly."""
    try:
        json.dumps(d)
        return d if isinstance(d, dict) else {"value": d}
    except (TypeError, ValueError):
        s = str(d)[: 50_000]
        return {"session_config_serialized": s}


def init_wandb_run(
    *,
    out_dir: str,
    repo_root: str,
    project: str,
    name: Optional[str] = None,
    entity: Optional[str] = None,
    group: Optional[str] = None,
    config_dict: Optional[dict] = None,
    include_dot_git: bool = False,
) -> Any:
    """
    ``pip install wandb`` and set ``WANDB_API_KEY`` (or ``wandb login``).

    Writes ``<out_dir>/git_info.txt`` and ``<out_dir>/code_snapshot.zip`` (excludes
    top-level ``outs/``; see ``code_snapshot.write_code_snapshot``), uploads them to
    the run, and returns the W&B run object (for ``log_step`` / ``finish``).
    """
    try:
        import wandb
    except ImportError as e:
        raise RuntimeError("wandb is not installed. Install with: pip install wandb") from e

    out = os.path.abspath(out_dir)
    os.makedirs(out, exist_ok=True)
    root = Path(repo_root).resolve()

    gi_path = os.path.join(out, "git_info.txt")
    with open(gi_path, "w", encoding="utf-8") as f:
        f.write(_git_info_text(root))
    zpath = write_code_snapshot_zip(
        root,
        os.path.join(out, "code_snapshot.zip"),
        include_dot_git=include_dot_git,
    )

    cfg = _safe_config(config_dict) if config_dict is not None else {}
    run = wandb.init(
        project=project,
        entity=entity or None,
        name=name or os.path.basename(out),
        group=group or None,
        config=cfg,
    )
    try:
        wandb.save(gi_path)
    except Exception:
        pass
    try:
        wandb.save(zpath)
    except Exception:
        pass
    return run


def wandb_log(rec: Dict[str, Any], run: Any) -> None:
    if run is None:
        return
    import wandb

    step = int(rec.get("global_step", 0))
    payload: Dict[str, Any] = {
        "reward": float(rec.get("reward", 0.0)),
        "loss": float(rec.get("loss", 0.0)),
    }
    for k in ("baseline", "entropy", "gumbel_tau", "grad_norm", "solved", "n_rules_total", "n_rules_active", "n_violations"):
        if k in rec:
            v = rec[k]
            if isinstance(v, bool):
                payload[k] = int(v)
            else:
                payload[k] = float(v) if isinstance(v, (int, float)) else v
    if "curriculum_stage" in rec and rec["curriculum_stage"] is not None:
        payload["curriculum_stage"] = int(rec["curriculum_stage"])
    if "phase" in rec:
        payload["phase"] = str(rec["phase"])
    if "from_policy" in rec:
        payload["from_policy"] = int(bool(rec["from_policy"]))
    wandb.log(payload, step=step)


def wandb_finish(run: Any) -> None:
    if run is None:
        return
    import wandb

    try:
        wandb.finish()
    except Exception:
        pass
