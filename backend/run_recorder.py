"""
Per-run training log: scalar metrics (``metrics.jsonl``) and full episode archives
(``episodes_all/``) with grid trajectories, rules, and REINFORCE diagnostics.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Optional

# W&B run handle (returned from ``wandb.init``); only typed as ``Any`` to avoid import.

from .viewer_log import build_episode_dict, make_train_extra_from_info


def _atomic_write(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    os.replace(tmp, path)


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    line = json.dumps(obj, separators=(",", ":")) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


class RunRecorder:
    def __init__(
        self,
        out_dir: str,
        *,
        save_episode_json: bool = True,
        wandb_run: Optional[Any] = None,
    ):
        self.out_dir = os.path.abspath(out_dir)
        self._save_episodes = bool(save_episode_json)
        self._wandb_run = wandb_run
        self.metrics_path = os.path.join(self.out_dir, "metrics.jsonl")
        self.episodes_all = os.path.join(self.out_dir, "episodes_all")
        if self._save_episodes:
            os.makedirs(self.episodes_all, exist_ok=True)
        self._t0 = time.time()
        _atomic_write(
            os.path.join(self.out_dir, "run_started.json"),
            {
                "created_wall_time": time.time(),
                "out_dir": self.out_dir,
                "argv": sys.argv,
            },
        )

    def log_step(
        self,
        *,
        global_step: int,
        step_in_phase: int,
        phase: str,
        info: dict,
    ) -> None:
        """
        *info* is the dict returned from ``pretrain_grid_step`` / ``train_step`` (includes
        automaton, reward, loss, etc.).
        """
        automaton = info["automaton"]
        extra = make_train_extra_from_info(info)
        rel_episode: Optional[str] = None
        if self._save_episodes:
            ep = build_episode_dict(
                automaton=automaton,
                rules=list(automaton.rules),
                reward=float(info["reward"]),
                solved=bool(info.get("solved", False)),
                global_step=global_step,
                phase=phase,
                extra=extra,
            )
            name = f"{global_step:08d}.json"
            path = os.path.join(self.episodes_all, name)
            _atomic_write(path, ep)
            rel_episode = os.path.join("episodes_all", name)

        rec: Dict[str, Any] = {
            "wall_time_s": round(time.time() - self._t0, 6),
            "ts": time.time(),
            "global_step": global_step,
            "step_in_phase": step_in_phase,
            "phase": phase,
            "reward": float(info["reward"]),
            "loss": float(info.get("loss", 0.0)),
            "baseline": float(info.get("baseline", 0.0)),
            "entropy": float(info.get("entropy", 0.0)),
            "gumbel_tau": float(info.get("gumbel_tau", 0.0)),
            "grad_norm": float(info.get("grad_norm", 0.0)),
            "solved": bool(info.get("solved", False)),
            "n_rules_total": int(info.get("n_rules_total", 0)),
            "n_rules_active": int(info.get("n_rules_active", 0)),
            "n_violations": int(info.get("n_violations", 0)),
        }
        if "from_policy" in info:
            rec["from_policy"] = bool(info["from_policy"])
        if "curriculum_stage" in info and info["curriculum_stage"] is not None:
            rec["curriculum_stage"] = int(info["curriculum_stage"])
        if rel_episode is not None:
            rec["episode_file"] = rel_episode
        _append_jsonl(self.metrics_path, rec)
        if self._wandb_run is not None:
            from .wandb_log import wandb_log as _wandb_log_step

            _wandb_log_step(rec, self._wandb_run)
