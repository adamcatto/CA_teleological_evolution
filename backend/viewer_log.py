"""
Write JSON artifacts under ``<out_dir>/viewer/`` for the live web visualizer.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from .rule import Rule
from .vocabulary import NeuralCellularAutomatonVocabulary, VOCABULARY_COLORS

_DEFAULT_IGNORE_RGB = (40, 40, 40)


def vocabulary_meta() -> List[Dict[str, Any]]:
    out = []
    for item in NeuralCellularAutomatonVocabulary:
        rgb = VOCABULARY_COLORS.get(int(item), _DEFAULT_IGNORE_RGB)
        out.append({"id": int(item), "name": item.name, "rgb": list(rgb)})
    return out


def _json_safe_grid(arr) -> List[List[int]]:
    if hasattr(arr, "astype"):
        return arr.astype(int).tolist()
    return [[int(x) for x in row] for row in arr]


def frames_from_automaton(automaton) -> List[List[List[int]]]:
    """Stacked NeuralXOR grid history as nested lists (token ids)."""
    hist = list(automaton.history)
    if not hist:
        g = automaton.grid.grid
        return [_json_safe_grid(g)]
    return [_json_safe_grid(a) for a in hist]


def _rules_to_json(rules: Optional[Sequence[Rule]]) -> List[Dict[str, Any]]:
    if not rules:
        return []
    out = []
    for r in rules:
        out.append(
            {
                "name": getattr(r, "name", ""),
                "input": r.input_grid.astype(int).tolist(),
                "output": r.output_grid.astype(int).tolist(),
            }
        )
    return out


def build_episode_dict(
    automaton,
    rules: Optional[Sequence[Rule]],
    reward: float,
    solved: bool,
    global_step: int,
    phase: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Serializable episode: CA frames, rules, metrics (shared by viewer + run archive)."""
    if rules is None:
        rules = list(automaton.rules)
    frames = frames_from_automaton(automaton)
    rules_j = _rules_to_json(rules)
    ep: Dict[str, Any] = {
        "id": int(global_step),
        "global_step": int(global_step),
        "phase": phase,
        "reward": float(reward),
        "solved": bool(solved),
        "n_frames": len(frames),
        "frames": frames,
        "rules": rules_j,
    }
    # Always attach ``extra`` (older code used ``if extra:`` which **omitted** the key for ``{}``).
    ep["extra"] = extra if extra is not None else {}
    return ep


@dataclass
class ViewerLogConfig:
    pretrain_window: int = 64
    max_episode_files: int = 10000
    pretrain_stream_interval_s: float = 1.5


class ViewerLogger:
    """
    Called from training: logs each completed episode (grid trajectory + rules) and
    a rolling pretrain window for "best reward in batch" style display.
    """

    def __init__(
        self,
        out_dir: str,
        *,
        pretrain_window: int = 64,
        max_episode_files: int = 10000,
        session_config: Optional[Dict[str, Any]] = None,
    ):
        self.out_dir = os.path.abspath(out_dir)
        self.vdir = os.path.join(self.out_dir, "viewer")
        self.epdir = os.path.join(self.vdir, "episodes")
        os.makedirs(self.epdir, exist_ok=True)
        self._pretrain_window = max(1, int(pretrain_window))
        self._pretrain_deque: Deque[Tuple[float, List[List[int]]]] = deque(maxlen=self._pretrain_window)
        self._max_episode_files = max(1, int(max_episode_files))
        self._episode_seq = 0
        self._config = session_config
        self._write_meta()

    def _write_meta(self) -> None:
        meta = {
            "out_dir": self.out_dir,
            "created": time.time(),
            "vocabulary": [t["name"] for t in vocabulary_meta()],
            "max_episode_files": self._max_episode_files,
        }
        if self._config is not None:
            meta["session_config"] = self._config
        with open(os.path.join(self.vdir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def set_session_config(self, config: Optional[Dict[str, Any]]) -> None:
        self._config = config
        self._write_meta()

    def _atomic_write(self, path: str, data: Any) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
        os.replace(tmp, path)

    def _append_manifest(self, entry: Dict[str, Any]) -> None:
        man_path = os.path.join(self.vdir, "episodes_manifest.json")
        try:
            with open(man_path, encoding="utf-8") as f:
                man = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError):
            man = {"items": []}
        items: List[Dict[str, Any]] = man.get("items") or []
        items.append(entry)
        man["items"] = items[-self._max_episode_files :]
        self._atomic_write(man_path, man)

    def log_episode(
        self,
        *,
        global_step: int,
        phase: str,
        automaton,
        reward: float,
        solved: bool,
        rules: Optional[Sequence[Rule]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record one REINFORCE episode: CA trajectory + (subset of) rules."""
        self._episode_seq += 1
        ep = build_episode_dict(
            automaton, rules, reward, solved, global_step, phase, extra
        )
        frames = ep.get("frames") or []
        if phase == "pretrain_grid" and frames:
            self._pretrain_deque.append((float(reward), frames[0]))

        slot = 1 + ((self._episode_seq - 1) % self._max_episode_files)
        ep_path = os.path.join(self.epdir, f"{slot:06d}.json")
        self._atomic_write(ep_path, ep)
        ep_file = f"{slot:06d}.json"
        self._append_manifest(
            {
                "id": global_step,
                "global_step": global_step,
                "phase": phase,
                "reward": float(reward),
                "solved": bool(solved),
                "n_frames": len(frames),
                "file_slot": slot,
                "file": ep_file,
            }
        )
        self._flush_stream(ep, phase=phase, episode_file=ep_file)

    def _pretrain_best(self) -> Optional[Dict[str, Any]]:
        if not self._pretrain_deque:
            return None
        best_reward, best_grid = max(self._pretrain_deque, key=lambda t: t[0])
        return {"reward": best_reward, "grid": best_grid, "window_len": len(self._pretrain_deque)}

    def _flush_stream(
        self,
        full_episode: Dict[str, Any],
        phase: str,
        episode_file: str,
    ) -> None:
        """
        `stream.json` stays small: no full `frames` array (read `viewer/episodes/<file>` for that).
        Includes `last_frame` for a quick first paint and pretrain best grid.
        """
        frames = full_episode.get("frames") or []
        last_frame = frames[-1] if frames else None
        first_frame = frames[0] if frames else None
        rules = full_episode.get("rules") or []
        rules_preview = rules[:24]
        last_sample = self._pretrain_deque[-1] if self._pretrain_deque else None
        lite: Dict[str, Any] = {
            "id": full_episode.get("id"),
            "global_step": full_episode.get("global_step"),
            "phase": full_episode.get("phase"),
            "reward": full_episode.get("reward"),
            "solved": full_episode.get("solved"),
            "n_frames": len(frames),
            "file": episode_file,
            "rules": rules_preview,
            "n_rules": len(rules),
            "extra": full_episode.get("extra"),
        }
        if first_frame is not None:
            lite["first_frame"] = first_frame
        if last_frame is not None:
            lite["last_frame"] = last_frame
        ex = full_episode.get("extra") or {}
        data = {
            "version": 1,
            "out_dir": self.out_dir,
            "updated_at": time.time(),
            "global_step": full_episode.get("global_step"),
            "phase": phase,
            # Redundant with ``latest_episode.extra`` so the web UI can read one stable field
            # (and stage **0** is not confused with "missing" in JSON).
            "curriculum_stage": ex.get("curriculum_stage"),
            "latest_episode": lite,
            "pretrain": {
                "window_size": self._pretrain_window,
                "last_sample": (
                    {
                        "reward": last_sample[0],
                        "grid": last_sample[1],
                    }
                    if last_sample
                    else None
                ),
                "best_in_window": self._pretrain_best(),
            },
        }
        self._atomic_write(os.path.join(self.vdir, "stream.json"), data)


def make_train_extra_from_info(info: dict) -> Dict[str, Any]:
    d = {
        "n_rules_total": info.get("n_rules_total", 0),
        "n_rules_active": info.get("n_rules_active", 0),
        "n_violations": info.get("n_violations", 0),
    }
    if "from_policy" in info:
        d["from_policy"] = bool(info["from_policy"])
    if "grid_switching_penalty" in info and info["grid_switching_penalty"] is not None:
        d["grid_switching_penalty"] = float(info["grid_switching_penalty"])
    for k in ("grid_motion_output_bonus", "grid_motion_input_bonus"):
        if k in info and info[k] is not None:
            d[k] = float(info[k])
    if "curriculum_stage" in info and info["curriculum_stage"] is not None:
        d["curriculum_stage"] = int(info["curriculum_stage"])
    return d
