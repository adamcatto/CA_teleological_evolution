"""
Session configuration: training, model, grid, and reward parameters with JSON
serialization so runs are reproducible and `train.py` can be driven from a file.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from .rewards import RewardConfig
from .rewards.potential import PotentialWeights

# Stages: `full` = joint REINFORCE (see `mode`). `pretrain_grid` = grid init only, init/shaping from Φ.
STAGES = ("full", "pretrain_grid")


@dataclass
class TrainingSettings:
    steps: int = 1000
    pretrain_grid_steps: int = 0
    lr: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 0
    log_every: int = 10
    ckpt_every: int = 0
    out_dir: str = ""  # empty / "auto" -> allocate outs/run_NNNN/ at train time
    load_checkpoint: Optional[str] = None


@dataclass
class SessionConfig:
    """
    All parameters for one training run. The `reward` field mirrors `RewardConfig`
    (including nested `potential_weights`); `training` mirrors `TrainingSettings`.
    """
    version: int = 1
    stage: str = "full"

    # Task / automaton
    mode: str = "rules_only"
    # Structural / reward: 1 = only HIDDEN_NEURON_1; 4 = HIDDEN_NEURON_1..4.
    num_hidden_neuron_types: int = 1
    grid_size: int = 16
    # Grid UNet (``GridGenerationUNet`` / ``GridGenerationUNetMultiHead``) — defaults keep params small;
    # override via ``grid_generator_kwargs``; unknown keys for other grid generators are not passed.
    grid_latent_dim: int = 32
    grid_start_spatial: int = 2
    grid_base_channels: int = 16
    max_rules: int = 64
    num_ca_steps: int = 16
    pretrain_num_ca_steps: int = 0
    pretrain_policy_mix: float = 1.0
    # Hamming-fraction × coef subtracted from reward when base grid changes vs previous episode
    grid_switching_coef: float = 0.08

    # Policy (sampling / REINFORCE) — τ is always clamped to [gumbel_tau_min, gumbel_tau_max]
    gumbel_tau: float = 0.5
    # First `gumbel_tau_warmup_steps`: linear ramp from `gumbel_tau_start` → `gumbel_tau`, then decay.
    gumbel_tau_start: float = 0.08
    gumbel_tau_warmup_steps: int = 500
    gumbel_tau_min: float = 0.05
    gumbel_tau_max: float = 0.5
    gumbel_tau_decay: float = 0.999
    entropy_coef: float = 0.01
    baseline_momentum: float = 0.95
    ema_decay: float = 0.999
    use_action_masking: bool = True

    # Generator class ids: "unet"|"unet_multihead"|"mlp" (grid), "transformer"|"mlp" (rules)
    grid_generator: str = "unet"
    rule_generator: str = "transformer"
    # Optional: per-group additive logits in grid sampling (I/O, synapse, hidden, empty, rest).
    # None = use `build_staged_logit_add` defaults in the policy.
    grid_staged_logit_priors: Optional[Dict[str, float]] = None
    grid_generator_kwargs: Dict[str, Any] = field(default_factory=dict)
    rule_generator_kwargs: Dict[str, Any] = field(default_factory=dict)

    reward: Optional[Dict[str, Any]] = None
    training: Optional[Dict[str, Any]] = None

    # Curriculum (grid pretrain / full): ``global_step`` → stage; see ``backend/curriculum.py``
    curriculum_enabled: bool = True
    # When ``curriculum_enabled``, grid pretrain runs for sum(...) steps (one update per
    # budget unit); set ``curriculum_enabled`` false to honor ``pretrain_grid_steps`` / ``steps`` only.
    curriculum_stage_steps: List[int] = field(
        default_factory=lambda: [1000, 1000, 1000, 1000, 1000, 1000, 1000]
    )
    curriculum_use_vocab_mask: bool = True
    # When True, re-run Gumbel τ warmup after each stage change (policy explores again).
    curriculum_reset_gumbel_tau: bool = True

    def __post_init__(self):
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {self.stage!r}")
        if self.gumbel_tau_max < self.gumbel_tau_min:
            raise ValueError("gumbel_tau_max must be >= gumbel_tau_min")
        lo, hi = self.gumbel_tau_min, self.gumbel_tau_max
        self.gumbel_tau = min(max(self.gumbel_tau, lo), hi)
        self.gumbel_tau_start = min(max(self.gumbel_tau_start, lo), hi)
        nht = int(self.num_hidden_neuron_types)
        if nht not in (1, 4):
            raise ValueError("num_hidden_neuron_types must be 1 or 4")
        self.num_hidden_neuron_types = nht
        gs, ss = int(self.grid_size), int(self.grid_start_spatial)
        if ss <= 0 or gs % ss != 0:
            raise ValueError(f"grid_size {gs} must be divisible by grid_start_spatial {ss}")
        up = gs // ss
        if up & (up - 1) != 0:
            raise ValueError(
                f"grid_size / grid_start_spatial must be a power of 2 for the conv UNet (got {up})"
            )


def _reward_to_dict(c: RewardConfig) -> Dict[str, Any]:
    d = asdict(c)
    return d


def _reward_from_dict(d: Optional[Dict[str, Any]]) -> RewardConfig:
    if not d:
        return RewardConfig()
    pw_d = d.get("potential_weights") or {}
    default_pw = PotentialWeights()
    potential = PotentialWeights(
        **{f.name: pw_d.get(f.name, getattr(default_pw, f.name)) for f in fields(PotentialWeights)}
    )
    known_reward = {f.name for f in fields(RewardConfig)}
    rest = {k: v for k, v in d.items() if k in known_reward and k != "potential_weights"}
    return RewardConfig(potential_weights=potential, **rest)


def _training_from_dict(d: Optional[Dict[str, Any]]) -> TrainingSettings:
    if not d:
        return TrainingSettings()
    known = {f.name for f in fields(TrainingSettings)}
    return TrainingSettings(**{k: v for k, v in d.items() if k in known})


def session_config_to_dict(cfg: SessionConfig) -> Dict[str, Any]:
    d = asdict(cfg)
    if d.get("reward") is None:
        d["reward"] = _reward_to_dict(get_reward_config(cfg))
    if d.get("training") is None:
        d["training"] = asdict(get_training_settings(cfg))
    return d


def session_config_from_dict(d: Dict[str, Any]) -> SessionConfig:
    known = {f.name for f in fields(SessionConfig)}
    core = {k: v for k, v in d.items() if k in known and k not in ("reward", "training")}
    reward = d.get("reward")
    training = d.get("training")
    return SessionConfig(**core, reward=reward if reward is not None else None, training=training if training is not None else None)


def save_session_config(cfg: SessionConfig, path: Union[str, Path], indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session_config_to_dict(cfg), f, indent=indent, sort_keys=True)


def load_session_config(path: Union[str, Path]) -> SessionConfig:
    with open(path, encoding="utf-8") as f:
        return session_config_from_dict(json.load(f))


def get_reward_config(cfg: SessionConfig) -> RewardConfig:
    return _reward_from_dict(cfg.reward)


def get_training_settings(cfg: SessionConfig) -> TrainingSettings:
    return _training_from_dict(cfg.training)


# --- resolve generator classes / kwargs for CellularAutomatonGenerationPolicyNetwork ---

from .network import (  # noqa: E402
    GridGenerationNetwork,
    GridGenerationUNet,
    GridGenerationUNetMultiHead,
    RuleGenerationDiffusionNetwork,
    RuleGenerationTransformer,
)

_GRID_REGISTRY: Dict[str, Type] = {
    "mlp": GridGenerationNetwork,
    "unet": GridGenerationUNet,
    "unet_mh": GridGenerationUNetMultiHead,
    "unet_multih": GridGenerationUNetMultiHead,
    "unet_multihead": GridGenerationUNetMultiHead,
    "GridGenerationNetwork": GridGenerationNetwork,
    "GridGenerationUNet": GridGenerationUNet,
    "GridGenerationUNetMultiHead": GridGenerationUNetMultiHead,
}

_RULE_REGISTRY: Dict[str, Type] = {
    "mlp": RuleGenerationDiffusionNetwork,
    "transformer": RuleGenerationTransformer,
    "RuleGenerationDiffusionNetwork": RuleGenerationDiffusionNetwork,
    "RuleGenerationTransformer": RuleGenerationTransformer,
}


def resolve_generators(cfg: SessionConfig) -> tuple:
    gname = (cfg.grid_generator or "unet").lower()
    rname = (cfg.rule_generator or "transformer").lower()
    if gname in ("grid_generation_network",):
        gname = "mlp"
    if gname == "grid_generation_unet":
        gname = "unet"
    if rname in ("diffusion", "rule_generation_diffusion"):
        rname = "mlp"
    if rname in ("trf", "rule_generation_transformer"):
        rname = "transformer"
    gcls = _GRID_REGISTRY.get(gname) or _GRID_REGISTRY.get(cfg.grid_generator)
    rcls = _RULE_REGISTRY.get(rname) or _RULE_REGISTRY.get(cfg.rule_generator)
    if gcls is None:
        raise ValueError(f"Unknown grid generator: {cfg.grid_generator!r}")
    if rcls is None:
        raise ValueError(f"Unknown rule generator: {cfg.rule_generator!r}")
    gkw = dict(cfg.grid_generator_kwargs or {})
    rkw = dict(cfg.rule_generator_kwargs or {})
    return gcls, rcls, gkw, rkw
