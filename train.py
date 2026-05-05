"""
Train the rule/grid generator on the NeuralXOR task.

  python train.py
  # If you omit --config, train.py will load ``./curriculum_session.json`` (cwd) or, if
  #  missing, ``<repo>/curriculum_session.json`` next to this script; otherwise it uses
  #  built-in ``SessionConfig`` defaults (uniform curriculum stages, etc.).
  python train.py --config session.json
  python train.py --stage pretrain_grid --pretrain-grid-steps 2000
  # With curriculum on, grid pretrain length defaults to sum(curriculum_stage_steps); the
  #  CLI pretrain count is only used as a fallback or when the curriculum budget sum is 0.
  python train.py --stage full --pretrain-grid-steps 5000   # pretrain, then full

Session parameters can live in a JSON file (see `backend/session_config.py` and
`default_session.json`). CLI flags override the config file.

By default, artifacts go to the next free folder ``outs/run_NNNN/`` (empty
``training.out_dir`` in config, or ``"auto"``). Override with ``--out-dir /path``.
Each run dir contains ``session_resolved.json``, ``metrics.jsonl``, ``episodes_all/``
(full grid/rule traces per step), ``run_started.json``, and checkpoints.
Optional: ``--wandb`` logs reward/loss each step to Weights & Biases and zips a code snapshot
(see ``--help``; ``pip install -r requirements-wandb.txt``).
"""
import argparse
import atexit
import collections
import os
import time
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Optional, Tuple

import torch

from backend.network import (
    CellularAutomatonGenerationPolicyNetwork,
    GridGenerationUNet,
    GridGenerationUNetMultiHead,
)
from backend.rewards import RewardConfig
from backend.session_config import (
    STAGES,
    SessionConfig,
    TrainingSettings,
    get_reward_config,
    get_training_settings,
    load_session_config,
    resolve_generators,
    save_session_config,
    session_config_from_dict,
    session_config_to_dict,
)
from backend.run_paths import next_run_index_dir, should_auto_allocate_out_dir
from backend.run_recorder import RunRecorder
from backend.viewer_log import ViewerLogger, make_train_extra_from_info
from backend.vocabulary import NeuralCellularAutomatonVocabulary as Voc

from backend.curriculum import apply_curriculum_step, total_curriculum_steps


def _merge_cli_into_config(cfg, args) -> "SessionConfig":
    """Apply argparse overrides to a `SessionConfig` (mutates a copy)."""
    from backend.session_config import SessionConfig

    t = get_training_settings(cfg)
    t_over: dict = {}
    if args.steps is not None:
        t_over["steps"] = args.steps
    if args.pretrain_grid_steps is not None:
        t_over["pretrain_grid_steps"] = args.pretrain_grid_steps
    if args.lr is not None:
        t_over["lr"] = args.lr
    if args.grad_clip is not None:
        t_over["grad_clip"] = args.grad_clip
    if args.seed is not None:
        t_over["seed"] = args.seed
    if args.log_every is not None:
        t_over["log_every"] = args.log_every
    if args.ckpt_every is not None:
        t_over["ckpt_every"] = args.ckpt_every
    if args.out_dir is not None:
        t_over["out_dir"] = args.out_dir
    if args.load_checkpoint is not None:
        t_over["load_checkpoint"] = args.load_checkpoint
    t_merged = replace(t, **t_over) if t_over else t

    core_over = {}
    if args.stage is not None:
        core_over["stage"] = args.stage
    if args.mode is not None:
        core_over["mode"] = args.mode
    if args.max_rules is not None:
        core_over["max_rules"] = args.max_rules
    if args.num_ca_steps is not None:
        core_over["num_ca_steps"] = args.num_ca_steps
    if args.pretrain_num_ca_steps is not None:
        core_over["pretrain_num_ca_steps"] = args.pretrain_num_ca_steps
    if args.pretrain_policy_mix is not None:
        core_over["pretrain_policy_mix"] = args.pretrain_policy_mix
    if args.grid_switching_coef is not None:
        core_over["grid_switching_coef"] = args.grid_switching_coef
    if args.grid_size is not None:
        core_over["grid_size"] = args.grid_size
    if args.grid_latent_dim is not None:
        core_over["grid_latent_dim"] = args.grid_latent_dim
    if args.grid_start_spatial is not None:
        core_over["grid_start_spatial"] = args.grid_start_spatial
    if args.grid_base_channels is not None:
        core_over["grid_base_channels"] = args.grid_base_channels
    if args.num_hidden_neuron_types is not None:
        core_over["num_hidden_neuron_types"] = args.num_hidden_neuron_types
    if args.gumbel_tau is not None:
        core_over["gumbel_tau"] = args.gumbel_tau
    if args.gumbel_tau_start is not None:
        core_over["gumbel_tau_start"] = args.gumbel_tau_start
    if args.gumbel_tau_warmup_steps is not None:
        core_over["gumbel_tau_warmup_steps"] = args.gumbel_tau_warmup_steps
    if args.gumbel_tau_min is not None:
        core_over["gumbel_tau_min"] = args.gumbel_tau_min
    if args.gumbel_tau_max is not None:
        core_over["gumbel_tau_max"] = args.gumbel_tau_max
    if args.gumbel_tau_decay is not None:
        core_over["gumbel_tau_decay"] = args.gumbel_tau_decay
    if args.entropy_coef is not None:
        core_over["entropy_coef"] = args.entropy_coef
    if args.baseline_momentum is not None:
        core_over["baseline_momentum"] = args.baseline_momentum
    if args.ema_decay is not None:
        core_over["ema_decay"] = args.ema_decay
    if args.grid_generator is not None:
        core_over["grid_generator"] = args.grid_generator
    if args.rule_generator is not None:
        core_over["rule_generator"] = args.rule_generator

    out = replace(cfg, **core_over) if core_over else cfg
    # training and reward live as optional dicts on SessionConfig
    obj = asdict(out)
    obj["training"] = {f.name: getattr(t_merged, f.name) for f in fields(TrainingSettings)}
    return session_config_from_dict(obj)


def _config_with_training(cfg, t: TrainingSettings):
    d = asdict(cfg)
    d["training"] = {f.name: getattr(t, f.name) for f in fields(TrainingSettings)}
    return session_config_from_dict(d)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Path to session JSON (see SessionConfig)")
    p.add_argument("--stage", choices=list(STAGES), default=None)
    p.add_argument(
        "--save-config",
        type=str,
        default=None,
        help="Optional path to write the resolved session JSON (default: <out_dir>/session_resolved.json)",
    )
    p.add_argument(
        "--steps", type=int, default=None, help="Main training steps (overrides config training.steps)"
    )
    p.add_argument(
        "--pretrain-grid-steps",
        type=int,
        default=None,
        help="Grid pretrain steps; with curriculum_enabled and positive stage budgets, train.py uses their sum instead",
    )
    p.add_argument("--mode", choices=["rules_only", "grid_and_rules"], default=None)
    p.add_argument("--max-rules", type=int, default=None)
    p.add_argument("--num-ca-steps", type=int, default=None)
    p.add_argument("--pretrain-num-ca-steps", type=int, default=None, help="CA steps during grid pretrain (0 = init reward only)")
    p.add_argument("--pretrain-policy-mix", type=float, default=None, help="P(policy) vs random in grid pretrain")
    p.add_argument(
        "--grid-switching-coef",
        type=float,
        default=None,
        help="Reward penalty (× Hamming fraction) for base grid change vs last episode; default 0.08",
    )
    p.add_argument("--grid-size", type=int, default=None)
    p.add_argument(
        "--grid-latent-dim",
        type=int,
        default=None,
        help="Grid UNet latent (default from SessionConfig, used for unet / unet_multihead)",
    )
    p.add_argument(
        "--grid-start-spatial",
        type=int,
        default=None,
        help="UNet start resolution (H/W must be start_spatial * 2^n; default 2)",
    )
    p.add_argument(
        "--grid-base-channels",
        type=int,
        default=None,
        help="UNet first trunk width (default 16)",
    )
    p.add_argument(
        "--num-hidden-neuron-types",
        type=int,
        default=None,
        choices=(1, 4),
        help="1 = HIDDEN_NEURON_1 only; 4 = four hidden neuron types (default: config or 1)",
    )
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--gumbel-tau", type=float, default=None, help="Target τ after warmup (then multiplicative decay)")
    p.add_argument(
        "--gumbel-tau-start",
        type=float,
        default=None,
        help="Initial τ (low = sharp sampling; linear ramp to --gumbel-tau over warmup steps)",
    )
    p.add_argument(
        "--gumbel-tau-warmup-steps",
        type=int,
        default=None,
        help="Linear τ ramp from gumbel_tau_start → gumbel_tau; 0 = disable warmup",
    )
    p.add_argument("--gumbel-tau-min", type=float, default=None, help="Lower bound for τ (default 0.05)")
    p.add_argument("--gumbel-tau-max", type=float, default=None, help="Upper bound for τ (default 0.5)")
    p.add_argument("--gumbel-tau-decay", type=float, default=None)
    p.add_argument("--entropy-coef", type=float, default=None)
    p.add_argument("--baseline-momentum", type=float, default=None)
    p.add_argument("--ema-decay", type=float, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Run directory (default: allocate outs/run_NNNN/). Pass a path to override.",
    )
    p.add_argument(
        "--no-episodes-all",
        action="store_true",
        help="Do not write per-step full JSON under episodes_all/ (metrics.jsonl is still written).",
    )
    p.add_argument("--ckpt-every", type=int, default=None)
    p.add_argument("--load", type=str, default=None, dest="load_checkpoint", help="Load policy state_dict (torch save)")
    p.add_argument("--grid-generator", type=str, default=None, choices=["unet", "mlp", "GridGenerationUNet", "GridGenerationNetwork"])
    p.add_argument("--rule-generator", type=str, default=None, choices=["transformer", "mlp", "RuleGenerationTransformer", "RuleGenerationDiffusionNetwork"])
    p.add_argument("--viewer", action="store_true", help="Write <out_dir>/viewer/ for the web UI (see viewer_server.py)")
    p.add_argument("--viewer-pretrain-window", type=int, default=64, help="Rolling window for best-reward pretrain display")
    p.add_argument(
        "--viewer-max-episodes",
        type=int,
        default=10000,
        help="On-disk ring buffer of episode JSON files in viewer/episodes/ (and manifest tail)",
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Log reward/loss (and other metrics) to Weights & Biases; requires pip install wandb",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default="ca-teleological-evolution",
        help="W&B project name (default: ca-teleological-evolution)",
    )
    p.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/team (optional)")
    p.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="W&B run name (default: run directory basename, e.g. run_0003)",
    )
    p.add_argument("--wandb-group", type=str, default=None, help="W&B run group (optional)")
    p.add_argument(
        "--wandb-snapshot-git",
        action="store_true",
        help="Include the full .git/ tree in code_snapshot.zip (large). Default: zip without .git, plus git_info.txt",
    )
    return p.parse_args()


def _default_config():
    from backend.session_config import SessionConfig

    return SessionConfig(
        training=asdict(TrainingSettings()),
        reward=asdict(RewardConfig()),
    )


def _load_base_session_config(args) -> Tuple[Optional[str], SessionConfig]:
    """
    Resolve which JSON to use when ``args.config`` is None: prefer ``curriculum_session.json``
    in the process cwd, then next to ``train.py``, so runs pick up the repo's curriculum
    file without having to pass ``--config`` every time.
    """
    if args.config:
        return args.config, load_session_config(args.config)
    cwd_file = Path.cwd() / "curriculum_session.json"
    if cwd_file.is_file():
        p = str(cwd_file.resolve())
        return p, load_session_config(p)
    here_file = Path(__file__).resolve().parent / "curriculum_session.json"
    if here_file.is_file():
        p = str(here_file.resolve())
        return p, load_session_config(p)
    return None, _default_config()


def make_policy(cfg, reward_config: RewardConfig):
    gcls, rcls, gkw, rkw = resolve_generators(cfg)
    gkw = dict(gkw)
    if gcls in (GridGenerationUNet, GridGenerationUNetMultiHead):
        gkw.setdefault("latent_dim", cfg.grid_latent_dim)
        gkw.setdefault("start_spatial", cfg.grid_start_spatial)
        gkw.setdefault("base_channels", cfg.grid_base_channels)
    t = get_training_settings(cfg)
    return CellularAutomatonGenerationPolicyNetwork(
        vocabulary=Voc,
        grid_size=(cfg.grid_size, cfg.grid_size),
        max_rules=cfg.max_rules,
        num_ca_steps=cfg.num_ca_steps,
        num_hidden_neuron_types=cfg.num_hidden_neuron_types,
        gumbel_tau=cfg.gumbel_tau,
        gumbel_tau_start=cfg.gumbel_tau_start,
        gumbel_tau_warmup_steps=cfg.gumbel_tau_warmup_steps,
        gumbel_tau_min=cfg.gumbel_tau_min,
        gumbel_tau_max=cfg.gumbel_tau_max,
        gumbel_tau_decay=cfg.gumbel_tau_decay,
        entropy_coef=cfg.entropy_coef,
        baseline_momentum=cfg.baseline_momentum,
        ema_decay=cfg.ema_decay,
        reward_config=reward_config,
        mode=cfg.mode,
        pretrain_policy_mix=cfg.pretrain_policy_mix,
        pretrain_num_ca_steps=cfg.pretrain_num_ca_steps,
        grid_switching_coef=cfg.grid_switching_coef,
        use_action_masking=cfg.use_action_masking,
        grid_staged_logit_priors=cfg.grid_staged_logit_priors,
        grid_generator_cls=gcls,
        rule_generator_cls=rcls,
        grid_generator_kwargs=gkw,
        rule_generator_kwargs=rkw,
    )


def _sync_curriculum(
    policy: CellularAutomatonGenerationPolicyNetwork,
    cfg,
    base_reward: RewardConfig,
    global_step: int,
    *,
    use_curriculum: bool = True,
) -> int:
    """
    Sets ``policy.reward_config`` and curriculum logit mask for ``global_step`` (1-based).
    If ``use_curriculum`` is False (e.g. ``full`` with ``mode=rules_only``), restores
    ``base_reward`` and an all-ones mask. Returns the curriculum stage or 0.
    """
    from backend.vocabulary import NeuralCellularAutomatonVocabulary as Voc

    dev = next(policy.parameters()).device
    v = int(max((int(t.value) for t in Voc), default=-1)) + 1
    if not use_curriculum or not getattr(cfg, "curriculum_enabled", False):
        policy.reward_config = base_reward
        policy.set_curriculum_logit_mask(torch.ones(v, dtype=torch.float32, device=dev))
        return 0
    rc, st, msk = apply_curriculum_step(global_step, cfg, base_reward)
    policy.reward_config = rc
    policy.set_curriculum_logit_mask(msk.to(dev))
    return st


def _resolve_pretrain_num_steps(t: TrainingSettings, cfg, *, label: str) -> int:
    """
    Grid pretrain step count. When ``curriculum_enabled`` and the per-stage budgets sum
    to a positive total, that sum is used so one pretrain run covers every curriculum stage.
    """
    requested = (t.pretrain_grid_steps or 0) or (t.steps or 0)
    if getattr(cfg, "curriculum_enabled", False):
        n_cur = total_curriculum_steps(cfg)
        if n_cur > 0:
            if requested > 0 and requested != n_cur:
                print(
                    f"[{label}] curriculum_enabled: running {n_cur} pretrain steps "
                    f"(sum of curriculum_stage_steps), not {requested}"
                )
            return n_cur
    if requested <= 0:
        raise ValueError(
            f"{label}: set training.pretrain_grid_steps (or --pretrain-grid-steps) or training.steps / --steps > 0"
        )
    return requested


def _log_viewer_episode(
    viewer: ViewerLogger, phase: str, info: dict, global_step: int
) -> None:
    viewer.log_episode(
        global_step=global_step,
        phase=phase,
        automaton=info["automaton"],
        reward=float(info["reward"]),
        solved=bool(info.get("solved", False)),
        extra=make_train_extra_from_info(info),
    )


def _run_pretrain(
    policy: CellularAutomatonGenerationPolicyNetwork,
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    t: TrainingSettings,
    label: str = "pretrain",
    viewer: Optional[ViewerLogger] = None,
    viewer_phase: str = "pretrain_grid",
    recorder: Optional[RunRecorder] = None,
    step_counter: dict = None,
    session_cfg: Optional[object] = None,
    base_reward: Optional[RewardConfig] = None,
):
    """Grid-generator-only REINFORCE; `optimizer` should cover `policy.grid_gen` (or a subset)."""
    if step_counter is None:
        step_counter = {"global_step": 0}
    if session_cfg is None:
        from backend.session_config import SessionConfig
        session_cfg = SessionConfig()
    if base_reward is None:
        base_reward = policy.reward_config
    recent_rewards = collections.deque(maxlen=100)
    t_start = time.time()
    last_curriculum_stage: Optional[int] = None
    for step in range(1, num_steps + 1):
        g_next = step_counter["global_step"] + 1
        st = _sync_curriculum(
            policy, session_cfg, base_reward, g_next, use_curriculum=getattr(session_cfg, "curriculum_enabled", False)
        )
        if (
            getattr(session_cfg, "curriculum_enabled", False)
            and getattr(session_cfg, "curriculum_reset_gumbel_tau", True)
            and last_curriculum_stage is not None
            and st != last_curriculum_stage
        ):
            policy.reset_gumbel_tau_warmup()
        last_curriculum_stage = st
        info = policy.pretrain_grid_step(optimizer, grad_clip=t.grad_clip)
        step_counter["global_step"] += 1
        g = step_counter["global_step"]
        if getattr(session_cfg, "curriculum_enabled", False):
            info["curriculum_stage"] = st
        if recorder is not None:
            recorder.log_step(global_step=g, step_in_phase=step, phase=viewer_phase, info=info)
        recent_rewards.append(info["reward"])
        if step % t.log_every == 0 or step == 1:
            mean_r = sum(recent_rewards) / len(recent_rewards)
            elapsed = time.time() - t_start
            sps = step / elapsed
            fp = 1.0 if info.get("from_policy", True) else 0.0
            show_cur = getattr(session_cfg, "curriculum_enabled", False)
            cur = f"  cur=stage{st}" if show_cur else ""
            print(
                f"[{label}] step {step:5d} | r={info['reward']:+7.3f}  meanR={mean_r:+7.3f}  "
                f"policy_samp={fp:.0f} | loss={info['loss']:+9.2f}  baseline={info['baseline']:+7.3f}  "
                f"H={info['entropy']:8.1f}  tau={info['gumbel_tau']:.3f}  |g|={info['grad_norm']:6.2f}  "
                f"({sps:.1f} ep/s){cur}"
            )
        if t.ckpt_every and step % t.ckpt_every == 0:
            ck = os.path.join(t.out_dir, f"policy_pretrain_{label}_step_{step:06d}.pt")
            torch.save({"step": step, "state_dict": policy.state_dict(), "phase": "pretrain_grid"}, ck)
            print(f"  [checkpoint] {ck}")
        if viewer is not None:
            _log_viewer_episode(viewer, viewer_phase, info, g)


def _run_full(
    policy: CellularAutomatonGenerationPolicyNetwork,
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    t: TrainingSettings,
    label: str = "full",
    viewer: Optional[ViewerLogger] = None,
    viewer_phase: str = "full",
    recorder: Optional[RunRecorder] = None,
    step_counter: dict = None,
    session_cfg: Optional[object] = None,
    base_reward: Optional[RewardConfig] = None,
):
    if step_counter is None:
        step_counter = {"global_step": 0}
    if session_cfg is None:
        from backend.session_config import SessionConfig
        session_cfg = SessionConfig()
    if base_reward is None:
        base_reward = policy.reward_config
    recent_rewards = collections.deque(maxlen=100)
    recent_solved = collections.deque(maxlen=100)
    best_reward = float("-inf")
    t_start = time.time()
    last_curriculum_stage: Optional[int] = None
    for step in range(1, num_steps + 1):
        g_next = step_counter["global_step"] + 1
        use_c = getattr(session_cfg, "curriculum_enabled", False) and getattr(
            session_cfg, "mode", "rules_only"
        ) == "grid_and_rules"
        st = _sync_curriculum(
            policy,
            session_cfg,
            base_reward,
            g_next,
            use_curriculum=use_c,
        )
        if (
            use_c
            and getattr(session_cfg, "curriculum_reset_gumbel_tau", True)
            and last_curriculum_stage is not None
            and st != last_curriculum_stage
        ):
            policy.reset_gumbel_tau_warmup()
        last_curriculum_stage = st
        info = policy.train_step(optimizer, grad_clip=t.grad_clip)
        step_counter["global_step"] += 1
        g = step_counter["global_step"]
        if getattr(session_cfg, "curriculum_enabled", False) and getattr(session_cfg, "mode", "") == "grid_and_rules":
            info["curriculum_stage"] = st
        if recorder is not None:
            recorder.log_step(global_step=g, step_in_phase=step, phase=viewer_phase, info=info)
        recent_rewards.append(info["reward"])
        recent_solved.append(1 if info["solved"] else 0)
        if info["reward"] > best_reward:
            best_reward = info["reward"]
        if step % t.log_every == 0 or step == 1:
            mean_r = sum(recent_rewards) / len(recent_rewards)
            solved_rate = sum(recent_solved) / len(recent_solved)
            elapsed = time.time() - t_start
            sps = step / elapsed
            show_cur = getattr(session_cfg, "curriculum_enabled", False) and getattr(
                session_cfg, "mode", ""
            ) == "grid_and_rules"
            cur = f"  cur=stage{st}" if show_cur else ""
            print(
                f"[{label}] step {step:5d} | r={info['reward']:+7.3f}  meanR={mean_r:+7.3f}  best={best_reward:+7.3f}  "
                f"solved={solved_rate:.0%} | loss={info['loss']:+9.2f}  baseline={info['baseline']:+7.3f}  "
                f"H={info['entropy']:8.1f}  tau={info['gumbel_tau']:.3f}  |g|={info['grad_norm']:6.2f}  "
                f"rules={info['n_rules_active']}/{info['n_rules_total']}  ({sps:.1f} ep/s){cur}"
            )
        if t.ckpt_every and step % t.ckpt_every == 0:
            ck = os.path.join(t.out_dir, f"policy_{label}_step_{step:06d}.pt")
            torch.save({"step": step, "state_dict": policy.state_dict(), "phase": "full"}, ck)
            print(f"  [checkpoint] {ck}")
        if viewer is not None:
            _log_viewer_episode(viewer, viewer_phase, info, g)


def main():
    args = parse_args()
    config_path, base = _load_base_session_config(args)
    if config_path:
        print(f"Session config file: {config_path}")
    else:
        print(
            "Session config: built-in defaults (no --config, and no curriculum_session.json "
            "in cwd or next to train.py)"
        )
    cfg = _merge_cli_into_config(base, args)
    t = get_training_settings(cfg)
    if args.load_checkpoint:
        t = replace(t, load_checkpoint=args.load_checkpoint)
    if should_auto_allocate_out_dir(t.out_dir):
        t = replace(t, out_dir=next_run_index_dir("outs"))
    cfg = _config_with_training(cfg, t)
    os.makedirs(t.out_dir, exist_ok=True)

    resolved_name = args.save_config or os.path.join(t.out_dir, "session_resolved.json")
    save_session_config(cfg, resolved_name)
    print(f"Wrote resolved session to {resolved_name}")
    if getattr(cfg, "curriculum_enabled", False):
        s = total_curriculum_steps(cfg)
        print(
            f"Curriculum: sum(curriculum_stage_steps)={s}  per_stage={cfg.curriculum_stage_steps!r}"
        )

    torch.manual_seed(t.seed)
    reward_config = get_reward_config(cfg)
    # Grid pretrain with 0 CA steps only sees init reward; huge fixed layout penalties
    # make the return identical to the baseline (no learning). Scale penalties down and
    # let Φ + small variations dominate until full training.
    if cfg.stage == "pretrain_grid" and cfg.pretrain_num_ca_steps == 0:
        new_scale = min(reward_config.init_penalty_scale, 0.25)
        if new_scale < reward_config.init_penalty_scale:
            print(
                f"pretrain_grid + 0 CA steps: capping init_penalty_scale at {new_scale} "
                f"(was {reward_config.init_penalty_scale}) so Φ stays in the learning signal."
            )
        reward_config = replace(reward_config, init_penalty_scale=new_scale)
    policy = make_policy(cfg, reward_config)
    base_reward_config = policy.reward_config

    if t.load_checkpoint and os.path.isfile(t.load_checkpoint):
        try:
            ck = torch.load(t.load_checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(t.load_checkpoint, map_location="cpu")
        policy.load_state_dict(ck["state_dict"], strict=True)
        print(f"Loaded state_dict from {t.load_checkpoint}")

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_grid = sum(p.numel() for p in policy.grid_gen.parameters() if p.requires_grad)
    n_rule = sum(p.numel() for p in policy.rule_gen.parameters() if p.requires_grad)
    print(
        f"stage={cfg.stage}  mode={cfg.mode}  grid={cfg.grid_size}x{cfg.grid_size}  max_rules={cfg.max_rules}  ca_steps={cfg.num_ca_steps}"
    )
    print(
        f"params: total={n_params:,}  grid_gen={n_grid:,}  rule_gen={n_rule:,}  lr={t.lr}  grad_clip={t.grad_clip}"
    )
    print(f"out_dir={t.out_dir}")
    print("-" * 100)

    wb_run = None
    if args.wandb:
        from backend.wandb_log import init_wandb_run, wandb_finish

        try:
            wb_run = init_wandb_run(
                out_dir=t.out_dir,
                repo_root=str(Path(__file__).resolve().parent),
                project=args.wandb_project,
                name=args.wandb_name,
                entity=args.wandb_entity,
                group=args.wandb_group,
                config_dict=session_config_to_dict(cfg),
                include_dot_git=bool(args.wandb_snapshot_git),
            )
        except (RuntimeError, OSError) as e:
            print(f"wandb: disabled — {e}")
        except Exception as e:
            print(f"wandb: could not start ({e}); continuing without W&B")
        else:

            def _finish_wandb_atexit() -> None:
                wandb_finish(wb_run)

            atexit.register(_finish_wandb_atexit)
            try:
                u = wb_run.get_url() if hasattr(wb_run, "get_url") else (getattr(wb_run, "url", None) or "")
            except Exception:
                u = ""
            if u:
                print(f"wandb: {u}")
            else:
                print("wandb: run started (set WANDB_API_KEY and see https://wandb.ai)")

    recorder = RunRecorder(
        t.out_dir,
        save_episode_json=not args.no_episodes_all,
        wandb_run=wb_run,
    )
    step_counter: dict = {"global_step": 0}

    viewer = None
    if args.viewer:
        viewer = ViewerLogger(
            t.out_dir,
            pretrain_window=args.viewer_pretrain_window,
            max_episode_files=args.viewer_max_episodes,
            session_config=session_config_to_dict(cfg),
        )
        print(f"Viewer: JSON under {t.out_dir}/viewer/ — run `uvicorn viewer_server:app --port 8765`")
        print(f"  Open http://127.0.0.1:8765/?session={t.out_dir}")

    if cfg.stage == "pretrain_grid":
        n_pre = _resolve_pretrain_num_steps(t, cfg, label="pretrain_grid")
        opt = torch.optim.AdamW(policy.grid_gen.parameters(), lr=t.lr)
        _run_pretrain(
            policy,
            opt,
            n_pre,
            t,
            label="pretrain_grid",
            viewer=viewer,
            viewer_phase="pretrain_grid",
            recorder=recorder,
            step_counter=step_counter,
            session_cfg=cfg,
            base_reward=base_reward_config,
        )
        final = os.path.join(t.out_dir, "policy_pretrain_grid_final.pt")
        torch.save({"step": n_pre, "state_dict": policy.state_dict(), "phase": "pretrain_grid"}, final)
        policy.copy_ema_to_params()
        ema = os.path.join(t.out_dir, "policy_pretrain_grid_final_ema.pt")
        torch.save({"step": n_pre, "state_dict": policy.state_dict(), "phase": "pretrain_grid"}, ema)
        print(f"saved {final} and {ema}")
        return

    if cfg.stage == "full" and t.pretrain_grid_steps > 0:
        n_pre_full = _resolve_pretrain_num_steps(t, cfg, label="pretrain_grid (before full)")
        opt = torch.optim.AdamW(policy.grid_gen.parameters(), lr=t.lr)
        _run_pretrain(
            policy,
            opt,
            n_pre_full,
            t,
            label="pretrain_grid",
            viewer=viewer,
            viewer_phase="pretrain_grid",
            recorder=recorder,
            step_counter=step_counter,
            session_cfg=cfg,
            base_reward=base_reward_config,
        )
        torch.save(
            {
                "step": n_pre_full,
                "state_dict": policy.state_dict(),
                "phase": "pretrain_done",
            },
            os.path.join(t.out_dir, "policy_after_pretrain_grid.pt"),
        )
        # Joint training with full policy (both generators)
    opt = torch.optim.AdamW(policy.parameters(), lr=t.lr)
    _run_full(
        policy,
        opt,
        t.steps,
        t,
        label="full",
        viewer=viewer,
        viewer_phase="full",
        recorder=recorder,
        step_counter=step_counter,
        session_cfg=cfg,
        base_reward=base_reward_config,
    )
    final = os.path.join(t.out_dir, "policy_final.pt")
    torch.save(
        {"step": t.steps, "state_dict": policy.state_dict(), "phase": "full", "config": session_config_to_dict(cfg)},
        final,
    )
    policy.copy_ema_to_params()
    ema = os.path.join(t.out_dir, "policy_final_ema.pt")
    torch.save(
        {"state_dict": policy.state_dict(), "phase": "full_ema", "config": session_config_to_dict(cfg)},
        ema,
    )
    print(f"saved {final} and {ema}")


if __name__ == "__main__":
    main()
