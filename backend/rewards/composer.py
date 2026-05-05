"""
Reward composition.

Per-step total reward (bounded, comparable scales):

    r_t = α · (Φ(s_t) - Φ(s_{t-1}))             # potential-based shaping (in [-α, α])
        + β · sum_i w_dyn_i · dyn_i(s_t)         # per-step dynamics signals (in [0, β])
        + γ · terminal(s_t)                      # dense terminal reward (in [0, γ])
        + δ · solved_bonus(s_t)                  # solid bonus once XOR is satisfied
        - λ · n_violations / n_rules             # bounded soft rule penalty (use mostly 0; prefer masking)
        + μ · rule_set_bonus                     # for desirable rule properties

Initial reward (after construction, before any step):

    r_0 = β_init · Φ(s_0)

The shaping term is the central trick: by Ng/Harada/Russell, F(s, s') = γΦ(s') - Φ(s)
leaves the optimal policy unchanged. We use γ = 1 (no discount). Sum over a trajectory
telescopes to Φ(s_T) - Φ(s_0), so this never inflates with episode length — fixing a
core problem with the original reward design.
"""
from dataclasses import dataclass, field
from typing import Tuple

from .potential import Phi, PotentialWeights
from .grid_rewards import (
    mean_empty_sparsity_band_l1_deficit,
    mean_excess_dominant_token_fraction,
    mean_excess_input_output_cells,
    mean_excess_oversized_segment_fraction,
    mean_fraction_forbidden_init_tokens,
    mean_fraction_subgrids_with_zero_input,
    mean_fraction_vertical_boundary_init,
    mean_input_cell_tier_penalty,
    mean_input_count_band_violation_nonzero,
    mean_input_euclidean_proximity_to_truth_seeds,
    mean_truth_output_cell_count_deviation,
)
from .dynamics_rewards import (
    phi_output_neuron_position_stable,
    phi_rightward_truth_value_flow,
)
from .terminal_rewards import terminal_reward, is_xor_solved
from .rule_rewards import (
    rule_violation_indices,
    has_center_output_to_truth_rules,
)


@dataclass
class RewardConfig:
    potential_weights: PotentialWeights = field(default_factory=PotentialWeights)

    shaping_alpha: float = 1.0           # weight on Φ(s_t) - Φ(s_{t-1})
    init_beta: float = 0.5               # weight on Φ(s_0)
    dynamics_beta: float = 0.1           # overall scale on dynamics signals
    dyn_w_position_stable: float = 1.0
    dyn_w_truth_flow: float = 1.0

    terminal_gamma: float = 5.0          # dense terminal scale; reward in [0, γ]
    solved_bonus: float = 10.0           # paid once when all 4 subgrids match XOR target

    rule_violation_lambda: float = 0.0   # 0 by default — prefer action masking instead
    rule_set_bonus_mu: float = 0.5       # bonus for having both center-output→truth rules

    # Subtracted: mean fraction of IGNORE / INTERMEDIATE_TRUTH_VALUE in the base layout.
    forbidden_init_token_penalty: float = 180.0
    # Subtracted: mean fraction of VERTICAL_BOUNDARY alone (much larger than ``forbidden_*``).
    vertical_boundary_init_penalty: float = 650.0
    # Subtracted: mean L1 distance outside the EMPTY band (lo/hi in ``grid_rewards``); very large.
    empty_sparsity_band_deficit_penalty: float = 320.0
    # Subtracted: scales per-subgrid mean |#OUT-1|+|#TRUE-1|+|#FALSE-1| in raw cell counts.
    truth_output_cell_penalty: float = 25.0
    # Multiplies the two penalties above (not ``init_beta * Φ``). Use below 1.0 in grid pretrain
    # when `pretrain_num_ca_steps` is 0 so the dense Φ term is not washed out and returns
    # are not all identical to the baseline.
    init_penalty_scale: float = 1.0
    # Clamp |advantage| in REINFORCE (0 = no clamp). Cuts off rare spike updates when
    # baselines or penalties place returns on a different scale from Φ.
    reinforce_advantage_clip: float = 12.0
    # Subtracted from init reward: mean over subgrids of max(0, n_in-1) + max(0, n_out-1) raw cells.
    duplicate_io_cell_penalty: float = 0.0
    # Subtracted: mean_input_cell_tier_penalty (single INPUT cluster with 1–3 cells); curriculum sets >0 in stage 2.
    input_cell_tier_init_penalty: float = 0.0
    # Subtracted: fraction of the 4 subgrids with **no** INPUT cells (0–1); keep coefficient large.
    input_zero_subgrid_init_penalty: float = 80.0
    # Subtracted: mean per-subgrid INPUT count error when n≥1, outside [2, 10] (see ``grid_rewards``).
    input_count_band_init_penalty: float = 4.0
    # Subtracted: mean normalized distance of INPUT from nearer seeded TRUE/FALSE (0–1, higher = worse).
    input_far_from_seeds_init_penalty: float = 12.0
    # Subtracted: per-subgrid sum of max(0, area/total−0.02) for each **segmented** component
    # (excludes EMPTY/IGNORE/VERTICAL_BOUNDARY blobs). 0 = no object above 2% of cells.
    oversized_segment_fraction_penalty: float = 12.0
    # Subtracted: per-subgrid sum of max(0, raw_count/total−0.05) for each vocab token **except**
    # EMPTY (so large empty regions are not penalized here).
    dominant_token_fraction_penalty: float = 12.0

    # Episode-to-episode (grid → next grid) bonuses applied in the policy, compared to
    # ``_prev_base_grid`` (see ``network``). 0 disables.
    # Output: when the previous layout had the output neuron on the **left** half, reward
    # proportional to columnwise progress toward the right half (see ``_get_grid_motion_bonuses``).
    grid_motion_output_toward_right_coef: float = 0.12
    # Input: reward when the input centroid’s distance to the nearest TRUE/FALSE cell
    # decreases from the previous layout to the current one.
    grid_motion_input_toward_truth_coef: float = 0.55


def compute_init_reward(automaton, config: RewardConfig) -> Tuple[float, float]:
    """
    Reward at episode start (before any step). Returns (reward, phi_0) — pass phi_0 in
    as `prev_phi` for the first call to `compute_step_reward`.
    """
    phi_0 = Phi(automaton, config.potential_weights)
    # Strong discouragement of sampling reserved tokens in the grid policy's layout.
    bad_frac = mean_fraction_forbidden_init_tokens(automaton)
    bnd_frac = mean_fraction_vertical_boundary_init(automaton)
    sp_def = mean_empty_sparsity_band_l1_deficit(automaton)
    dev = mean_truth_output_cell_count_deviation(automaton)
    ex_io = mean_excess_input_output_cells(automaton)
    tier = mean_input_cell_tier_penalty(automaton)
    in_zero = mean_fraction_subgrids_with_zero_input(automaton)
    in_band = mean_input_count_band_violation_nonzero(automaton)
    in_prox = mean_input_euclidean_proximity_to_truth_seeds(automaton)
    seg_ex = mean_excess_oversized_segment_fraction(automaton)
    tok_ex = mean_excess_dominant_token_fraction(automaton)
    pen = (
        config.forbidden_init_token_penalty * bad_frac
        + config.vertical_boundary_init_penalty * bnd_frac
        + config.empty_sparsity_band_deficit_penalty * sp_def
        + config.truth_output_cell_penalty * dev
        + config.duplicate_io_cell_penalty * ex_io
        + config.input_cell_tier_init_penalty * tier
        + config.input_zero_subgrid_init_penalty * in_zero
        + config.input_count_band_init_penalty * in_band
        + config.input_far_from_seeds_init_penalty * in_prox
        + config.oversized_segment_fraction_penalty * seg_ex
        + config.dominant_token_fraction_penalty * tok_ex
    ) * config.init_penalty_scale
    return config.init_beta * phi_0 - pen, phi_0


def compute_step_reward(automaton, prev_phi: float, config: RewardConfig) -> Tuple[float, float]:
    """
    Reward for the transition into the *current* automaton state. Caller must call
    this AFTER `automaton.step()` and pass in the Φ value from the *previous* state.

    Returns (reward, new_phi). Persist new_phi for the next step.
    """
    new_phi = Phi(automaton, config.potential_weights)
    shaping = config.shaping_alpha * (new_phi - prev_phi)

    dyn = (
        config.dyn_w_position_stable * phi_output_neuron_position_stable(automaton)
        + config.dyn_w_truth_flow * phi_rightward_truth_value_flow(automaton)
    )
    total_dyn_w = config.dyn_w_position_stable + config.dyn_w_truth_flow
    dyn_normalized = dyn / total_dyn_w if total_dyn_w > 0 else 0.0

    term = config.terminal_gamma * terminal_reward(automaton)
    solved = config.solved_bonus if is_xor_solved(automaton) else 0.0

    return (shaping + config.dynamics_beta * dyn_normalized + term + solved, new_phi)


def compute_rule_set_score(rules, vocabulary, config: RewardConfig) -> float:
    """
    Once-per-rule-set score: bounded soft penalty for hard-constraint violators (only
    used when not action-masking) plus a small bonus for desirable rule properties.
    """
    n_rules = max(len(list(rules)), 1)
    violators = rule_violation_indices(rules, vocabulary)
    penalty = config.rule_violation_lambda * (len(violators) / n_rules)

    has_true, has_false = has_center_output_to_truth_rules(rules, vocabulary)
    bonus = 0.0
    if has_true:
        bonus += 0.5
    if has_false:
        bonus += 0.5
    bonus *= config.rule_set_bonus_mu

    return bonus - penalty


def rule_active_mask(rules, vocabulary):
    """
    Build a 0/1 mask over `rules` suitable for `TensorCellularAutomaton.step_soft`:
    1 for rules that pass hard constraints, 0 for violators.
    """
    import torch
    n = len(list(rules))
    mask = torch.ones(n)
    for i in rule_violation_indices(rules, vocabulary):
        mask[i] = 0.0
    return mask
