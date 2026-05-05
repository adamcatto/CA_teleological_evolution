"""
Curriculum learning: map ``global_step`` to a stage (0..6), per-stage ``PotentialWeights``,
``input_cell_tier_init_penalty``, and optional per-vocab **logit masks** for grid sampling.
"""
from __future__ import annotations

from dataclasses import fields, replace
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import torch

from .rewards.composer import RewardConfig
from .rewards.potential import PotentialWeights
from .vocabulary import NeuralCellularAutomatonVocabulary as Voc
from .vocabulary import hidden_neuron_token_values

if TYPE_CHECKING:
    from .session_config import SessionConfig

NUM_CURRICULUM_STAGES = 7

# Match ``phi_empty_sparsity`` (stronger pull than most other φ terms; see potential.py).
_EMPTY_SPARSITY_W = 1.35


def _w(**kwargs) -> PotentialWeights:
    z = {f.name: 0.0 for f in fields(PotentialWeights)}
    z.update(kwargs)
    return PotentialWeights(**z)


def _cumulative_max_potential_weights(
    per_stage: List[PotentialWeights],
) -> List[PotentialWeights]:
    """
    Each stage’s Φ weights are the **element-wise maximum** of all previous stage rows and
    the current row. A coefficient never drops when moving to a later stage; new stages
    only add or strengthen terms. Rows may omit a field (0) to mean “no change to the
    running maximum from earlier stages.”
    """
    acc = {f.name: 0.0 for f in fields(PotentialWeights)}
    out: List[PotentialWeights] = []
    for row in per_stage:
        for f in fields(PotentialWeights):
            n = f.name
            acc[n] = max(acc[n], float(getattr(row, n)))
        out.append(
            PotentialWeights(
                **{f.name: acc[f.name] for f in fields(PotentialWeights)}
            )
        )
    return out


# Per-stage *additions* (omitted fields = 0 for that row; cumulative max never decreases).
# Narrative: output placement → input structure → all neuron types+synapse → hidden targets →
# hidden geometry → path topology → readout-to-hidden rules.
STAGE_POTENTIAL_WEIGHTS_RAW: List[PotentialWeights] = [
    _w(
        empty_sparsity=_EMPTY_SPARSITY_W,
        output_right_half=1.5,
        output_size=1.0,
        init=0.5,
    ),
    _w(
        input_output_count=1.0,
        input_truth_adjacency=2.0,
        input_single_cluster=1.0,
        input_min_4_cells=1.5,
        input_left_half=1.9,
        input_centroid_near_truth=2.2,
        # Pull INPUT toward truth seeds, not into OUTPUT (see ``phi_input_separated_from_output``).
        input_separated_from_output=1.6,
    ),
    _w(
        neuron_types_presence=1.5,
        input_left_half=0.65,
        input_centroid_near_truth=0.5,
    ),
    _w(
        hidden_all_types=1.2,
        hidden_neuron_count=0.6,
        neuron_types_presence=0.5,
        input_left_half=0.55,
        input_centroid_near_truth=0.45,
    ),
    _w(
        neuron_size=0.9,
        hidden_neuron_clustering=0.5,
        hidden_neuron_count=0.3,
        input_left_half=0.5,
        input_centroid_near_truth=0.4,
    ),
    _w(
        neuron_type_non_adjacency=1.0,
        neuron_synapse_path_pairing=1.0,
        synapse_neuron_adjacency=0.3,
        input_left_half=0.45,
        input_centroid_near_truth=0.4,
    ),
    _w(
        output_two_hidden_synapse=1.0,
        synapse_output_adjacency=0.6,
        input_left_half=0.5,
        input_centroid_near_truth=0.4,
    ),
]

# Exposed to training: monotonic cumulative weights (stage k ≥ all weights from 0..k-1, per field).
STAGE_POTENTIAL_WEIGHTS: List[PotentialWeights] = _cumulative_max_potential_weights(
    STAGE_POTENTIAL_WEIGHTS_RAW
)

# Subtracted in ``compute_init_reward`` (before ``init_penalty_scale``) — only stage 1 (input).
STAGE_INPUT_TIER_PENALTY: List[float] = [0.0, 0.28, 0.0, 0.0, 0.0, 0.0, 0.0]


def normalize_curriculum_stage_budgets(
    stage_budgets: Optional[Sequence[int]],
) -> List[int]:
    """
    Pad or truncate to ``NUM_CURRICULUM_STAGES`` and clamp each budget with ``max(0, n)`` —
    same as ``stage_from_global_step`` / ``apply_curriculum_step`` use.
    """
    b = list(stage_budgets or [])
    if len(b) < NUM_CURRICULUM_STAGES:
        b = b + [0] * (NUM_CURRICULUM_STAGES - len(b))
    elif len(b) > NUM_CURRICULUM_STAGES:
        b = b[:NUM_CURRICULUM_STAGES]
    return [max(0, int(n)) for n in b]


def total_curriculum_steps(session_cfg: "SessionConfig") -> int:
    """
    Total optimizer steps for a full pass through the curriculum: sum of
    ``curriculum_stage_steps`` (after the same padding/truncation as scheduling).
    """
    b = normalize_curriculum_stage_budgets(getattr(session_cfg, "curriculum_stage_steps", None))
    return int(sum(b))


def stage_from_global_step(global_step: int, stage_budgets: List[int]) -> int:
    """
    1-based step counter: the first update uses ``global_step == 1``.
    ``stage_budgets[i]`` is how many steps stage ``i`` lasts; after the sum, we stay
    on the last stage.
    """
    if global_step < 1:
        return 0
    b = normalize_curriculum_stage_budgets(stage_budgets)
    if sum(b) == 0:
        return 0
    c = 0
    for i, n in enumerate(b):
        c += n
        if global_step <= c:
            return i
    return NUM_CURRICULUM_STAGES - 1


def build_vocab_logit_mask(
    vocabulary,
    stage: int,
    num_hidden_neuron_types: int = 1,
) -> torch.Tensor:
    """
    Per-token multipliers 1 = allowed, 0 = -inf logit in ``_sample_grid``). Stages
    progressively allow more cell types. TRUE/FALSE are masked as disallowed: seeds are
    always applied in the automaton, so the policy need not set them in the base layout.
    ``num_hidden_neuron_types`` controls which HIDDEN tokens are allowed from stage 2 on.
    """
    n = int(max(int(t.value) for t in vocabulary)) + 1
    m = torch.zeros(n, dtype=torch.float32)
    nh = int(num_hidden_neuron_types) if int(num_hidden_neuron_types) in (1, 4) else 1
    h_vals = set(hidden_neuron_token_values(Voc, nh))
    if stage <= 0:
        allow = {Voc.EMPTY.value, Voc.OUTPUT_NEURON.value}
    elif stage == 1:
        allow = {Voc.EMPTY.value, Voc.INPUT_NEURON.value, Voc.OUTPUT_NEURON.value}
    elif stage == 2:
        allow = {
            Voc.EMPTY.value,
            Voc.INPUT_NEURON.value,
            Voc.OUTPUT_NEURON.value,
            Voc.SYNAPSE.value,
        } | h_vals
    else:
        allow = {int(t.value) for t in vocabulary} - {Voc.TRUE.value, Voc.FALSE.value}
    for i in allow:
        if 0 <= i < n:
            m[i] = 1.0
    return m


def merge_curriculum_reward(base: RewardConfig, stage: int) -> RewardConfig:
    s = min(max(0, stage), NUM_CURRICULUM_STAGES - 1)
    pw = STAGE_POTENTIAL_WEIGHTS[s]
    tier = STAGE_INPUT_TIER_PENALTY[s] if s < len(STAGE_INPUT_TIER_PENALTY) else 0.0
    return replace(
        base,
        potential_weights=pw,
        input_cell_tier_init_penalty=tier,
    )


def _vocab_size() -> int:
    return int(max((int(t.value) for t in Voc), default=-1)) + 1


def apply_curriculum_step(
    global_step: int,
    session_cfg: "SessionConfig",
    base: RewardConfig,
) -> Tuple[RewardConfig, int, torch.Tensor]:
    """
    If curriculum is off, returns ``(base, 0, all-ones mask)`` (no logit change).
    Otherwise returns merged reward, stage index, and a ``(V,)`` float mask (move to
    model device before ``set_curriculum_logit_mask``).
    """
    v = _vocab_size()
    if not getattr(session_cfg, "curriculum_enabled", False):
        return base, 0, torch.ones(v, dtype=torch.float32)

    budgets = getattr(session_cfg, "curriculum_stage_steps", None) or [0] * NUM_CURRICULUM_STAGES
    st = stage_from_global_step(global_step, list(budgets))
    rc = merge_curriculum_reward(base, st)
    if getattr(session_cfg, "curriculum_use_vocab_mask", True):
        nh = int(getattr(session_cfg, "num_hidden_neuron_types", 1) or 1)
        m = build_vocab_logit_mask(Voc, st, num_hidden_neuron_types=nh)
    else:
        m = torch.ones(v, dtype=torch.float32)
    return rc, st, m
