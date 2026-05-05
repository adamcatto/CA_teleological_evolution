"""
Generator networks + REINFORCE policy.

Generators (all expose the same `logits()` interface so the policy is generator-agnostic):

  * `GridGenerationNetwork`         — MLP decoder. Simple, fast, weak inductive bias.
  * `GridGenerationUNet`            — convolutional upsampling decoder (default). Spatial
                                      inductive bias makes neighborhood-coherent grids
                                      easier to discover.
  * `RuleGenerationDiffusionNetwork` — flat MLP over the rule tensor. Fast, no rule
                                      coordination.
  * `RuleGenerationTransformer`     — self-attention across R learnable rule slots
                                      (default). Lets rules coordinate so the rule set
                                      hangs together as a program rather than R
                                      independent guesses.

`CellularAutomatonGenerationPolicyNetwork` samples a grid + rule set, runs the numpy CA
inside an `AutomataSession`, scores with the shaped reward, and applies a REINFORCE
update with a running-mean baseline. Hard rule-set constraints
(truth-value-out-of-thin-air, output-neuron teleport) are enforced by **action masking**
— violators are dropped from the rule list before simulation, not penalized.
"""
import math
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .vocabulary import Vocabulary, hidden_neuron_token_values
from .grid import Grid
from .rule import Rule
from .tensor_ca import TensorCellularAutomaton
from .cellular_automata import NeuralXORAutomaton
from .session import AutomataSession
from .rewards import RewardConfig, rule_violation_indices


def _vocab_size(vocabulary) -> int:
    return max(item.value for item in vocabulary) + 1


def build_staged_logit_add(
    vocabulary,
    priors: Optional[dict] = None,
    *,
    num_hidden_neuron_types: int = 1,
) -> torch.Tensor:
    """
    Per-vocab **additive** boost applied to grid logits before τ-scaling, to **prioritize**
    which token classes to sample first (I/O, then synapse/hidden, etc.). Shaped ``(V,)``;
    broadcast in ``_sample_grid`` to ``(V, H, W)``.

    ``priors``:
      * ``None`` — use built-in defaults (I/O boosted, ``rest`` slightly down).
      * ``{}`` — all zeros (disable).

    Optional keys::

        input, output, synapse, hidden  — one float each (same boost for all 4 hiddens)
        empty, rest  — for EMPTY, and for (IGNORE, TRUE, FALSE, INTERMEDIATE, VERTICAL_BOUNDARY)
    """
    v = int(_vocab_size(vocabulary))
    V = vocabulary
    if priors is not None and len(priors) == 0:
        return torch.zeros(v, dtype=torch.float32)
    d = {
        "input": 0.45,
        "output": 0.45,
        "synapse": 0.22,
        "hidden": 0.20,
        "empty": 0.0,
        "rest": -0.12,
    }
    if priors:
        d.update(priors)
    t = torch.zeros(v, dtype=torch.float32)
    t[int(V.INPUT_NEURON.value)] = float(d["input"])
    t[int(V.OUTPUT_NEURON.value)] = float(d["output"])
    t[int(V.SYNAPSE.value)] = float(d["synapse"])
    nh = int(num_hidden_neuron_types) if int(num_hidden_neuron_types) in (1, 4) else 1
    h_boost = float(d["hidden"])
    for v in hidden_neuron_token_values(V, nh):
        t[int(v)] = h_boost
    t[int(V.EMPTY.value)] = float(d["empty"])
    r = float(d["rest"])
    for item in (V.IGNORE, V.TRUE, V.FALSE, V.INTERMEDIATE_TRUTH_VALUE, V.VERTICAL_BOUNDARY):
        t[int(item.value)] = r
    return t


def _init_head_bias_favor_empty(
    bias_1d: torch.Tensor, vocabulary, *, empty_boost: float = 2.5, other: float = -0.05
) -> None:
    """1D bias over vocab classes (e.g. Conv2d head). Strong EMPTY logit, mild negative elsewhere."""
    ev = int(vocabulary.EMPTY.value)
    bias_1d.fill_(other)
    bias_1d[ev] = other + empty_boost


def _init_mlp_output_bias_favor_empty(
    last_linear: nn.Linear, grid_hw: Tuple[int, int], vocabulary, *, empty_boost: float = 2.5, other: float = -0.05
) -> None:
    """Last flat linear mapping to H*W*V: each cell's vocab block is biased toward EMPTY."""
    H, W = grid_hw
    v = _vocab_size(vocabulary)
    if last_linear.bias is None:
        return
    with torch.no_grad():
        b = last_linear.bias.data.view(H, W, v)
        b.fill_(other)
        b[:, :, int(vocabulary.EMPTY.value)] = other + empty_boost
        # Zeros: with zero latent, logits = bias only (same EMPTY prior every cell). Random
        # weights would add a cell-dependent W·h and smear the distribution at init.
        nn.init.zeros_(last_linear.weight)


class GridGenerationNetwork(nn.Module):
    """
    Decoder producing logits over the vocabulary at every cell of an (H, W) grid.
    Output of `forward()` is a Gumbel-softmax sample with shape (V, H, W).
    """
    def __init__(
        self,
        vocabulary,
        grid_size: Tuple[int, int] = (16, 16),
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        empty_logit_bias: float = 0.0,
    ):
        super().__init__()
        self.vocabulary = vocabulary
        self.vocab_size = _vocab_size(vocabulary)
        self.grid_size = grid_size
        self.embedding_dim = embedding_dim
        self.empty_logit_bias = float(empty_logit_bias)

        # Start at "all EMPTY" prior: zero latent + last-layer bias favors EMPTY per cell.
        self.latent_representation = nn.Parameter(torch.zeros(embedding_dim))
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, grid_size[0] * grid_size[1] * self.vocab_size),
        )
        _init_mlp_output_bias_favor_empty(self.decoder[3], grid_size, vocabulary)

    def logits(self) -> torch.Tensor:
        """Returns (V, H, W) logits. Channel-first to align with TensorCA conventions."""
        flat = self.decoder(self.latent_representation)
        out = flat.view(self.grid_size[0], self.grid_size[1], self.vocab_size).permute(2, 0, 1).contiguous()
        if self.empty_logit_bias != 0.0:
            ev = int(self.vocabulary.EMPTY.value)
            out[ev, :, :].add_(self.empty_logit_bias)
        return out

    def forward(self, tau: float = 1.0) -> torch.Tensor:
        """Returns a (V, H, W) Gumbel-softmax sample (hard, straight-through)."""
        return F.gumbel_softmax(self.logits(), tau=tau, hard=True, dim=0)


class RuleGenerationDiffusionNetwork(nn.Module):
    """
    Flat MLP decoder producing logits for an (R, V, 2, 3, 3) rule tensor. Fast, but
    the rules can't see each other — they're R independent guesses. Kept for ablation
    against `RuleGenerationTransformer`.

    (The "Diffusion" name in the original sketch was aspirational. A real D3PM-style
    path over discrete tokens is outside the current scope.)
    """
    def __init__(self, vocabulary, max_rules: int = 256, hidden_dim: int = 512, bottleneck_dim: Optional[int] = None):
        super().__init__()
        self.vocabulary = vocabulary
        self.vocab_size = _vocab_size(vocabulary)
        self.max_rules = max_rules
        self.output_shape = (max_rules, self.vocab_size, 2, 3, 3)
        flat_size = max_rules * self.vocab_size * 2 * 3 * 3
        b = bottleneck_dim if bottleneck_dim is not None else max(128, min(hidden_dim, flat_size) // 4)

        self.latent_representation = nn.Parameter(torch.randn(*self.output_shape) * 0.1)
        self.decoder = nn.Sequential(
            nn.Linear(flat_size, b),
            nn.LayerNorm(b),
            nn.GELU(),
            nn.Linear(b, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, flat_size),
        )

    def logits(self) -> torch.Tensor:
        """Returns (R, V, 2, 3, 3) logits."""
        flat = self.decoder(self.latent_representation.view(-1))
        return flat.view(*self.output_shape)

    def forward(self, tau: float = 1.0) -> torch.Tensor:
        """Returns a (R, V, 2, 3, 3) Gumbel-softmax sample (hard, straight-through)."""
        return F.gumbel_softmax(self.logits(), tau=tau, hard=True, dim=1)


# --------------------------------------------------------------------------------------
# Better generators: U-Net-ish spatial decoder + transformer rule decoder
# --------------------------------------------------------------------------------------

class _UpBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        gn_groups = min(8, c_out) if c_out % min(8, c_out) == 0 else 1
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.GroupNorm(gn_groups, c_out),
            nn.GELU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.GroupNorm(gn_groups, c_out),
            nn.GELU(),
        )

    def forward(self, x):
        return self.body(self.up(x))


class GridGenerationUNet(nn.Module):
    """
    Convolutional upsampling decoder from a learned latent to (V, H, W) logits.

    Starts from a ``start_spatial``×``start_spatial`` seed projected from a latent vector, then upsamples by
    powers of two until reaching (H, W). Channels are halved at each level (bounded
    below by `base_channels`) so parameter count stays reasonable. GroupNorm + GELU
    throughout; a final 1x1 conv produces the vocab logits.

    This isn't a full encoder-decoder U-Net (there's no encoder input to skip from)
    but the name fits the interface the user requested and reserves room to add a
    conditional-encoder path later without breaking callers.
    """
    def __init__(
        self,
        vocabulary,
        grid_size: Tuple[int, int] = (16, 16),
        latent_dim: int = 32,
        base_channels: int = 16,
        start_spatial: int = 2,
        proj_bottleneck_dim: Optional[int] = None,
        empty_logit_bias: float = 0.0,
    ):
        super().__init__()
        H, W = grid_size
        assert H == W, "GridGenerationUNet currently assumes square grids."
        assert H % start_spatial == 0 and (H // start_spatial) & ((H // start_spatial) - 1) == 0, \
            f"grid_size {grid_size} must be start_spatial * power-of-2"
        n_up = int(math.log2(H // start_spatial))

        self.vocabulary = vocabulary
        self.vocab_size = _vocab_size(vocabulary)
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.start_spatial = start_spatial

        top_channels = base_channels * (2 ** n_up)
        out_flat = top_channels * start_spatial * start_spatial
        if proj_bottleneck_dim is not None:
            b = proj_bottleneck_dim
        else:
            b = max(32, latent_dim // 4) if latent_dim > 64 else max(16, latent_dim // 2)
        # Zero latent → spatial features near 0 before head; head bias encodes EMPTY prior.
        self.latent = nn.Parameter(torch.zeros(latent_dim))
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, b),
            nn.GELU(),
            nn.Linear(b, out_flat),
        )
        self.proj_norm = nn.LayerNorm(out_flat)
        self._top_channels = top_channels

        blocks = []
        c = top_channels
        for _ in range(n_up):
            c_next = max(c // 2, base_channels)
            blocks.append(_UpBlock(c, c_next))
            c = c_next
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(c, self.vocab_size, kernel_size=1)
        self.empty_logit_bias = float(empty_logit_bias)
        with torch.no_grad():
            # Logits = bias only at init (independent of upstream features) → uniform EMPTY
            # prior on every (y,x); re-enable learning by letting optimizer update weights.
            nn.init.zeros_(self.head.weight)
            if self.head.bias is not None:
                _init_head_bias_favor_empty(self.head.bias, vocabulary)

    def logits(self) -> torch.Tensor:
        h = self.proj_norm(self.proj(self.latent))
        h = h.view(self._top_channels, self.start_spatial, self.start_spatial).unsqueeze(0)
        for block in self.blocks:
            h = block(h)
        out = self.head(h).squeeze(0)  # (V, H, W)
        if self.empty_logit_bias != 0.0:
            ev = int(self.vocabulary.EMPTY.value)
            out[ev, :, :].add_(self.empty_logit_bias)
        return out

    def forward(self, tau: float = 1.0) -> torch.Tensor:
        return F.gumbel_softmax(self.logits(), tau=tau, hard=True, dim=0)


class GridGenerationUNetMultiHead(nn.Module):
    """
    Same spatial trunk as `GridGenerationUNet`, but **separate 1×1 heads** per semantic
    group: EMPTY, (INPUT, OUTPUT), SYNAPSE, N HIDDEN type logits (1 or 4), and REMAINING
    reserved layout tokens. Logits are assembled into a full ``(V, H, W)`` tensor for the
    same per-cell Categorical policy as the single-head UNet.

    Use with `build_staged_logit_add` + `grid_staged_logit_priors` on the policy to
    prioritize I/O, then synapse/hidden, etc.
    """

    def __init__(
        self,
        vocabulary,
        grid_size: Tuple[int, int] = (16, 16),
        latent_dim: int = 32,
        base_channels: int = 16,
        start_spatial: int = 2,
        proj_bottleneck_dim: Optional[int] = None,
        empty_logit_bias: float = 0.0,
        num_hidden_neuron_types: int = 1,
    ):
        super().__init__()
        H, W = grid_size
        assert H == W, "GridGenerationUNetMultiHead currently assumes square grids."
        assert H % start_spatial == 0 and (H // start_spatial) & ((H // start_spatial) - 1) == 0, \
            f"grid_size {grid_size} must be start_spatial * power-of-2"
        n_up = int(math.log2(H // start_spatial))

        self.vocabulary = vocabulary
        self.vocab_size = _vocab_size(vocabulary)
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.start_spatial = start_spatial
        nh = int(num_hidden_neuron_types)
        if nh not in (1, 4):
            raise ValueError("num_hidden_neuron_types must be 1 or 4")
        self._num_hidden_neuron_types = nh

        top_channels = base_channels * (2 ** n_up)
        out_flat = top_channels * start_spatial * start_spatial
        if proj_bottleneck_dim is not None:
            b = proj_bottleneck_dim
        else:
            b = max(32, latent_dim // 4) if latent_dim > 64 else max(16, latent_dim // 2)
        self.latent = nn.Parameter(torch.zeros(latent_dim))
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, b),
            nn.GELU(),
            nn.Linear(b, out_flat),
        )
        self.proj_norm = nn.LayerNorm(out_flat)
        self._top_channels = top_channels

        blocks = []
        c = top_channels
        for _ in range(n_up):
            c_next = max(c // 2, base_channels)
            blocks.append(_UpBlock(c, c_next))
            c = c_next
        self.blocks = nn.ModuleList(blocks)
        self._trunk_c = c
        # One head per group; channel counts sum to full vocab for direct slot mapping.
        self.head_empty = nn.Conv2d(c, 1, kernel_size=1)
        self.head_io = nn.Conv2d(c, 2, kernel_size=1)
        self.head_synapse = nn.Conv2d(c, 1, kernel_size=1)
        self.head_hidden = nn.Conv2d(c, self._num_hidden_neuron_types, kernel_size=1)
        self.head_rest = nn.Conv2d(c, 5, kernel_size=1)
        self.empty_logit_bias = float(empty_logit_bias)
        V = vocabulary
        self._idx_empty = int(V.EMPTY.value)
        self._idx_in = int(V.INPUT_NEURON.value)
        self._idx_out = int(V.OUTPUT_NEURON.value)
        self._idx_syn = int(V.SYNAPSE.value)
        self._idx_h = [int(V.HIDDEN_NEURON_1.value), int(V.HIDDEN_NEURON_2.value),
                       int(V.HIDDEN_NEURON_3.value), int(V.HIDDEN_NEURON_4.value)]
        self._idx_rest = [
            int(V.IGNORE.value),
            int(V.TRUE.value),
            int(V.FALSE.value),
            int(V.INTERMEDIATE_TRUTH_VALUE.value),
            int(V.VERTICAL_BOUNDARY.value),
        ]
        with torch.no_grad():
            for m in (self.head_io, self.head_synapse, self.head_hidden, self.head_rest):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            nn.init.zeros_(self.head_empty.weight)
            if self.head_empty.bias is not None:
                # One channel: favor EMPTY at init (same spirit as _init_head_bias_favor_empty).
                self.head_empty.bias[0] = 2.45

    def logits(self) -> torch.Tensor:
        h = self.proj_norm(self.proj(self.latent))
        h = h.view(self._top_channels, self.start_spatial, self.start_spatial).unsqueeze(0)
        for block in self.blocks:
            h = block(h)
        he = self.head_empty(h)
        io = self.head_io(h)
        sy = self.head_synapse(h)
        hd = self.head_hidden(h)
        rs = self.head_rest(h)
        Hh, Ww = he.shape[2], he.shape[3]
        Vn = self.vocab_size
        out = torch.zeros((Vn, Hh, Ww), device=h.device, dtype=h.dtype)
        out[self._idx_empty] = he[0, 0]
        out[self._idx_in] = io[0, 0]
        out[self._idx_out] = io[0, 1]
        out[self._idx_syn] = sy[0, 0]
        for k in range(self._num_hidden_neuron_types):
            out[self._idx_h[k]] = hd[0, k]
        for k in range(5):
            out[self._idx_rest[k]] = rs[0, k]
        if self.empty_logit_bias != 0.0:
            out[self._idx_empty, :, :].add_(self.empty_logit_bias)
        return out

    def forward(self, tau: float = 1.0) -> torch.Tensor:
        return F.gumbel_softmax(self.logits(), tau=tau, hard=True, dim=0)


class RuleGenerationTransformer(nn.Module):
    """
    Self-attention across R learnable rule-slot embeddings. Each slot decodes to a
    (V, 2, 3, 3) rule pattern via a per-slot linear head. Because the slots attend to
    each other before decoding, the generator can produce rule sets that cohere as a
    program (e.g. a matching "apply synapse" rule for every "propagate truth" rule)
    rather than independent per-rule guesses.
    """
    def __init__(
        self,
        vocabulary,
        max_rules: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_mult: int = 2,
        dropout: float = 0.0,
        head_bottleneck_dim: Optional[int] = None,
    ):
        super().__init__()
        self.vocabulary = vocabulary
        self.vocab_size = _vocab_size(vocabulary)
        self.max_rules = max_rules
        self.output_shape = (max_rules, self.vocab_size, 2, 3, 3)
        out_per_slot = self.vocab_size * 2 * 3 * 3
        hb = head_bottleneck_dim if head_bottleneck_dim is not None else max(32, d_model // 4)

        self.rule_queries = nn.Parameter(torch.randn(max_rules, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, hb),
            nn.GELU(),
            nn.Linear(hb, out_per_slot),
        )

    def logits(self) -> torch.Tensor:
        h = self.rule_queries.unsqueeze(0)            # (1, R, D)
        h = self.transformer(h).squeeze(0)            # (R, D)
        h = self.norm(h)
        out = self.head(h)                            # (R, V*2*3*3)
        return out.view(*self.output_shape)

    def forward(self, tau: float = 1.0) -> torch.Tensor:
        return F.gumbel_softmax(self.logits(), tau=tau, hard=True, dim=1)


# --------------------------------------------------------------------------------------
# Policy network
# --------------------------------------------------------------------------------------

class CellularAutomatonGenerationPolicyNetwork(nn.Module):
    """
    Combines `GridGenerationNetwork` and `RuleGenerationDiffusionNetwork` into a single
    policy that samples a grid + rule set, runs the CA, and applies a REINFORCE update.

    Two task modes:
      * `mode='rules_only'` (default): only the rule generator is trained. The grid is
        constructed by `NeuralXORAutomaton` (4-stack with seeded truth-value pairs)
        starting from an all-EMPTY layout, so the grid generator's output is unused.
      * `mode='grid_and_rules'`: same XOR task, same 4-stack structure, but the
        sampled grid becomes the *base layout* that `NeuralXORAutomaton` replicates
        into each of the 4 stacked subgrids. Truth-value seed positions are still
        overwritten with the canonical XOR inputs, so the policy can place
        neurons / synapses anywhere except at those four cells. The 3-row
        VERTICAL_BOUNDARY blocks between stacks remain fixed (the stacking trick
        relies on them).

    Hard rule-set constraints are enforced by **action masking** — `rule_violation_indices`
    is queried on the sampled rules and the violators are dropped before the CA runs.
    No giant penalties.

    Variance reduction: a running-mean baseline subtracts the mean recent reward from
    each return, which is the standard low-overhead REINFORCE baseline.
    """

    def __init__(
        self,
        vocabulary,
        grid_size: Tuple[int, int] = (16, 16),
        max_rules: int = 64,
        num_ca_steps: int = 16,
        # --- sampling / exploration (τ clamped to [gumbel_tau_min, gumbel_tau_max] every step) ---
        gumbel_tau: float = 0.5,             # target τ after warmup; then multiplicative decay
        gumbel_tau_start: float = 0.08,      # first step(s): low τ ≈ sharp categorical (near argmax)
        gumbel_tau_warmup_steps: int = 500,  # linear ramp start → target; 0 = begin at target
        gumbel_tau_min: float = 0.05,        # hard floor for τ
        gumbel_tau_max: float = 0.5,         # hard ceiling for τ
        gumbel_tau_decay: float = 0.999,   # per-step multiplicative decay (post-warmup)
        entropy_coef: float = 0.01,          # weight on entropy bonus (encourages exploration)
        # --- REINFORCE / stability ---
        baseline_momentum: float = 0.95,
        ema_decay: float = 0.999,            # Polyak averaging of params for evaluation
        # --- composition ---
        reward_config: Optional[RewardConfig] = None,
        mode: str = "rules_only",
        use_action_masking: bool = True,
        # --- grid pretrain (Φ from init; empty rule set) ---
        pretrain_policy_mix: float = 0.5,
        pretrain_num_ca_steps: int = 0,
        # Penalize fraction of base cells that differ from previous episode (pretrain / grid_and_rules).
        grid_switching_coef: float = 0.08,
        # Additive per-vocab logits for staged sampling (I/O, then synapse/hidden, …). None = built-in defaults.
        grid_staged_logit_priors: Optional[dict] = None,
        # 1 = HIDDEN_NEURON_1 only; 4 = four hidden types. Masks disallowed hiddens at sample time.
        num_hidden_neuron_types: int = 1,
        # --- generator swap points ---
        grid_generator_cls: Optional[Type[nn.Module]] = None,
        rule_generator_cls: Optional[Type[nn.Module]] = None,
        grid_generator_kwargs: Optional[dict] = None,
        rule_generator_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        assert mode in {"rules_only", "grid_and_rules"}
        self.vocabulary = vocabulary
        self.vocab_size = _vocab_size(vocabulary)
        self.grid_size = grid_size
        self.max_rules = max_rules
        self.num_ca_steps = num_ca_steps

        lo, hi = float(gumbel_tau_min), float(gumbel_tau_max)
        if hi < lo:
            raise ValueError("gumbel_tau_max must be >= gumbel_tau_min")

        def _clip_tau(x: float) -> float:
            return min(max(x, lo), hi)

        self.gumbel_tau_target = _clip_tau(float(gumbel_tau))
        self.gumbel_tau_start = _clip_tau(float(gumbel_tau_start))
        self.gumbel_tau_warmup_steps = int(gumbel_tau_warmup_steps)
        self.gumbel_tau_init = gumbel_tau  # alias: pre-clamp value from config (for logs)
        self.gumbel_tau_min = lo
        self.gumbel_tau_max = hi
        self.gumbel_tau_decay = gumbel_tau_decay
        self.entropy_coef = entropy_coef

        self.baseline_momentum = baseline_momentum
        self.ema_decay = ema_decay
        self.mode = mode
        self.use_action_masking = use_action_masking
        self.pretrain_policy_mix = pretrain_policy_mix
        self.pretrain_num_ca_steps = pretrain_num_ca_steps
        self.grid_switching_coef = float(grid_switching_coef)
        self._prev_base_grid: Optional[np.ndarray] = None
        nht = int(num_hidden_neuron_types)
        if nht not in (1, 4):
            raise ValueError("num_hidden_neuron_types must be 1 or 4")
        self.num_hidden_neuron_types = nht
        h_all = hidden_neuron_token_values(self.vocabulary, 4)
        h_active = set(hidden_neuron_token_values(self.vocabulary, nht))
        sm = torch.ones(self.vocab_size, dtype=torch.float32)
        for v in h_all:
            if int(v) not in h_active:
                sm[int(v)] = 0.0
        self.register_buffer("_static_hidden_logit_mask", sm)

        self.register_buffer(
            "_grid_staged_logit_add",
            build_staged_logit_add(
                self.vocabulary, grid_staged_logit_priors, num_hidden_neuron_types=nht
            ),
        )

        self.reward_config = reward_config if reward_config is not None else RewardConfig()
        _clip = getattr(self.reward_config, "reinforce_advantage_clip", 0.0) or 0.0
        self.reinforce_advantage_clip = float(_clip) if _clip > 0 else 0.0
        _rc = self.reward_config
        self._grid_motion_out_coef = float(getattr(_rc, "grid_motion_output_toward_right_coef", 0.0) or 0.0)
        self._grid_motion_in_coef = float(getattr(_rc, "grid_motion_input_toward_truth_coef", 0.0) or 0.0)

        # Default to the stronger generators; let callers swap if they want ablations.
        grid_cls = grid_generator_cls if grid_generator_cls is not None else GridGenerationUNet
        rule_cls = rule_generator_cls if rule_generator_cls is not None else RuleGenerationTransformer
        gkw0 = dict(grid_generator_kwargs or {})
        if grid_cls is GridGenerationUNetMultiHead:
            gkw0.setdefault("num_hidden_neuron_types", nht)
        grid_kwargs = {"grid_size": grid_size, **gkw0}
        rule_kwargs = {"max_rules": max_rules, **(rule_generator_kwargs or {})}
        self.grid_gen = grid_cls(vocabulary, **grid_kwargs)
        self.rule_gen = rule_cls(vocabulary, **rule_kwargs)
        self.tensor_ca = TensorCellularAutomaton(self.vocab_size, num_steps=num_ca_steps)

        # Buffers (moved with .to(device), checkpointed with state_dict).
        self.register_buffer("baseline", torch.zeros(1))
        self.register_buffer("baseline_initialized", torch.tensor(False))
        _tau0 = self.gumbel_tau_start if self.gumbel_tau_warmup_steps > 0 else self.gumbel_tau_target
        self.register_buffer("gumbel_tau", torch.tensor(_clip_tau(float(_tau0))))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer(
            "_curriculum_logit_mask", torch.ones(self.vocab_size, dtype=torch.float32)
        )
        self._clamp_tau_buffer()

        # Polyak-averaged (EMA) copy of trainable params for evaluation; see `ema_params()`.
        self._ema_params: Optional[dict] = None
        self._init_ema()

    @torch.no_grad()
    def _clip_tau_scalar(self, x: float) -> float:
        return min(max(x, self.gumbel_tau_min), self.gumbel_tau_max)

    @torch.no_grad()
    def _clamp_tau_buffer(self) -> None:
        """Keep ``gumbel_tau`` in [min, max] (e.g. after loading an older checkpoint)."""
        self.gumbel_tau.fill_(self._clip_tau_scalar(float(self.gumbel_tau.item())))

    @torch.no_grad()
    def set_curriculum_logit_mask(self, mask: Optional[torch.Tensor]) -> None:
        """
        When ``mask`` is ``None`` or all-ones, sampling is unchanged. Otherwise ``mask`` is
        a length-``V`` float tensor (1 = allowed, 0 = disallowed) broadcast in ``_sample_grid``.
        """
        b = self._curriculum_logit_mask
        if mask is None:
            b.fill_(1.0)
            return
        m = mask.detach().to(device=b.device, dtype=torch.float32).view(-1)
        if m.numel() != self.vocab_size:
            raise ValueError(
                f"curriculum logit mask length {m.numel()} != vocab size {self.vocab_size}"
            )
        b.copy_(m)

    @torch.no_grad()
    def reset_gumbel_tau_warmup(self) -> None:
        """
        Reset τ and the per-step counter used by ``_anneal_tau`` so the linear **warmup**
        from ``gumbel_tau_start`` → target runs again. Call when the curriculum **stage**
        changes so sampling stays stochastic while Φ / masks shift (avoids a frozen policy).
        """
        self.step_count.zero_()
        if self.gumbel_tau_warmup_steps > 0:
            t0 = self._clip_tau_scalar(float(self.gumbel_tau_start))
        else:
            t0 = self._clip_tau_scalar(float(self.gumbel_tau_target))
        self.gumbel_tau.fill_(t0)

    # ----- EMA parameter tracking -----

    def _init_ema(self):
        self._ema_params = {
            name: p.detach().clone() for name, p in self.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def _update_ema(self):
        if self._ema_params is None:
            self._init_ema()
            return
        d = self.ema_decay
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            self._ema_params[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_ema_to_params(self):
        """Overwrite current parameters with their EMA values. Useful before evaluation."""
        if self._ema_params is None:
            return
        for name, p in self.named_parameters():
            if name in self._ema_params:
                p.data.copy_(self._ema_params[name])

    # ----- sampling -----

    def _sample_grid(self):
        """
        Returns (grid_idx (H, W), grid_logp scalar, grid_entropy scalar).
        Sampling is Categorical per-cell (not Gumbel-softmax). Gumbel temperature
        controls the sharpness of the *logits* passed to the distribution — lower tau
        means more confident, so annealing tau pushes the policy from explore to exploit
        without changing the sampling mechanism.
        """
        self._clamp_tau_buffer()
        logits = self.grid_gen.logits()  # (V, H, W)
        logits = logits + self._grid_staged_logit_add.to(logits.device).view(self.vocab_size, 1, 1)
        sh = self._static_hidden_logit_mask.to(logits.device)
        cm = (self._curriculum_logit_mask.to(logits.device) * sh).view(self.vocab_size, 1, 1)
        logits = logits + (1.0 - cm) * (-1e9)
        per_cell_logits = (logits / self.gumbel_tau.clamp_min(1e-3)).permute(1, 2, 0)  # (H, W, V)
        dist = Categorical(logits=per_cell_logits)
        idx = dist.sample()
        logp = dist.log_prob(idx).sum()
        entropy = dist.entropy().sum()
        return idx, logp, entropy

    def _sample_rules(self):
        """Returns (rule_idx (R, 2, 3, 3), rule_logp scalar, rule_entropy scalar)."""
        self._clamp_tau_buffer()
        logits = self.rule_gen.logits()
        per_cell_logits = (logits / self.gumbel_tau.clamp_min(1e-3)).permute(0, 2, 3, 4, 1)  # (R, 2, 3, 3, V)
        dist = Categorical(logits=per_cell_logits)
        idx = dist.sample()
        logp = dist.log_prob(idx).sum()
        entropy = dist.entropy().sum()
        return idx, logp, entropy

    def _to_rules(self, rule_idx: torch.Tensor) -> List[Rule]:
        arr = rule_idx.detach().cpu().numpy().astype(int)  # (R, 2, 3, 3)
        rules = []
        for i in range(arr.shape[0]):
            rules.append(Rule(
                name=f"r_{i}",
                description="",
                input_grid=arr[i, 0],
                output_grid=arr[i, 1],
                vocabulary=self.vocabulary,
            ))
        return rules

    # ----- one episode -----

    def run_episode(self) -> dict:
        """
        Sample one (grid, rule_set), run the CA + reward composer, return a dict with
        what's needed to compute the policy loss + diagnostics.
        """
        grid_idx, grid_logp, grid_entropy = self._sample_grid()
        rule_idx, rule_logp, rule_entropy = self._sample_rules()

        rules_all = self._to_rules(rule_idx)

        if self.use_action_masking:
            violators = set(rule_violation_indices(rules_all, self.vocabulary))
            rules_active = [r for i, r in enumerate(rules_all) if i not in violators]
        else:
            violators = set()
            rules_active = rules_all

        if self.mode == "rules_only":
            ca = NeuralXORAutomaton(
                dimensions=self.grid_size,
                rules=rules_active,
                max_steps=self.num_ca_steps,
                num_hidden_neuron_types=self.num_hidden_neuron_types,
            )
            log_prob = rule_logp
            entropy = rule_entropy
            grid_np: Optional[np.ndarray] = None
        else:
            grid_np = grid_idx.detach().cpu().numpy().astype(int)
            ca = NeuralXORAutomaton(
                dimensions=self.grid_size,
                rules=rules_active,
                max_steps=self.num_ca_steps,
                base_grid=grid_np,
                num_hidden_neuron_types=self.num_hidden_neuron_types,
            )
            log_prob = grid_logp + rule_logp
            entropy = grid_entropy + rule_entropy

        sess = AutomataSession(ca, self.reward_config)
        reward = sess.run(num_steps=self.num_ca_steps)
        b_out, b_in = 0.0, 0.0
        if self.mode == "grid_and_rules" and grid_np is not None:
            b_out, b_in = self._get_grid_motion_bonuses(grid_np)
            reward = reward + b_out + b_in
            sp = self._switching_penalty(grid_np)
            reward = reward - sp
        else:
            sp = 0.0

        out = {
            "reward": reward,
            "log_prob": log_prob,
            "entropy": entropy,
            "solved": sess.evaluates_xor,
            "n_rules_total": len(rules_all),
            "n_rules_active": len(rules_active),
            "n_violations": len(violators),
            "automaton": ca,
            "grid_switching_penalty": sp,
            "grid_motion_output_bonus": b_out,
            "grid_motion_input_bonus": b_in,
        }
        return out

    def run_episode_grid_pretrain(self) -> dict:
        """
        Sample a base layout: with prob ``pretrain_policy_mix`` from the policy (REINFORCE);
        else the **same** logits / τ as ``_sample_grid`` but with ``no_grad`` and zero
        log-prob so the baseline can update without a policy gradient. The old i.i.d.
        uniform over tokens looked like a flat 1/V histogram (misleading in the viewer).
        """
        dev = self._param_device()
        from_policy = bool(torch.rand(1, device=dev) < self.pretrain_policy_mix)
        if from_policy:
            grid_idx, grid_logp, grid_entropy = self._sample_grid()
        else:
            # Match π(a|s) for visualization / comparable Φ, but do not backprop this draw.
            with torch.no_grad():
                self._clamp_tau_buffer()
                logits = self.grid_gen.logits()
                logits = logits + self._grid_staged_logit_add.to(logits.device).view(self.vocab_size, 1, 1)
                sh = self._static_hidden_logit_mask.to(logits.device)
                cm = (self._curriculum_logit_mask.to(logits.device) * sh).view(self.vocab_size, 1, 1)
                logits = logits + (1.0 - cm) * (-1e9)
                per_cell_logits = (logits / self.gumbel_tau.clamp_min(1e-3)).permute(1, 2, 0)
                dist = Categorical(logits=per_cell_logits)
                grid_idx = dist.sample()
            grid_logp = torch.zeros((), device=dev, dtype=torch.float32)
            grid_entropy = torch.zeros((), device=dev, dtype=torch.float32)

        grid_np = grid_idx.detach().cpu().numpy().astype(int)
        ca = NeuralXORAutomaton(
            dimensions=self.grid_size,
            rules=[],
            max_steps=self.num_ca_steps,
            base_grid=grid_np,
            num_hidden_neuron_types=self.num_hidden_neuron_types,
        )
        n_steps = self.pretrain_num_ca_steps
        sess = AutomataSession(ca, self.reward_config)
        reward = sess.run(num_steps=n_steps)
        b_out, b_in = self._get_grid_motion_bonuses(grid_np)
        reward = reward + b_out + b_in
        sp = self._switching_penalty(grid_np)
        reward = reward - sp

        log_prob = grid_logp
        return {
            "reward": reward,
            "log_prob": log_prob,
            "entropy": grid_entropy,
            "solved": sess.evaluates_xor,
            "n_rules_total": 0,
            "n_rules_active": 0,
            "n_violations": 0,
            "automaton": ca,
            "from_policy": from_policy,
            "grid_switching_penalty": sp,
            "grid_motion_output_bonus": b_out,
            "grid_motion_input_bonus": b_in,
        }

    def _param_device(self):
        return next(self.parameters()).device

    def _switching_penalty(self, grid_np: np.ndarray) -> float:
        """
        Subtract up to ``grid_switching_coef`` from reward when the new base layout
        differs from the previous episode's (Hamming fraction). Encourages smooth
        policy updates instead of wild frame-to-frame jumps.
        """
        g = np.asarray(grid_np, dtype=np.int64)
        c = self.grid_switching_coef
        if c <= 0:
            self._prev_base_grid = g.copy()
            return 0.0
        if self._prev_base_grid is None:
            self._prev_base_grid = g.copy()
            return 0.0
        p = self._prev_base_grid
        if p.shape != g.shape:
            self._prev_base_grid = g.copy()
            return 0.0
        frac = float(np.mean(p != g))
        self._prev_base_grid = g.copy()
        return c * frac

    @staticmethod
    def _output_left_half_dist_to_right_boundary(g: np.ndarray, vocabulary) -> Optional[float]:
        """
        If OUTPUT exists and its rightmost column is strictly left of W//2, return the
        column gap to the right-half boundary (mid - rmax). Otherwise 0 if output exists
        and already reaches the right half; None if no output cells.
        """
        h, w = g.shape
        if w < 1:
            return None
        mid = w // 2
        o = g == vocabulary.OUTPUT_NEURON.value
        if not o.any():
            return None
        rmax = int(np.max(np.where(o)[1]))
        if rmax >= mid:
            return 0.0
        return float(mid - rmax)

    @staticmethod
    def _input_centroid_min_dist_to_truth(g: np.ndarray, vocabulary) -> Optional[float]:
        """Euclidean distance from the input-neuron area centroid to the nearest TRUE/FALSE cell."""
        inp = g == vocabulary.INPUT_NEURON.value
        tmask = (g == vocabulary.TRUE.value) | (g == vocabulary.FALSE.value)
        if not inp.any() or not tmask.any():
            return None
        iy, ix = np.where(inp)
        cy, cx = float(iy.mean()), float(ix.mean())
        ty, tx = np.where(tmask)
        return float(
            min(math.hypot(cy - float(y), cx - float(x)) for y, x in zip(ty, tx))
        )

    def _get_grid_motion_bonuses(self, grid_np: np.ndarray) -> Tuple[float, float]:
        """
        Compare current base layout to ``_prev_base_grid`` (set at end of the previous
        episode). Rewards motion of output rightward (when previously on the left) and
        input toward the XOR truth cells.
        """
        p = self._prev_base_grid
        g = np.asarray(grid_np, dtype=np.int64)
        if p is None or p.shape != g.shape:
            return 0.0, 0.0
        V = self.vocabulary
        b_out, b_in = 0.0, 0.0

        if self._grid_motion_out_coef > 0.0:
            d_prev = self._output_left_half_dist_to_right_boundary(p, V)
            d_cur = self._output_left_half_dist_to_right_boundary(g, V)
            if d_prev is not None and d_cur is not None and d_prev > 0.0:
                # Previous layout had output strictly on the left; credit any reduction
                # in column gap (including fully entering the right half, d_cur == 0).
                improv = d_prev - d_cur
                if improv > 0.0:
                    b_out = self._grid_motion_out_coef * improv

        if self._grid_motion_in_coef > 0.0:
            dp = self._input_centroid_min_dist_to_truth(p, V)
            dg = self._input_centroid_min_dist_to_truth(g, V)
            if dp is not None and dg is not None:
                improv = dp - dg
                if improv > 0.0:
                    b_in = self._grid_motion_in_coef * improv

        return b_out, b_in

    # ----- REINFORCE loss -----

    def policy_loss(
        self,
        log_prob: torch.Tensor,
        reward: float,
        entropy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        REINFORCE loss with running-mean baseline and an optional entropy bonus.

        The baseline is detached state (no gradient) — it just shifts the advantage
        to reduce gradient variance. The entropy bonus `-c_ent * H[π]` (subtracted
        from the loss, since we minimize) keeps the policy from collapsing onto a
        single deterministic sample too early.
        """
        if not bool(self.baseline_initialized):
            self.baseline.fill_(reward)
            self.baseline_initialized.fill_(True)
        baseline_val = float(self.baseline.item())
        advantage = reward - baseline_val
        c = self.reinforce_advantage_clip
        if c > 0.0:
            if advantage > c:
                advantage = c
            elif advantage < -c:
                advantage = -c

        # EMA update of the baseline (after reading it).
        self.baseline.mul_(self.baseline_momentum).add_(reward * (1.0 - self.baseline_momentum))

        loss = -log_prob * advantage
        if entropy is not None and self.entropy_coef > 0:
            loss = loss - self.entropy_coef * entropy
        return loss

    def _anneal_tau(self):
        """
        After each train / pretrain step: either linear τ warmup (start → target) or
        multiplicative decay toward ``gumbel_tau_min``. ``step_count`` is still 0 on the
        first call; we use ``step_count + 1`` as the 1-based index of the step just finished.
        """
        i = int(self.step_count.item()) + 1
        w = self.gumbel_tau_warmup_steps
        target = self.gumbel_tau_target
        start = self.gumbel_tau_start
        if w > 0 and i <= w:
            alpha = min(1.0, i / float(w))
            new_tau = start + (target - start) * alpha
        else:
            cur = float(self.gumbel_tau.item())
            new_tau = max(cur * self.gumbel_tau_decay, self.gumbel_tau_min)
        self.gumbel_tau.fill_(self._clip_tau_scalar(new_tau))

    def train_step(self, optimizer: torch.optim.Optimizer, grad_clip: float = 1.0) -> dict:
        """
        Run one episode and apply one optimizer step. Returns diagnostics.
        """
        info = self.run_episode()
        loss = self.policy_loss(info["log_prob"], info["reward"], entropy=info["entropy"])

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            grad_norm = nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip)
        else:
            grad_norm = torch.tensor(float("nan"))
        optimizer.step()

        self._update_ema()
        self._anneal_tau()
        self.step_count += 1

        info["loss"] = float(loss.item())
        info["baseline"] = float(self.baseline.item())
        info["gumbel_tau"] = float(self.gumbel_tau.item())
        info["grad_norm"] = float(grad_norm)
        info["entropy"] = float(info["entropy"].item())
        return info

    def pretrain_grid_step(self, optimizer: torch.optim.Optimizer, grad_clip: float = 1.0) -> dict:
        """
        One REINFORCE step using `run_episode_grid_pretrain` (grid-only signal). Only
        `grid_gen` parameters receive non-zero gradients; pass an optimizer over
        `policy.grid_gen.parameters()` to avoid stepping unused modules.
        """
        info = self.run_episode_grid_pretrain()
        loss = self.policy_loss(info["log_prob"], info["reward"], entropy=info["entropy"])

        optimizer.zero_grad()
        if loss.requires_grad:
            loss.backward()
            _pg = [p for p in self.grid_gen.parameters() if p.requires_grad]
            if grad_clip is not None and grad_clip > 0:
                grad_norm = nn.utils.clip_grad_norm_(_pg, max_norm=grad_clip)
            else:
                grad_norm = nn.utils.clip_grad_norm_(_pg, max_norm=float("inf"))
            optimizer.step()
        else:
            # Random grid (no policy log-prob): baseline was still updated in policy_loss;
            # no gradient to apply.
            grad_norm = torch.tensor(0.0, device=self._param_device())

        self._update_ema()
        self._anneal_tau()
        self.step_count += 1

        info["loss"] = float(loss.item())
        info["baseline"] = float(self.baseline.item())
        info["gumbel_tau"] = float(self.gumbel_tau.item())
        info["grad_norm"] = float(grad_norm)
        info["entropy"] = float(info["entropy"].item())
        return info
