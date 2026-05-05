"""
Differentiable cellular automaton operating on soft one-hot grid tensors.

This is the torch-side counterpart of `Rule.apply` / `RuleSet.apply` (numpy). It exists
so gradients can flow from a scalar reward back through the CA simulation to the
generator networks that produced the initial grid and rule set.

Soft semantics (relaxation of the hard numpy CA):
  - G has shape (V, H, W), each (i, j) a distribution over V tokens.
  - Each rule r: (rule_in[r], rule_out[r]), shape (V, 3, 3) each.
  - Match score for patch top-lefted at (i, j) under rule r:
        m_r(i, j) = prod_(di, dj) sum_v rule_in[r, v, di, dj] * G[v, i+di, j+dj]
    With one-hot G and one-hot rule_in this is exactly the indicator that the patch
    matches the rule's input pattern.
  - Update aggregates over all rules and all patches that touch a given cell:
        contribution[v, i', j'] = sum_{r, P containing (i', j')} m_r(P) * rule_out[r, v, di, dj]
        weight[i', j']          = sum_{r, P containing (i', j')} m_r(P)
        blend[i', j']           = clamp(weight, 0, 1)
        new_G[v, i', j']        = (1 - blend) * G[v, i', j'] + blend * contribution / max(weight, eps)

Divergence from the hard CA: when multiple rules / multiple match positions touch the
same cell with conflicting outputs, this averages weighted by match score rather than
"last-writer-wins". For training the soft path, use this; for evaluation / interpretation,
convert rules with `convert_tensor_to_rules` and run the numpy `RuleSet.apply`.
"""
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .grid import Grid
from .rule import Rule


def grid_to_one_hot(grid: Grid, vocab_size: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Convert a numpy `Grid` to a (V, H, W) one-hot torch tensor."""
    arr = grid.grid
    H, W = arr.shape
    oh = torch.zeros(vocab_size, H, W, dtype=dtype, device=device)
    flat_idx = torch.from_numpy(arr.astype(np.int64))
    oh.scatter_(0, flat_idx.unsqueeze(0), 1.0)
    return oh


def one_hot_to_grid(G: torch.Tensor) -> Grid:
    """Argmax a (V, H, W) tensor back to a numpy `Grid`."""
    arr = G.argmax(dim=0).detach().cpu().numpy().astype(int)
    return Grid(arr)


def split_rule_tensor(rule_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(R, V, 2, 3, 3) -> rule_in (R, V, 3, 3), rule_out (R, V, 3, 3)."""
    assert rule_tensor.ndim == 5 and rule_tensor.shape[2] == 2 and rule_tensor.shape[-2:] == (3, 3)
    return rule_tensor[:, :, 0], rule_tensor[:, :, 1]


def convert_tensor_to_rules(
    rule_tensor: torch.Tensor,
    vocabulary,
    name_prefix: str = "rule",
    skip_predicate=None,
) -> List[Rule]:
    """
    Convert a generated rule tensor into a list of `Rule` objects suitable for the
    numpy CA.

    Args:
        rule_tensor: shape (R, V, 2, 3, 3). Need not be one-hot — argmax is taken over V.
        vocabulary: the vocabulary Enum class (passed through to each Rule for reference).
        name_prefix: rule names will be f"{name_prefix}_{i}".
        skip_predicate: optional callable `(input_grid, output_grid) -> bool`. Rules for
            which this returns True are dropped. Useful e.g. to drop identity rules
            (input == output) or rules that match nothing meaningful.

    Returns:
        List[Rule] of length <= R.
    """
    assert rule_tensor.ndim == 5 and rule_tensor.shape[2] == 2 and rule_tensor.shape[-2:] == (3, 3)
    indices = rule_tensor.argmax(dim=1).detach().cpu().numpy().astype(int)  # (R, 2, 3, 3)

    rules: List[Rule] = []
    for r in range(indices.shape[0]):
        in_grid = indices[r, 0]
        out_grid = indices[r, 1]
        if skip_predicate is not None and skip_predicate(in_grid, out_grid):
            continue
        rules.append(Rule(
            name=f"{name_prefix}_{r}",
            description="",
            input_grid=in_grid,
            output_grid=out_grid,
            vocabulary=vocabulary,
        ))
    return rules


class TensorCellularAutomaton(nn.Module):
    """
    Differentiable CA. See module docstring for soft semantics.

    Usage:
        ca = TensorCellularAutomaton(vocab_size=V, num_steps=16)
        G_final, history = ca(G_init, rule_in, rule_out)
        # G_final: (V, H, W), history: list of (V, H, W) tensors of length num_steps + 1
    """

    def __init__(self, vocab_size: int, num_steps: int = 16, eps: float = 1e-8):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.eps = eps

    def step_soft(
        self,
        G: torch.Tensor,
        rule_in: torch.Tensor,
        rule_out: torch.Tensor,
        rule_active: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        One soft CA step.

        Args:
            G:           (V, H, W)  soft one-hot grid (cells sum to 1 over V).
            rule_in:     (R, V, 3, 3)  soft one-hot rule input patterns.
            rule_out:    (R, V, 3, 3)  soft one-hot rule output patterns.
            rule_active: (R,)  optional [0, 1] gating per rule. If provided, match
                              scores are scaled by it (so a rule with weight 0 is inert).

        Returns:
            new G: (V, H, W).
        """
        V, H, W = G.shape
        assert V == self.vocab_size, f"G has V={V}, expected {self.vocab_size}"
        assert rule_in.shape == rule_out.shape
        R = rule_in.shape[0]

        if R == 0 or H < 3 or W < 3:
            return G

        # G_patches[v, di, dj, i, j] = G[v, i+di, j+dj]; shape (V, 3, 3, H', W')
        G_unfold = F.unfold(G.unsqueeze(0), kernel_size=3).squeeze(0)  # (V*9, L)
        H_p, W_p = H - 2, W - 2
        L = H_p * W_p
        G_patches = G_unfold.view(V, 3, 3, H_p, W_p)

        # Per-cell agreement: probability cell at (di, dj) of patch (i, j) matches rule r's symbol
        agreement = torch.einsum('vdeij,rvde->rdeij', G_patches, rule_in)
        agreement = agreement.clamp_min(self.eps)

        # Patch match score per rule (product over 9 cells, in log space)
        match_score = torch.log(agreement).sum(dim=(1, 2)).exp()  # (R, H', W')
        if rule_active is not None:
            match_score = match_score * rule_active.view(R, 1, 1)

        # Sum contributions across rules, per-patch
        patch_contrib = torch.einsum('rij,rvde->vdeij', match_score, rule_out)  # (V, 3, 3, H', W')
        patch_weight = match_score.sum(dim=0)  # (H', W')

        # Fold patches back to (V, H, W). F.fold sums overlapping contributions.
        contrib_flat = patch_contrib.reshape(V * 9, L).unsqueeze(0)  # (1, V*9, L)
        contribution = F.fold(contrib_flat, output_size=(H, W), kernel_size=3).squeeze(0)  # (V, H, W)

        # Total weight at each cell: each patch's weight is "spread" across its 3x3 footprint.
        weight_flat = patch_weight.unsqueeze(0).expand(9, H_p, W_p).reshape(1, 9, L)
        weight_total = F.fold(weight_flat, output_size=(H, W), kernel_size=3).squeeze(0).squeeze(0)  # (H, W)

        blend = weight_total.clamp(min=0.0, max=1.0).unsqueeze(0)  # (1, H, W)
        weighted_output = contribution / weight_total.clamp_min(self.eps).unsqueeze(0)
        new_G = (1.0 - blend) * G + blend * weighted_output
        return new_G

    def forward(
        self,
        G_init: torch.Tensor,
        rule_in: torch.Tensor,
        rule_out: torch.Tensor,
        num_steps: Optional[int] = None,
        rule_active: Optional[torch.Tensor] = None,
        return_history: bool = True,
    ):
        """
        Run `num_steps` soft CA updates starting from `G_init`.

        Args:
            G_init:    (V, H, W)
            rule_in:   (R, V, 3, 3)
            rule_out:  (R, V, 3, 3)
            num_steps: defaults to self.num_steps
            rule_active: (R,) optional rule gating
            return_history: if True, return (G_final, [G_init, G_1, ..., G_T])

        Returns:
            G_final, [optional history]
        """
        if num_steps is None:
            num_steps = self.num_steps
        G = G_init
        history = [G] if return_history else None
        for _ in range(num_steps):
            G = self.step_soft(G, rule_in, rule_out, rule_active=rule_active)
            if return_history:
                history.append(G)
        return (G, history) if return_history else G
