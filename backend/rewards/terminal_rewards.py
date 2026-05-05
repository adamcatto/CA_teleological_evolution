"""
Dense terminal reward in [0, 1].

Replaces the original sparse "+250k per correct subgrid, +1M bonus if all 4 correct"
with a dense, per-subgrid score that gives partial credit:
  - +1 if the (formerly-OUTPUT_NEURON) cell now holds the intended truth value
  - +0.5 if it holds the *wrong* truth value (still made the right kind of transition)
  - 0 if it's still OUTPUT_NEURON (no transition) or any other token

The session can additionally watch `is_xor_solved(automaton)` to detect terminal success.
"""
import numpy as np

from ..cellular_automata import NeuralXORAutomaton


__all__ = [
    "terminal_reward",
    "is_xor_solved",
]


def _per_subgrid_terminal_score(automaton: NeuralXORAutomaton):
    """Yield (intended_truth_value, current_value_at_output_neuron_cell) for each subgrid."""
    VOC = automaton.vocabulary
    history = automaton.history
    if len(history) < 2:
        return
    prev, curr = history[-2], history[-1]

    intended_evals = automaton.intended_evals_per_grid
    row_slices = automaton._subgrid_row_slices

    for i, s in enumerate(row_slices):
        prev_sub = prev[s, :]
        curr_sub = curr[s, :]

        out_positions = np.argwhere(prev_sub == VOC.OUTPUT_NEURON.value)
        if len(out_positions) == 0:
            yield None, None
            continue
        r, c = out_positions[0]
        intended_truth = VOC.TRUE.value if intended_evals[i] else VOC.FALSE.value
        yield intended_truth, int(curr_sub[r, c])


def terminal_reward(automaton: NeuralXORAutomaton) -> float:
    """
    Score in [0, 1]. Mean over the 4 subgrids of the per-subgrid terminal score.
    """
    VOC = automaton.vocabulary
    truth_values = {VOC.TRUE.value, VOC.FALSE.value}

    total = 0.0
    n = 0
    for intended, current in _per_subgrid_terminal_score(automaton):
        if intended is None:
            continue
        n += 1
        if current == intended:
            total += 1.0
        elif current in truth_values:
            total += 0.5  # right kind of transition, wrong polarity
    if n == 0:
        return 0.0
    return total / n


def is_xor_solved(automaton: NeuralXORAutomaton) -> bool:
    """All 4 subgrids' output-neuron cells now hold the intended XOR truth value."""
    n_correct = 0
    n = 0
    for intended, current in _per_subgrid_terminal_score(automaton):
        if intended is None:
            continue
        n += 1
        if current == intended:
            n_correct += 1
    return n == 4 and n_correct == 4
