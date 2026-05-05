"""
Per-step dynamics signals normalized to [0, 1].

These read the automaton's history (last two grid states) to score *transitions*,
not absolute states. Phi-based shaping covers absolute-state quality; these capture
desirable motion patterns the policy should learn.
"""
import numpy as np

from ..cellular_automata import NeuralXORAutomaton


__all__ = [
    "phi_output_neuron_position_stable",
    "phi_rightward_truth_value_flow",
]


def phi_output_neuron_position_stable(automaton: NeuralXORAutomaton) -> float:
    """
    Score in [0, 1] for whether output-neuron cells haven't moved between the previous
    and current grid. 1 if every output-neuron cell present in the previous frame is
    still in the same place; 0 if all moved; partial credit otherwise.
    """
    VOC = automaton.vocabulary
    history = automaton.history
    if len(history) < 2:
        return 1.0
    prev, curr = history[-2], history[-1]

    prev_mask = (prev == VOC.OUTPUT_NEURON.value)
    n_prev = int(prev_mask.sum())
    if n_prev == 0:
        # No output neuron yet: no bonus to pay (the policy hasn't produced one). The
        # absence itself is captured by Φ via input_output_count, not double-counted here.
        return 0.0
    still_there = int((prev_mask & (curr == VOC.OUTPUT_NEURON.value)).sum())
    return still_there / n_prev


def phi_rightward_truth_value_flow(automaton: NeuralXORAutomaton) -> float:
    """
    Score in [0, 1] for the fraction of subgrids that have an intermediate-truth-value
    cell positioned strictly between the input and output neurons (column-wise rightward).
    """
    VOC = automaton.vocabulary

    score = 0.0
    n = 0
    for sub in automaton.individual_grids:
        n += 1
        in_pos = np.argwhere(sub == VOC.INPUT_NEURON.value)
        out_pos = np.argwhere(sub == VOC.OUTPUT_NEURON.value)
        itv_pos = np.argwhere(sub == VOC.INTERMEDIATE_TRUTH_VALUE.value)
        if len(in_pos) == 0 or len(out_pos) == 0 or len(itv_pos) == 0:
            continue
        in_col = int(in_pos[:, 1].min())
        out_col = int(out_pos[:, 1].max())
        if in_col >= out_col:
            continue
        between = (itv_pos[:, 1] > in_col) & (itv_pos[:, 1] < out_col)
        if between.any():
            score += 1.0
    if n == 0:
        return 0.0
    return score / n
