"""
Per-component structural scores in [0, 1] — closer to 1 means closer to a valid
NeuralXOR configuration. These are the building blocks of the potential function Φ
in `rewards/potential.py`. Each function returns a scalar averaged over the 4 stacked
subgrids of a NeuralXORAutomaton.
"""
import math
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, label

from ..cellular_automata import NeuralXORAutomaton
from ..vocabulary import hidden_neuron_token_values


__all__ = [
    "phi_hidden_neuron_count",
    "phi_input_output_count",
    "phi_input_adjacent_to_truth",
    "phi_input_single_cluster",
    "phi_input_cluster_at_least_4",
    "phi_input_in_left_half",
    "phi_input_centroid_near_truth",
    "phi_input_separated_from_output",
    "phi_topology_five_neuron_bodies",
    "phi_topology_input_upper_half",
    "phi_topology_output_right_quarter",
    "phi_topology_neuron_bbox_fill",
    "phi_output_in_right_half",
    "phi_all_hidden_types_present",
    "phi_neuron_types_presence",
    "phi_output_size",
    "phi_neuron_size",
    "phi_synapse_neuron_adjacency",
    "phi_synapse_output_adjacency",
    "phi_neuron_connectivity",
    "phi_grid_initialization",
    "phi_empty_sparsity",
    "phi_hidden_neuron_clustering",
    "mean_fraction_forbidden_init_tokens",
    "mean_fraction_vertical_boundary_init",
    "mean_empty_sparsity_band_l1_deficit",
    "mean_truth_output_cell_count_deviation",
    "mean_excess_input_output_cells",
    "mean_fraction_subgrids_with_zero_input",
    "mean_input_count_band_violation_nonzero",
    "mean_input_euclidean_proximity_to_truth_seeds",
    "mean_input_cell_tier_penalty",
    "phi_different_neuron_types_non_adjacent",
    "phi_neuron_pairwise_synapse_reachable",
    "phi_output_reaches_two_hidden_types_via_synapse",
    "phi_synapse_dag_layers",
    "mean_excess_oversized_segment_fraction",
    "mean_excess_dominant_token_fraction",
]


_NEURON_SIZE_RANGE = (9, 100)
# When only ``HIDDEN_NEURON_1`` is active, structural rewards target this many **components**.
_SINGLE_TYPE_HIDDEN_COMPONENT_TARGET = 4
_STRUCT3 = np.ones((3, 3), dtype=bool)


def _num_hidden_types(automaton: NeuralXORAutomaton) -> int:
    n = int(getattr(automaton, "num_hidden_neuron_types", 1) or 1)
    return n if n in (1, 4) else 1


def _hidden_values(automaton: NeuralXORAutomaton) -> List[int]:
    return hidden_neuron_token_values(automaton.vocabulary, _num_hidden_types(automaton))


def _all_neuron_values(automaton: NeuralXORAutomaton) -> List[int]:
    v = automaton.vocabulary
    return [v.INPUT_NEURON.value, v.OUTPUT_NEURON.value] + _hidden_values(automaton)


def _neuron_body_token_values(automaton: NeuralXORAutomaton) -> List[int]:
    """Hand-sketch 'red' modules: `INPUT_NEURON` plus active HIDDEN* only (excludes `OUTPUT_NEURON`)."""
    v = automaton.vocabulary
    return [v.INPUT_NEURON.value] + _hidden_values(automaton)


def _per_subgrid_mean(automaton, fn, subgrid_index: Optional[int] = None) -> float:
    """
    If `subgrid_index` is None, average `fn` over the 4 stacked subgrids.
    If set to 0..3, return `fn` for that subgrid only (for Φ_subgrid / debugging).
    """
    reps = automaton.grid_representations
    if not reps:
        return 0.0
    if subgrid_index is not None:
        if not (0 <= subgrid_index < len(reps)):
            return 0.0
        return float(fn(reps[subgrid_index], automaton.vocabulary))
    vals = [float(fn(rep, automaton.vocabulary)) for rep in reps]
    return float(np.mean(vals))


def phi_hidden_neuron_count(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """
    [0, 1] Hidden structure: with **four** active hidden **types**, fraction of types that
    have ≥1 segmented component. With **one** active type (``HIDDEN_NEURON_1`` only),
    ``min(1, n / 4)`` where *n* is the **number of H1 components** (segmented count), so
    four separate hidden blobs are the target.
    """
    def per(rep, VOC):
        seg = rep["segmented_objects"]
        hv = _hidden_values(automaton)
        if not hv:
            return 0.0
        if _num_hidden_types(automaton) == 1:
            n = int(seg[hv[0]][2])
            return min(1.0, float(n) / float(_SINGLE_TYPE_HIDDEN_COMPONENT_TARGET))
        present = sum(1 for v in hv if seg[v][2] > 0)
        return present / len(hv)
    return _per_subgrid_mean(automaton, per, subgrid_index)


def phi_input_output_count(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """Fraction of subgrids with exactly 1 input neuron and 1 output neuron."""
    def per(rep, VOC):
        seg = rep["segmented_objects"]
        n_in = seg[VOC.INPUT_NEURON.value][2]
        n_out = seg[VOC.OUTPUT_NEURON.value][2]
        return 1.0 if (n_in == 1 and n_out == 1) else 0.0
    return _per_subgrid_mean(automaton, per, subgrid_index)


def phi_input_adjacent_to_truth(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """
    [0, 1] Chebyshev-adjacent (8-neigh) contact between INPUT cells and the seeded
    TRUE/FALSE cells. **1.0** if dilated input touches both truth cells; a much smaller
    **0.04** if it touches only one — strong preference to span both feed sources, not
    a single one.
    """
    def per(rep, VOC) -> float:
        g = rep["grid"]
        inp = g == VOC.INPUT_NEURON.value
        if not inp.any():
            return 0.0
        t = g == VOC.TRUE.value
        f = g == VOC.FALSE.value
        if not t.any() and not f.any():
            return 0.0
        zone = binary_dilation(inp, structure=_STRUCT3)
        hit_t = bool((zone & t).any())
        hit_f = bool((zone & f).any())
        if hit_t and hit_f:
            return 1.0
        if hit_t or hit_f:
            return 0.04
        return 0.0
    return _per_subgrid_mean(automaton, per, subgrid_index)


def phi_input_single_cluster(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """1.0 if INPUT_NEURON segmentation has exactly one connected component."""
    def per(rep, VOC):
        n = rep["segmented_objects"][VOC.INPUT_NEURON.value][2]
        return 1.0 if n == 1 else 0.0
    return _per_subgrid_mean(automaton, per, subgrid_index)


def phi_input_cluster_at_least_4(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """1.0 if there is exactly one input cluster and it contains at least 4 cells (segmentation)."""
    def per(rep, VOC):
        comps, _, n = rep["segmented_objects"][VOC.INPUT_NEURON.value]
        if n != 1:
            return 0.0
        for c in comps.values():
            s = int((c > 0).sum())
            return 1.0 if s >= 4 else 0.0
        return 0.0
    return _per_subgrid_mean(automaton, per, subgrid_index)


# --- Feedforward / sketch topology (Φ, optional weights): maps hand-drawn "5 reds + path + output"
# to existing tokens. TRUE/FALSE are still CA-seeded, not part of the policy body layout.
_STRUCT8I = np.ones((3, 3), dtype=int)


def phi_topology_five_neuron_bodies(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """
    [0, 1] How close the count of 8-connected components of (``INPUT_NEURON`` ∪ HIDDEN\*) is
    to **five** (the sketch: five red neuron modules, excluding the separate output cell).
    1.0 at exactly 5; linear decay: ``max(0, 1 - |n-5|/5)`` per subgrid.
    Works best in **four-type** mode (one INPUT cluster + four hidden clusters). In single-type
    mode you may leave this weight 0 and rely on other terms.
    """
    def per(rep, VOC) -> float:
        g = rep["grid"]
        body = np.zeros_like(g, dtype=bool)
        for t in _neuron_body_token_values(automaton):
            body |= g == t
        if not body.any():
            return 0.0
        _, n = label(body, structure=_STRUCT8I)
        return float(max(0.0, 1.0 - abs(float(n) - 5.0) / 5.0))
    return _per_subgrid_mean(automaton, per, subgrid_index)


def phi_topology_input_upper_half(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """
    1.0 if the `INPUT_NEURON` **centroid** lies in the **upper** half of the subgrid
    (row index < H/2), matching "inputs / sources toward the top" in the sketch. 0 if no input.
    """
    def per(rep, VOC) -> float:
        g = rep["grid"]
        inp = g == VOC.INPUT_NEURON.value
        if not inp.any():
            return 0.0
        iy, _ = np.where(inp)
        cy = float(iy.mean())
        h = g.shape[0]
        return 1.0 if cy < 0.5 * float(h) else 0.0
    return _per_subgrid_mean(automaton, per, subgrid_index)


def phi_topology_output_right_quarter(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """
    1.0 if every `OUTPUT_NEURON` cell lies in the **rightmost ~25%** of columns
    (``col >= W - max(1, W//4)``) — stricter "far right" than `phi_output_in_right_half`. 0 if no output.
    """
    def per(rep, VOC) -> float:
        g = rep["grid"]
        h, w = g.shape
        if w < 1:
            return 0.0
        m = g == VOC.OUTPUT_NEURON.value
        if not m.any():
            return 0.0
        col0 = w - max(1, w // 4)
        cols = np.where(m)[1]
        return 1.0 if bool((cols >= col0).all()) else 0.0
    return _per_subgrid_mean(automaton, per, subgrid_index)


def phi_topology_neuron_bbox_fill(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """
    [0, 1] Mean over neuron-body (INPUT ∪ HIDDEN\*) 8-connected components of
    (pixel count) / (axis-aligned bounding-box area) — high for **solid** blocks, lower
    for thin or hollow paths (a cheap proxy for 'square / blob' red modules, not a full ring test).
    """
    def per(rep, VOC) -> float:
        g = rep["grid"]
        body = np.zeros_like(g, dtype=bool)
        for t in _neuron_body_token_values(automaton):
            body |= g == t
        if not body.any():
            return 0.0
        lab, ncomp = label(body, structure=_STRUCT8I)
        scores = []
        for lid in range(1, ncomp + 1):
            m = lab == lid
            ys, xs = np.where(m)
            bb = (int(xs.max() - xs.min() + 1) * int(ys.max() - ys.min() + 1))
            c = int(m.sum())
            if bb > 0:
                scores.append(c / bb)
        return float(np.mean(scores)) if scores else 0.0
    return _per_subgrid_mean(automaton, per, subgrid_index)


def phi_output_in_right_half(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """
    1.0 if all OUTPUT_NEURON cells lie in the right half of the subgrid (column index >= W//2).
    If there is no output, 0.0.
    """
    def per(rep, VOC) -> float:
        g = rep["grid"]
        _, w = g.shape
        if w < 1:
            return 0.0
        mid = w // 2
        m = g == VOC.OUTPUT_NEURON.value
        if not m.any():
            return 0.0
        cols = np.where(m)[1]
        return 1.0 if bool((cols >= mid).all()) else 0.0
    return _per_subgrid_mean(automaton, per, subgrid_index)


def phi_input_in_left_half(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """
    1.0 if all INPUT_NEURON cells lie in the **left** half (column index < W//2), 0.0 if none.
    """
    def per(rep, VOC) -> float:
        g = rep["grid"]
        _, w = g.shape
        if w < 1:
            return 0.0
        mid = w // 2
        m = g == VOC.INPUT_NEURON.value
        if not m.any():
            return 0.0
        cols = np.where(m)[1]
        return 1.0 if bool((cols < mid).all()) else 0.0
    return _per_subgrid_mean(automaton, per, subgrid_index)


# Gaussian falloff: ~1.0 at the seed; ~0.03 at d≈2 (grid units); far lower than "almost there".
_INPUT_CENTROID_TRUTH_SIGMA = 0.85


def phi_input_separated_from_output(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """
    [0, 1] high when every INPUT is **far** from every OUTPUT (Euclidean, cell centers).
    0.0 for overlap; ~0.0 when 8-adjacent; approaches 1.0 as separation passes several cells.
    """
    def per(rep, VOC) -> float:
        g = rep["grid"]
        ixy = np.argwhere(g == VOC.INPUT_NEURON.value)
        oxy = np.argwhere(g == VOC.OUTPUT_NEURON.value)
        if ixy.size == 0 or oxy.size == 0:
            return 0.0
        d_min = min(
            math.hypot(float(i[0]) - float(o[0]), float(i[1]) - float(o[1]))
            for i in ixy
            for o in oxy
        )
        if d_min <= 0.0:
            return 0.0
        # d_min ~1 → small score; d_min >= ~5 → ~1.0
        return float(min(1.0, (d_min / 5.0) ** 1.35))

    return _per_subgrid_mean(automaton, per, subgrid_index)


def phi_input_centroid_near_truth(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """
    [0, 1] Gaussian in Euclidean distance (grid units) from the **INPUT centroid** to the
    **nearer** of the two hand-placed truth seeds (``truth_value_seed_col`` / ``truth_value_seed_rows``,
    same as init / ``_truth_seed_positions_subgrid``). Steep: near-1 only when the centroid
    is essentially on the sources; a couple of cells away is **much** worse than ``1 - d/d_max`` linear.
    """
    def per(rep, VOC) -> float:
        g = rep["grid"]
        inp = g == VOC.INPUT_NEURON.value
        if not inp.any():
            return 0.0
        iy, ix = np.where(inp)
        cy, cx = float(iy.mean()), float(ix.mean())
        col, r_lo, r_hi = _truth_seed_positions_subgrid(automaton)
        d1 = math.hypot(cy - float(r_lo), cx - float(col))
        d2 = math.hypot(cy - float(r_hi), cx - float(col))
        d = min(d1, d2)
        s = _INPUT_CENTROID_TRUTH_SIGMA
        return float(max(0.0, min(1.0, math.exp(-((d / s) ** 2)))))

    return _per_subgrid_mean(automaton, per, subgrid_index)


def phi_all_hidden_types_present(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """
    1.0 only if each **used** hidden type (1 or 4) has at least one segmented component;
    0.0 if any required type is missing.
    """
    def per(rep, VOC):
        seg = rep["segmented_objects"]
        ok = all(seg[v][2] > 0 for v in _hidden_values(automaton))
        return 1.0 if ok else 0.0
    return _per_subgrid_mean(automaton, per, subgrid_index)


def phi_neuron_types_presence(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """
    [0, 1] Encourages every **neuronal** token and **SYNAPSE** to appear at least once.
    Per subgrid, mean of indicators over ``INPUT_NEURON``, ``OUTPUT_NEURON``, the active
    hidden type(s) (1 or 4), and ``SYNAPSE``. Averaged over the 4 stacked subgrids.
    """
    def per(rep, VOC) -> float:
        g = rep["grid"]
        vals = [VOC.INPUT_NEURON.value, VOC.OUTPUT_NEURON.value] + _hidden_values(automaton) + [VOC.SYNAPSE.value]
        return float(np.mean([float((g == t).any()) for t in vals]))
    return _per_subgrid_mean(automaton, per, subgrid_index)


def phi_output_size(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """Fraction of output neuron clusters that are exactly one cell."""
    def per(rep, VOC):
        comps, _, n = rep["segmented_objects"][VOC.OUTPUT_NEURON.value]
        if n == 0:
            return 0.0
        ok = sum(1 for c in comps.values() if int((c > 0).sum()) == 1)
        return ok / n
    return _per_subgrid_mean(automaton, per, subgrid_index)


def phi_neuron_size(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """Fraction of (non-output) neuron clusters whose cell-count falls in the desired range."""
    lo, hi = _NEURON_SIZE_RANGE
    def per(rep, VOC):
        seg = rep["segmented_objects"]
        sizes = []
        for v in [VOC.INPUT_NEURON.value] + _hidden_values(automaton):
            comps, _, _ = seg[v]
            sizes.extend(int((c > 0).sum()) for c in comps.values())
        if not sizes:
            return 0.0
        ok = sum(1 for s in sizes if lo <= s <= hi)
        return ok / len(sizes)
    return _per_subgrid_mean(automaton, per, subgrid_index)


def _full_mask(rep, value):
    return rep["grid"] == value


def phi_synapse_neuron_adjacency(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """Fraction of synapse cells adjacent (Chebyshev <= 1) to any neuron cell.
    Returns 1 if there are no synapses (vacuously satisfied)."""
    def per(rep, VOC):
        synapse = _full_mask(rep, VOC.SYNAPSE.value)
        n_syn = int(synapse.sum())
        if n_syn == 0:
            return 1.0
        neuron = np.zeros_like(synapse, dtype=bool)
        for v in _all_neuron_values(automaton):
            neuron |= _full_mask(rep, v)
        if not neuron.any():
            return 0.0
        adj_zone = binary_dilation(neuron, structure=_STRUCT3)
        adjacent = int((synapse & adj_zone).sum())
        return adjacent / n_syn
    return _per_subgrid_mean(automaton, per, subgrid_index)


def phi_synapse_output_adjacency(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """Fraction of synapse cells adjacent to an output-neuron cell. 1 if no synapses."""
    def per(rep, VOC):
        synapse = _full_mask(rep, VOC.SYNAPSE.value)
        n_syn = int(synapse.sum())
        if n_syn == 0:
            return 1.0
        out = _full_mask(rep, VOC.OUTPUT_NEURON.value)
        if not out.any():
            return 0.0
        adj_zone = binary_dilation(out, structure=_STRUCT3)
        adjacent = int((synapse & adj_zone).sum())
        return adjacent / n_syn
    return _per_subgrid_mean(automaton, per, subgrid_index)


def phi_neuron_connectivity(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """Fraction of neuron clusters that have at least one adjacent synapse cell.
    Returns 1 if there are no neurons (vacuously)."""
    def per(rep, VOC):
        synapse = _full_mask(rep, VOC.SYNAPSE.value)
        synapse_zone = binary_dilation(synapse, structure=_STRUCT3) if synapse.any() else synapse
        seg = rep["segmented_objects"]
        n_total = 0
        n_connected = 0
        for v in _all_neuron_values(automaton):
            _, labeled, num = seg[v]
            for label_id in range(1, num + 1):
                n_total += 1
                if (labeled == label_id).any() and ((labeled == label_id) & synapse_zone).any():
                    n_connected += 1
        if n_total == 0:
            return 1.0
        return n_connected / n_total
    return _per_subgrid_mean(automaton, per, subgrid_index)


def phi_grid_initialization(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """
    Fraction of subgrids that are initialized correctly:
      - exactly 1 input neuron, 1 output neuron, 2 truth-value tokens
      - output neuron in the right half of the subgrid
    """
    def per(rep, VOC):
        seg = rep["segmented_objects"]
        n_in = seg[VOC.INPUT_NEURON.value][2]
        n_out_comp = seg[VOC.OUTPUT_NEURON.value][2]
        n_truth = seg[VOC.TRUE.value][2] + seg[VOC.FALSE.value][2]
        counts_ok = (n_in == 1 and n_out_comp == 1 and n_truth == 2)
        if not counts_ok:
            return 0.0
        # Coordinates from the full-subgrid labeled grid (components_dict gives bounding boxes only).
        labeled_out = seg[VOC.OUTPUT_NEURON.value][1]
        coords = np.argwhere(labeled_out > 0)
        col = int(coords[0][1])
        return 1.0 if col >= rep["grid"].shape[1] // 2 else 0.5
    return _per_subgrid_mean(automaton, per, subgrid_index)


def mean_fraction_forbidden_init_tokens(automaton: NeuralXORAutomaton) -> float:
    """
    Mean fraction of cells (over the 4 stacked subgrids) that use ``IGNORE`` or
    ``INTERMEDIATE_TRUTH_VALUE`` in the base layout. (``VERTICAL_BOUNDARY`` is penalized
    separately with a larger coefficient — see ``mean_fraction_vertical_boundary_init``.)
    """
    from ..vocabulary import NeuralCellularAutomatonVocabulary as V

    bad = (V.IGNORE.value, V.INTERMEDIATE_TRUTH_VALUE.value)
    if not automaton.grid_representations:
        return 0.0
    fracs = []
    for rep in automaton.grid_representations:
        g = rep["grid"]
        fracs.append(float(np.isin(g, bad).mean()))
    return float(np.mean(fracs)) if fracs else 0.0


def mean_fraction_vertical_boundary_init(automaton: NeuralXORAutomaton) -> float:
    """
    Mean fraction of cells that are ``VERTICAL_BOUNDARY`` in the policy base layout
    (per subgrid, averaged over 4). Should be 0; boundary rows between stacked XOR
    copies are not part of subgrids — any such token here is a policy mistake.
    """
    from ..vocabulary import NeuralCellularAutomatonVocabulary as V

    b = V.VERTICAL_BOUNDARY.value
    if not automaton.grid_representations:
        return 0.0
    fracs = [float((rep["grid"] == b).mean()) for rep in automaton.grid_representations]
    return float(np.mean(fracs)) if fracs else 0.0


# Match ``phi_empty_sparsity`` / ``_phi_empty_banded`` targets.
_SPARSITY_LO = 0.78
_SPARSITY_HI = 0.92


def mean_empty_sparsity_band_l1_deficit(automaton: NeuralXORAutomaton) -> float:
    """
    Mean over 4 subgrids of **L1 distance outside** the EMPTY band: ``max(0, lo−f) +
    max(0, f−hi)`` where ``f`` is EMPTY fraction. Unbounded above 0; pair with a large
    init penalty coefficient (separate from Φ) to crush layouts that are far from the band.
    """
    if not automaton.grid_representations:
        return 0.0
    VOC = automaton.vocabulary
    lo, hi = _SPARSITY_LO, _SPARSITY_HI
    s = 0.0
    for rep in automaton.grid_representations:
        g = rep["grid"]
        f = float((g == VOC.EMPTY.value).mean())
        below = max(0.0, lo - f)
        above = max(0.0, f - hi)
        s += below + above
    return s / 4.0


def phi_hidden_neuron_clustering(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """
    [0, 1] score: each hidden-neuron *type* should form one tight cluster; cluster
    **centroids** should be well separated. Combines per-type largest-CC / cell ratio
    (intra-type cohesion) with normalized minimum pairwise centroid distance.
    """
    _STRUCT8 = np.ones((3, 3), dtype=int)  # Chebyshev ≤1 connectivity

    def per(rep, VOC) -> float:
        g = rep["grid"]
        h, w = g.shape
        diag = float(np.hypot(h, w)) if h > 0 and w > 0 else 1.0
        cohesions = []
        centroids = []
        for v in _hidden_values(automaton):
            mask = g == v
            n = int(mask.sum())
            if n == 0:
                cohesions.append(0.0)
                continue
            lab, ncomp = label(mask, structure=_STRUCT8)
            largest = 0
            for lid in range(1, ncomp + 1):
                largest = max(largest, int((lab == lid).sum()))
            cohesions.append(largest / n)
            yy, xx = np.where(mask)
            centroids.append((float(yy.mean()), float(xx.mean())))

        coh = float(np.mean(cohesions)) if cohesions else 0.0
        if len(centroids) < 2:
            sep = 1.0
        else:
            dmin = 1e30
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    a, b = centroids[i], centroids[j]
                    d = np.hypot(a[0] - b[0], a[1] - b[1])
                    dmin = min(dmin, d)
            # Full score if nearest pair of types is at least ~18% of grid diagonal apart
            sep = float(min(1.0, dmin / (0.18 * max(diag, 1e-6))))

        return 0.55 * coh + 0.45 * sep

    return _per_subgrid_mean(automaton, per, subgrid_index)


def _phi_empty_banded(
    f: float,
    lo: float = _SPARSITY_LO,
    hi: float = _SPARSITY_HI,
    *,
    low_exponent: float = 5.5,
    high_exponent: float = 4.0,
) -> float:
    """
    Score in [0, 1] with maximum on ``f`` (EMPTY fraction) in [lo, hi].
    **Below** ``lo`` uses (f/lo)^low_exponent (high power → very small φ when far under the band).
    **Above** ``hi`` uses a strong power on the upper tail.
    """
    if lo <= f <= hi:
        return 1.0
    if f < lo:
        if lo <= 0:
            return float(f)
        return float((f / lo) ** low_exponent)
    # f > hi
    if hi >= 1.0:
        return max(0.0, 1.0 - (f - hi))
    t = (1.0 - f) / (1.0 - hi) if hi < 1.0 else 0.0
    return float(max(0.0, t) ** high_exponent)


def phi_empty_sparsity(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """
    Encourages most subgrid cells to be EMPTY: score 1.0 when EMPTY fraction is in
    the band; **outside** the band, φ collapses quickly (see ``_phi_empty_banded`` exponents).
    """
    def per(rep, VOC) -> float:
        g = rep["grid"]
        f = float((g == VOC.EMPTY.value).mean())
        return _phi_empty_banded(f, lo=_SPARSITY_LO, hi=_SPARSITY_HI)

    return _per_subgrid_mean(automaton, per, subgrid_index)


def _truth_seed_positions_subgrid(automaton: NeuralXORAutomaton) -> Tuple[int, int, int]:
    """(col, r_lo, r_hi) in subgrid coordinates matching `NeuralXORAutomaton` seeding."""
    col = int(getattr(automaton, "truth_value_seed_col", 2))
    rows = getattr(automaton, "truth_value_seed_rows", None)
    if rows is not None and len(rows) == 2:
        return col, int(rows[0]), int(rows[1])
    h = automaton.grid_representations[0]["grid"].shape[0] if automaton.grid_representations else 8
    r_lo = (31 * h) // 64
    r_hi = (33 * h) // 64
    if r_hi >= h:
        r_hi = h - 1
    if r_lo >= h:
        r_lo = max(0, h - 2)
    if r_hi <= r_lo and h >= 2:
        r_hi = min(h - 1, r_lo + 1)
    return 2, r_lo, r_hi


# Desired raw INPUT count band per subgrid (policy layout); truth cells are always exactly 1 T + 1 F.
_INPUT_N_LO = 2
_INPUT_N_HI = 10


def mean_excess_input_output_cells(automaton: NeuralXORAutomaton) -> float:
    """
    Mean over the 4 subgrids of **output** excess only: ``max(0, n_out - 1)`` in raw cells.
    Input multiplicity is handled by ``mean_input_count_band_violation`` and related init terms.
    """
    if not automaton.grid_representations:
        return 0.0
    VOC = automaton.vocabulary
    s = 0.0
    for rep in automaton.grid_representations:
        g = rep["grid"]
        no = int((g == VOC.OUTPUT_NEURON.value).sum())
        s += max(0, no - 1)
    return s / 4.0


def mean_fraction_subgrids_with_zero_input(automaton: NeuralXORAutomaton) -> float:
    """In ``[0, 1]``: fraction of the 4 stacked subgrids with no INPUT_NEURON cells."""
    if not automaton.grid_representations:
        return 0.0
    VOC = automaton.vocabulary
    k = 0
    for rep in automaton.grid_representations:
        n = int((rep["grid"] == VOC.INPUT_NEURON.value).sum())
        if n == 0:
            k += 1
    return k / 4.0


def mean_input_count_band_violation_nonzero(automaton: NeuralXORAutomaton) -> float:
    """
    Mean per-subgrid penalty for raw INPUT when **n ≥ 1** and outside ``[2, 10]``:
    ``max(0, 2 - n) + max(0, n - 10)`` (0 on subgrids with no INPUT, so "zero INPUT" is handled
    only by ``mean_fraction_subgrids_with_zero_input``).
    """
    if not automaton.grid_representations:
        return 0.0
    VOC = automaton.vocabulary
    s = 0.0
    for rep in automaton.grid_representations:
        g = rep["grid"]
        n = int((g == VOC.INPUT_NEURON.value).sum())
        if n == 0:
            continue
        if n < _INPUT_N_LO:
            s += float(_INPUT_N_LO - n)
        elif n > _INPUT_N_HI:
            s += float(n - _INPUT_N_HI)
    return s / 4.0


def mean_input_euclidean_proximity_to_truth_seeds(automaton: NeuralXORAutomaton) -> float:
    """
    Mean over the 4 subgrids: average over INPUT cells of ``min_d / d_max`` where ``min_d`` is the
    Euclidean distance to the **nearer** of the two hand-placed ``TRUE`` / ``FALSE`` seed positions
    (same rows/col as in ``NeuralXORAutomaton``). 0.0 if there are no INPUT cells on that subgrid.
    """
    if not automaton.grid_representations:
        return 0.0
    VOC = automaton.vocabulary
    col, r_lo, r_hi = _truth_seed_positions_subgrid(automaton)
    acc = 0.0
    for rep in automaton.grid_representations:
        g = rep["grid"]
        h, w = g.shape
        d_max = math.hypot(float(max(h - 1, 1)), float(max(w - 1, 1)))
        if d_max <= 0:
            d_max = 1.0
        inp = np.argwhere(g == VOC.INPUT_NEURON.value)
        if inp.size == 0:
            continue
        dist_sum = 0.0
        for (y, x) in inp:
            d1 = math.hypot(float(y) - float(r_lo), float(x) - float(col))
            d2 = math.hypot(float(y) - float(r_hi), float(x) - float(col))
            dist_sum += min(d1, d2)
        acc += (dist_sum / float(len(inp))) / d_max
    return acc / 4.0


def mean_input_cell_tier_penalty(automaton: NeuralXORAutomaton) -> float:
    """
    Mean over 4 subgrids of a tiered penalty when INPUT is a **single** segmented
    object with **fewer than 4** cells: 3 → 1, 2 → 2, 1 → 3 (harsher when fewer).
    0 if the cluster has ≥4 cells, or if INPUT is not a single component.
    """
    if not automaton.grid_representations:
        return 0.0
    VOC = automaton.vocabulary
    s = 0.0
    for rep in automaton.grid_representations:
        seg = rep["segmented_objects"][VOC.INPUT_NEURON.value]
        comps, _, n = seg
        if n != 1:
            continue
        for c in comps.values():
            ncells = int((c > 0).sum())
            if ncells >= 4:
                p = 0.0
            elif ncells == 3:
                p = 1.0
            elif ncells == 2:
                p = 2.0
            else:
                p = 3.0
            s += p
    return s / 4.0


def _bfs_synapse_reachable(
    syn: np.ndarray, start_syn: np.ndarray, goal_syn: np.ndarray
) -> bool:
    """
    8-neighbor BFS on ``syn`` cells. ``start_syn`` and ``goal_syn`` should be subsets of
    ``syn`` (e.g. syn cells Chebyshev-adjacent to a neuron).
    """
    s0 = (start_syn & syn).astype(bool)
    gz = (goal_syn & syn).astype(bool)
    if not s0.any() or not gz.any():
        return False
    h, w = syn.shape
    q: deque[Tuple[int, int]] = deque()
    vis = np.zeros((h, w), dtype=bool)
    for y, x in np.argwhere(s0):
        y, x = int(y), int(x)
        if not vis[y, x]:
            vis[y, x] = True
            q.append((y, x))
    while q:
        y, x = q.popleft()
        if gz[y, x]:
            return True
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and syn[ny, nx] and not vis[ny, nx]:
                    vis[ny, nx] = True
                    q.append((ny, nx))
    return False


def _syn_starts_to_targets(syn: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    """A synapse-only path from `a` to `b` (Chebyshev-dilated 8-nb of each region)."""
    s_start = (binary_dilation(a, structure=_STRUCT3) & syn).astype(bool)
    t_goal = (binary_dilation(b, structure=_STRUCT3) & syn).astype(bool)
    return _bfs_synapse_reachable(syn, s_start, t_goal)


def _iter_neuron_component_masks(rep, automaton: NeuralXORAutomaton) -> List[Tuple[int, np.ndarray]]:
    out: List[Tuple[int, np.ndarray]] = []
    seg = rep["segmented_objects"]
    for v in _all_neuron_values(automaton):
        comps, labeled, n = seg[v]
        for lid in range(1, n + 1):
            m = labeled == lid
            if m.any():
                out.append((v, m))
    return out


def phi_different_neuron_types_non_adjacent(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """
    1.0 if no two **different** neuron token types are Chebyshev-8-neighbors; else 0.0
    (synapse and non-neuron cells ignored).
    """
    def per(rep, VOC) -> float:
        g = rep["grid"]
        h, w = g.shape
        nv = _all_neuron_values(automaton)
        sset = set(nv)
        for y in range(h):
            for x in range(w):
                t = int(g[y, x])
                if t not in sset:
                    continue
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, y + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            u = int(g[ny, nx])
                            if u in sset and u != t:
                                return 0.0
        return 1.0

    return _per_subgrid_mean(automaton, per, subgrid_index)


def phi_neuron_pairwise_synapse_reachable(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """
    For each **segmented neuron** component, require a synapse-only 8-path to at least one
    **other** neuron component. 1.0 if all such components are paired; 0.0 if none.
    If there is only one component total, 0.0.
    """
    def per(rep, VOC) -> float:
        g = rep["grid"]
        syn = g == VOC.SYNAPSE.value
        comps = _iter_neuron_component_masks(rep, automaton)
        n = len(comps)
        if n == 0:
            return 0.0
        if n == 1:
            return 0.0
        ok = 0
        for i in range(n):
            _, mi = comps[i]
            paired = False
            for j in range(n):
                if i == j:
                    continue
                _, mj = comps[j]
                if _syn_starts_to_targets(syn, mi, mj):
                    paired = True
                    break
            if paired:
                ok += 1
        return ok / n

    return _per_subgrid_mean(automaton, per, subgrid_index)


def phi_output_reaches_two_hidden_types_via_synapse(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """
    If four hidden **types** are used: 1.0 if OUTPUT is synapse-reachable to ≥2 **distinct
    types** among those present. If only one hidden type is used: same score but counting
    **≥2 distinct segmented components** of that type reachable from OUTPUT via synapse.
    Partial credit ``0.5 * c`` for c∈{0,1} met targets.
    """
    def per(rep, VOC) -> float:
        g = rep["grid"]
        syn = g == VOC.SYNAPSE.value
        out = g == VOC.OUTPUT_NEURON.value
        if not out.any():
            return 0.0
        hv = _hidden_values(automaton)
        if len(hv) >= 2:
            n_types = 0
            for v in hv:
                hm = g == v
                if not hm.any():
                    continue
                if _syn_starts_to_targets(syn, out, hm):
                    n_types += 1
            return 1.0 if n_types >= 2 else 0.5 * float(n_types)
        v = hv[0]
        seg = rep["segmented_objects"]
        comps, labeled, ncomp = seg[v]
        n_reach = 0
        for lid in range(1, ncomp + 1):
            hm = labeled == lid
            if not hm.any():
                continue
            if _syn_starts_to_targets(syn, out, hm):
                n_reach += 1
        return 1.0 if n_reach >= 2 else 0.5 * float(n_reach)

    return _per_subgrid_mean(automaton, per, subgrid_index)


def _synapse_dag_layers_score_subgrid(rep, automaton: NeuralXORAutomaton) -> float:
    """
    [0, 1] feedforward / layered score on a single subgrid (see public φ docstring).
    """
    g = rep["grid"]
    VOC = automaton.vocabulary
    syn = (g == VOC.SYNAPSE.value).astype(bool)
    pairs = _iter_neuron_component_masks(rep, automaton)
    n = len(pairs)
    in_ids = [i for i, (t, _) in enumerate(pairs) if t == VOC.INPUT_NEURON.value]
    if not in_ids:
        return 0.0
    out_ids = {i for i, (t, _) in enumerate(pairs) if t == VOC.OUTPUT_NEURON.value}
    if not out_ids:
        return 0.0

    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        mi = pairs[i][1]
        for j in range(i + 1, n):
            mj = pairs[j][1]
            if _syn_starts_to_targets(syn, mi, mj):
                adj[i].append(j)
                adj[j].append(i)

    dist = [-1] * n
    q: deque[int] = deque()
    for s in in_ids:
        if dist[s] < 0:
            dist[s] = 0
            q.append(s)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)

    if not any(dist[i] >= 0 for i in out_ids):
        return 0.0

    edges: List[Tuple[int, int]] = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    if not edges:
        return 0.0

    acc = 0.0
    for u, v in edges:
        du, dv = dist[u], dist[v]
        if du < 0 or dv < 0:
            acc += 0.0
            continue
        lo, hi = (du, dv) if du <= dv else (dv, du)
        if hi - lo == 1:
            acc += 1.0
        elif lo == hi:
            acc += 0.0
        else:
            acc += 0.35
    return float(acc / len(edges))


def phi_synapse_dag_layers(automaton: NeuralXORAutomaton, subgrid_index: Optional[int] = None) -> float:
    """
    [0, 1] **Feedforward / layered** synapse graph among neuron *components* (8-connected
    `INPUT` / `HIDDEN*`, `OUTPUT` as separate nodes). Undirected edge iff a **synapse-only**
    8-path connects the two components (see ``_syn_starts_to_targets``). Unweighted
    BFS from **all** `INPUT` components gives each node a *layer* distance from the source.
    Over unique edges ``(u, v)``, mean reward: **1.0** if BFS layer heights differ by 1, **0.0** if
    they match (lateral in the same BFS level — parallel shortcuts), **0.35** if they differ by
    2+ (skip connections). 0.0 if there is no `INPUT` or `OUTPUT` component, BFS does not reach
    any `OUTPUT`, or the component graph has no edges.

    This does *not* prove global acyclicity; it rewards **agreement** of synapse-chosen
    component adjacency with a shortest-path layering from the input, approximating
    "inputs → hiddens → output" in the sketch.
    """
    def per(rep, VOC) -> float:
        return _synapse_dag_layers_score_subgrid(rep, automaton)
    return _per_subgrid_mean(automaton, per, subgrid_index)


def mean_truth_output_cell_count_deviation(automaton: NeuralXORAutomaton) -> float:
    """
    Mean over the 4 subgrids of ``|#OUTPUT - 1| + |#TRUE - 1| + |#FALSE - 1|`` in **raw cells**
    (not segmentation). 0 means exactly one OUTPUT_NEURON, one TRUE, and one FALSE per subgrid.
    """
    if not automaton.grid_representations:
        return 0.0
    s = 0.0
    for rep in automaton.grid_representations:
        g = rep["grid"]
        VOC = automaton.vocabulary
        s += abs(int((g == VOC.OUTPUT_NEURON.value).sum()) - 1)
        s += abs(int((g == VOC.TRUE.value).sum()) - 1)
        s += abs(int((g == VOC.FALSE.value).sum()) - 1)
    return s / 4.0


# --- Init penalties: limit huge blobs (segmentation) and any single token dominating raw counts ---

_SEGMENT_BLOB_FRACTION_MAX = 0.02
_TOKEN_RAW_FRACTION_MAX = 0.05


def _excluded_token_values_for_oversized_segments(VOC):
    return {
        int(VOC.EMPTY.value),
        int(VOC.IGNORE.value),
        int(VOC.VERTICAL_BOUNDARY.value),
    }


def mean_excess_oversized_segment_fraction(automaton: NeuralXORAutomaton) -> float:
    """
    Mean over 4 subgrids of: for each **segmented** connected component (all token
    types except EMPTY / IGNORE / VERTICAL_BOUNDARY), add ``max(0, a/T − 0.02)`` where
    `a` is cell count in that component and `T` the subgrid area. 0 is ideal.
    """
    if not automaton.grid_representations:
        return 0.0
    VOC = automaton.vocabulary
    skip = _excluded_token_values_for_oversized_segments(VOC)
    s = 0.0
    for rep in automaton.grid_representations:
        g = rep["grid"]
        h, w = g.shape
        tot = float(h * w)
        if tot <= 0:
            continue
        sub = 0.0
        seg = rep["segmented_objects"]
        for token_val, pack in seg.items():
            if int(token_val) in skip:
                continue
            comps = pack[0] if pack else {}
            for c in comps.values():
                a = int((c > 0).sum())
                frac = a / tot
                if frac > _SEGMENT_BLOB_FRACTION_MAX:
                    sub += frac - _SEGMENT_BLOB_FRACTION_MAX
        s += sub
    return s / 4.0


def mean_excess_dominant_token_fraction(automaton: NeuralXORAutomaton) -> float:
    """
    Mean over 4 subgrids of: for each raw vocabulary value **except EMPTY**, add
    ``max(0, n/T − 0.05)`` where `n` is the count of that token. 0 is ideal.
    (EMPTY routinely occupies most of the grid, so it is excluded.)
    """
    if not automaton.grid_representations:
        return 0.0
    VOC = automaton.vocabulary
    ev = int(VOC.EMPTY.value)
    s = 0.0
    for rep in automaton.grid_representations:
        g = rep["grid"]
        h, w = g.shape
        tot = float(h * w)
        if tot <= 0:
            continue
        sub = 0.0
        for item in VOC:
            t = int(item.value)
            if t == ev:
                continue
            n = int((g == t).sum())
            frac = n / tot
            if frac > _TOKEN_RAW_FRACTION_MAX:
                sub += frac - _TOKEN_RAW_FRACTION_MAX
        s += sub
    return s / 4.0
