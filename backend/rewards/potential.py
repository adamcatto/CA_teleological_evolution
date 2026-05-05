"""
Potential function Φ(automaton) ∈ [0, 1] used for potential-based reward shaping
(Ng, Harada, Russell 1999): per-step shaping reward = Φ(s_{t+1}) - Φ(s_t) is
policy-invariant, dense, and bounded — gradients are well-behaved without changing
the optimal policy.

Φ is a weighted average of normalized structural quality components defined in
`grid_rewards.py`. Each component is in [0, 1] (1 = good); a weighted average over
the 4 stacked subgrids is taken inside each component.
"""
from dataclasses import dataclass
from typing import Optional

from . import grid_rewards as gr


@dataclass
class PotentialWeights:
    """
    Weights for each Φ component. Defaults emphasize the structural invariants that
    the system *must* satisfy (init, single-cell output neuron, input/output count)
    and de-emphasize the noisier adjacency signals.
    """
    init: float = 1.5
    empty_sparsity: float = 2.25
    hidden_neuron_clustering: float = 0.4
    input_output_count: float = 1.0
    input_truth_adjacency: float = 1.2
    input_single_cluster: float = 1.0
    input_min_4_cells: float = 1.5
    # INPUT layout: all INPUT cells in columns < W//2; continuous proximity of input centroid
    # to seeded TRUE/FALSE (complements ``phi_input_adjacent_to_truth``).
    input_left_half: float = 0.0
    input_centroid_near_truth: float = 0.0
    # 1 when INPUT is far from OUTPUT; 0 when overlapping or touching (discourages I/O clumping).
    input_separated_from_output: float = 0.0
    output_right_half: float = 1.5
    hidden_all_types: float = 1.0
    neuron_types_presence: float = 1.5  # φ: INPUT, OUTPUT, 4× hidden, SYNAPSE
    output_size: float = 1.0
    hidden_neuron_count: float = 0.5
    neuron_size: float = 0.3
    synapse_neuron_adjacency: float = 0.6
    synapse_output_adjacency: float = 0.6
    neuron_connectivity: float = 0.5
    # Layout / graph (curriculum stages 5–6): Chebyshev adjacency, synapse-only paths.
    neuron_type_non_adjacency: float = 0.0
    neuron_synapse_path_pairing: float = 0.0
    output_two_hidden_synapse: float = 0.0
    # Sketch / feedforward shape (opt-in; see `phi_topology_*` in `grid_rewards.py`).
    topology_five_neuron_bodies: float = 0.0
    topology_input_upper_half: float = 0.0
    topology_output_right_quarter: float = 0.0
    topology_neuron_bbox_fill: float = 0.0
    # BFS-layer agreement on synapse-linked neuron components (see ``phi_synapse_dag_layers``).
    synapse_dag_layers: float = 0.0

    def total(self) -> float:
        return (
            self.init
            + self.empty_sparsity
            + self.hidden_neuron_clustering
            + self.input_output_count
            + self.input_truth_adjacency
            + self.input_single_cluster
            + self.input_min_4_cells
            + self.input_left_half
            + self.input_centroid_near_truth
            + self.input_separated_from_output
            + self.output_right_half
            + self.hidden_all_types
            + self.neuron_types_presence
            + self.output_size
            + self.hidden_neuron_count
            + self.neuron_size
            + self.synapse_neuron_adjacency
            + self.synapse_output_adjacency
            + self.neuron_connectivity
            + self.neuron_type_non_adjacency
            + self.neuron_synapse_path_pairing
            + self.output_two_hidden_synapse
            + self.topology_five_neuron_bodies
            + self.topology_input_upper_half
            + self.topology_output_right_quarter
            + self.topology_neuron_bbox_fill
            + self.synapse_dag_layers
        )


def Phi(automaton, weights: PotentialWeights = None) -> float:
    """
    Aggregate potential in [0, 1].
    """
    w = weights if weights is not None else PotentialWeights()
    total_w = w.total()
    if total_w == 0.0:
        return 0.0

    components = (
        w.init * gr.phi_grid_initialization(automaton),
        w.empty_sparsity * gr.phi_empty_sparsity(automaton),
        w.hidden_neuron_clustering * gr.phi_hidden_neuron_clustering(automaton),
        w.input_output_count * gr.phi_input_output_count(automaton),
        w.input_truth_adjacency * gr.phi_input_adjacent_to_truth(automaton),
        w.input_single_cluster * gr.phi_input_single_cluster(automaton),
        w.input_min_4_cells * gr.phi_input_cluster_at_least_4(automaton),
        w.input_left_half * gr.phi_input_in_left_half(automaton),
        w.input_centroid_near_truth * gr.phi_input_centroid_near_truth(automaton),
        w.input_separated_from_output * gr.phi_input_separated_from_output(automaton),
        w.topology_five_neuron_bodies * gr.phi_topology_five_neuron_bodies(automaton),
        w.topology_input_upper_half * gr.phi_topology_input_upper_half(automaton),
        w.topology_output_right_quarter * gr.phi_topology_output_right_quarter(automaton),
        w.topology_neuron_bbox_fill * gr.phi_topology_neuron_bbox_fill(automaton),
        w.synapse_dag_layers * gr.phi_synapse_dag_layers(automaton),
        w.output_right_half * gr.phi_output_in_right_half(automaton),
        w.hidden_all_types * gr.phi_all_hidden_types_present(automaton),
        w.neuron_types_presence * gr.phi_neuron_types_presence(automaton),
        w.output_size * gr.phi_output_size(automaton),
        w.hidden_neuron_count * gr.phi_hidden_neuron_count(automaton),
        w.neuron_size * gr.phi_neuron_size(automaton),
        w.synapse_neuron_adjacency * gr.phi_synapse_neuron_adjacency(automaton),
        w.synapse_output_adjacency * gr.phi_synapse_output_adjacency(automaton),
        w.neuron_connectivity * gr.phi_neuron_connectivity(automaton),
        w.neuron_type_non_adjacency * gr.phi_different_neuron_types_non_adjacent(automaton),
        w.neuron_synapse_path_pairing * gr.phi_neuron_pairwise_synapse_reachable(automaton),
        w.output_two_hidden_synapse * gr.phi_output_reaches_two_hidden_types_via_synapse(automaton),
    )
    return float(sum(components) / total_w)


def Phi_subgrid(automaton, subgrid_index: int, weights: Optional[PotentialWeights] = None) -> float:
    """
    Same weighted blend as `Phi`, but each structural component is evaluated on a
    single stacked subgrid (0..3) instead of averaged over all four. Useful for
    analyzing one truth-table case or for subgrid-only objectives.
    """
    w = weights if weights is not None else PotentialWeights()
    total_w = w.total()
    if total_w == 0.0:
        return 0.0
    k = subgrid_index
    components = (
        w.init * gr.phi_grid_initialization(automaton, k),
        w.empty_sparsity * gr.phi_empty_sparsity(automaton, k),
        w.hidden_neuron_clustering * gr.phi_hidden_neuron_clustering(automaton, k),
        w.input_output_count * gr.phi_input_output_count(automaton, k),
        w.input_truth_adjacency * gr.phi_input_adjacent_to_truth(automaton, k),
        w.input_single_cluster * gr.phi_input_single_cluster(automaton, k),
        w.input_min_4_cells * gr.phi_input_cluster_at_least_4(automaton, k),
        w.input_left_half * gr.phi_input_in_left_half(automaton, k),
        w.input_centroid_near_truth * gr.phi_input_centroid_near_truth(automaton, k),
        w.input_separated_from_output * gr.phi_input_separated_from_output(automaton, k),
        w.topology_five_neuron_bodies * gr.phi_topology_five_neuron_bodies(automaton, k),
        w.topology_input_upper_half * gr.phi_topology_input_upper_half(automaton, k),
        w.topology_output_right_quarter * gr.phi_topology_output_right_quarter(automaton, k),
        w.topology_neuron_bbox_fill * gr.phi_topology_neuron_bbox_fill(automaton, k),
        w.synapse_dag_layers * gr.phi_synapse_dag_layers(automaton, k),
        w.output_right_half * gr.phi_output_in_right_half(automaton, k),
        w.hidden_all_types * gr.phi_all_hidden_types_present(automaton, k),
        w.neuron_types_presence * gr.phi_neuron_types_presence(automaton, k),
        w.output_size * gr.phi_output_size(automaton, k),
        w.hidden_neuron_count * gr.phi_hidden_neuron_count(automaton, k),
        w.neuron_size * gr.phi_neuron_size(automaton, k),
        w.synapse_neuron_adjacency * gr.phi_synapse_neuron_adjacency(automaton, k),
        w.synapse_output_adjacency * gr.phi_synapse_output_adjacency(automaton, k),
        w.neuron_connectivity * gr.phi_neuron_connectivity(automaton, k),
        w.neuron_type_non_adjacency * gr.phi_different_neuron_types_non_adjacent(automaton, k),
        w.neuron_synapse_path_pairing * gr.phi_neuron_pairwise_synapse_reachable(automaton, k),
        w.output_two_hidden_synapse * gr.phi_output_reaches_two_hidden_types_via_synapse(automaton, k),
    )
    return float(sum(components) / total_w)
