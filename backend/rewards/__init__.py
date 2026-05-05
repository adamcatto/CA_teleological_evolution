from .grid_rewards import (
    phi_hidden_neuron_count,
    phi_input_output_count,
    phi_input_adjacent_to_truth,
    phi_input_single_cluster,
    phi_input_cluster_at_least_4,
    phi_input_in_left_half,
    phi_input_centroid_near_truth,
    phi_input_separated_from_output,
    phi_topology_five_neuron_bodies,
    phi_topology_input_upper_half,
    phi_topology_output_right_quarter,
    phi_topology_neuron_bbox_fill,
    phi_output_in_right_half,
    phi_all_hidden_types_present,
    phi_neuron_types_presence,
    phi_output_size,
    phi_neuron_size,
    phi_synapse_neuron_adjacency,
    phi_synapse_output_adjacency,
    phi_neuron_connectivity,
    phi_synapse_dag_layers,
    phi_grid_initialization,
    phi_empty_sparsity,
    phi_hidden_neuron_clustering,
    mean_fraction_forbidden_init_tokens,
    mean_fraction_vertical_boundary_init,
    mean_empty_sparsity_band_l1_deficit,
    mean_truth_output_cell_count_deviation,
    mean_fraction_subgrids_with_zero_input,
    mean_input_count_band_violation_nonzero,
    mean_input_euclidean_proximity_to_truth_seeds,
)

from .dynamics_rewards import (
    phi_output_neuron_position_stable,
    phi_rightward_truth_value_flow,
)

from .terminal_rewards import terminal_reward, is_xor_solved

from .rule_rewards import (
    find_truth_value_violators,
    find_output_neuron_creation_violators,
    rule_violation_indices,
    has_center_output_to_truth_rules,
)

from .potential import Phi, Phi_subgrid, PotentialWeights
from .composer import (
    RewardConfig,
    compute_init_reward,
    compute_step_reward,
    compute_rule_set_score,
    rule_active_mask,
)
