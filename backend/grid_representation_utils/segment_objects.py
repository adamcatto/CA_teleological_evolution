import numpy as np
from scipy.ndimage import label, binary_dilation


def extract_component_subgrids(grid: np.ndarray, labeled_grid: np.ndarray):
    """
    Returns dict: {label: subgrid}, where each subgrid is the minimal bounding box
    containing that component, with all other cells zeroed out.
    """
    components = {}

    labels = np.unique(labeled_grid)
    labels = labels[labels != 0]  # exclude background

    for lbl in labels:
        coords = np.argwhere(labeled_grid == lbl)
        min_r, min_c = coords.min(axis=0)
        max_r, max_c = coords.max(axis=0)

        subgrid = grid[min_r:max_r + 1, min_c:max_c + 1].copy()
        mask = (labeled_grid[min_r:max_r + 1, min_c:max_c + 1] == lbl)
        subgrid[~mask] = 0

        components[lbl] = subgrid

    return components


def _segment_with_structure(grid: np.ndarray, value: int, structure: np.ndarray):
    mask = (grid == value)
    labeled_grid, num_features = label(mask, structure=structure)
    components = extract_component_subgrids(grid, labeled_grid)
    return components, labeled_grid, num_features


def segment_neurons(grid: np.ndarray, neuron_value: int):
    """
    Chebyshev <= 2 connectivity for neuron clustering. scipy.ndimage.label only accepts
    3x3 structures, so we dilate the mask by one step first (giving cells within
    Chebyshev distance 2 a path through the dilation), label that, then restrict labels
    back to the original cells.
    """
    mask = (grid == neuron_value)
    structure3 = np.ones((3, 3), dtype=int)
    dilated = binary_dilation(mask, structure=structure3, iterations=1)
    labeled_full, num_features = label(dilated, structure=structure3)
    labeled_grid = np.where(mask, labeled_full, 0)
    components = extract_component_subgrids(grid, labeled_grid)
    return components, labeled_grid, num_features


def segment_synapses(grid: np.ndarray, synapse_value: int):
    """3x3 (Chebyshev <= 1) connectivity for synapses."""
    return _segment_with_structure(grid, synapse_value, np.ones((3, 3), dtype=int))


def segment_truth_value(grid: np.ndarray, truth_value_token: int):
    return _segment_with_structure(grid, truth_value_token, np.ones((3, 3), dtype=int))


def segment_ignore_empty_vertical_boundary(grid: np.ndarray, token_value: int):
    return _segment_with_structure(grid, token_value, np.ones((3, 3), dtype=int))


_NEURON_TOKEN_NAMES = {"INPUT_NEURON", "OUTPUT_NEURON"}


def _segmentation_function_for(token_name: str):
    if token_name.startswith("HIDDEN_NEURON") or token_name in _NEURON_TOKEN_NAMES:
        return segment_neurons
    if token_name == "SYNAPSE":
        return segment_synapses
    if token_name in {"TRUE", "FALSE", "INTERMEDIATE_TRUTH_VALUE"}:
        return segment_truth_value
    if token_name in {"EMPTY", "IGNORE", "VERTICAL_BOUNDARY"}:
        return segment_ignore_empty_vertical_boundary
    raise ValueError(f"Unrecognized token name: {token_name}")


def segment_object(grid_array: np.ndarray, token_value: int, vocabulary):
    """
    Segment a single token type from `grid_array` (a 2D numpy array). The segmentation
    strategy is picked from `vocabulary` based on the token's name.
    """
    token = next((t for t in vocabulary if t.value == token_value), None)
    if token is None:
        raise ValueError(f"Token value {token_value} not in vocabulary {vocabulary}.")
    return _segmentation_function_for(token.name)(grid_array, token_value)
