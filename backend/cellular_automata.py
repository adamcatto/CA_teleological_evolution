from typing import Iterable, Tuple, Optional
from collections import deque

import numpy as np

from .grid import Grid
from .rule import Rule, RuleSet
from .vocabulary import NeuralCellularAutomatonVocabulary, Vocabulary
from .grid_representation_utils.segment_objects import segment_object


class CellularAutomaton:
    def __init__(
        self,
        grid: Grid,
        rules: Iterable[Rule],
        vocabulary: Optional[Vocabulary] = None,
        max_steps: int = -1,
        history_length: int = 128,
    ):
        self.grid = grid
        self.rule_set = rules if isinstance(rules, RuleSet) else RuleSet(rules)
        self.rules = self.rule_set
        self.vocabulary = vocabulary
        self.max_steps = max_steps
        self.history_length = history_length
        self.current_step = 0

        self.history = deque(maxlen=history_length)
        self.history.append(self.grid.grid.copy())

    def step(self):
        new_array = self.rule_set.apply(self.grid.grid)
        self.grid = Grid(new_array)
        self.history.append(new_array.copy())
        self._step_hooks()
        self.current_step += 1

    def _step_hooks(self):
        """Override in subclasses to compute representations / metrics after each step."""
        pass


class NeuralXORAutomaton(CellularAutomaton):
    def __init__(
        self,
        dimensions: Tuple[int, int],
        rules: Iterable[Rule],
        vocabulary: NeuralCellularAutomatonVocabulary = NeuralCellularAutomatonVocabulary,
        max_steps: int = 128,
        history_length: int = 128,
        base_grid: Optional[np.ndarray] = None,
        num_hidden_neuron_types: int = 1,
    ):
        """
        Args:
            dimensions: (H, W) of one stacked subgrid (the full automaton grid will be
                4*H + 9 rows tall — 3 boundary rows between each pair of copies).
            rules: iterable of Rule (or a RuleSet) to apply each step.
            vocabulary: Enum class of cell tokens.
            max_steps, history_length: as in `CellularAutomaton`.
            base_grid: optional (H, W) numpy array of vocabulary tokens that will be
                used as the *layout* replicated into each of the 4 stacked subgrids.
                If None, the layout starts as all EMPTY. Truth-value seeds are written
                on top of this layout in either case, so the policy can place neurons
                / synapses anywhere except the four truth-value seed positions
                (which always end up holding the seeded values regardless).
            num_hidden_neuron_types: use only `HIDDEN_NEURON_1` (1) or all four hidden
                classes (4). Affects potential terms and (via the policy) sampling masks.
        """
        # `vocabulary` is the Enum *class* (so we can iterate members and look up by name).
        self.vocabulary = vocabulary
        nh = int(num_hidden_neuron_types)
        if nh not in (1, 4):
            raise ValueError("num_hidden_neuron_types must be 1 or 4")
        self.num_hidden_neuron_types = nh
        self.vocabulary_token_values = set(item.value for item in self.vocabulary)
        self.vocabulary_token_names = set(item.name for item in self.vocabulary)

        if base_grid is None:
            base = np.full(dimensions, vocabulary.EMPTY.value, dtype=int)
        else:
            assert base_grid.shape == tuple(dimensions), \
                f"base_grid shape {base_grid.shape} != dimensions {dimensions}"
            base = base_grid.astype(int, copy=True)
        _grid = Grid(base)

        super().__init__(_grid, rules, vocabulary, max_steps, history_length)

        for rule in self.rule_set:
            assert np.isin(rule.input_grid, list(self.vocabulary_token_values)).all(), \
                "Rule input grid contains invalid vocabulary tokens."
            assert np.isin(rule.output_grid, list(self.vocabulary_token_values)).all(), \
                "Rule output grid contains invalid vocabulary tokens."

        # Stack 4x copies of the grid vertically with VERTICAL_BOUNDARY rows between them
        # so the four XOR truth-table cases run in parallel under the same rule set.
        # Boundary rows do not match any rule input pattern, so rules never propagate across copies.
        empty = vocabulary.EMPTY.value
        boundary = vocabulary.VERTICAL_BOUNDARY.value
        H, W = dimensions
        boundary_block = np.array([[empty] * W, [boundary] * W, [empty] * W], dtype=int)
        copies = [base.copy() for _ in range(4)]

        stacked = copies[0]
        for c in copies[1:]:
            stacked = np.vstack([stacked, boundary_block, c])
        self.grid = Grid(stacked)

        # Seed the four input-pair locations: (T,F), (F,T), (T,T), (F,F)
        # First two should evaluate True under XOR; last two False.
        offsets = [0, H + 3, 2 * (H + 3), 3 * (H + 3)]
        seed_pairs = [
            (vocabulary.TRUE.value, vocabulary.FALSE.value),
            (vocabulary.FALSE.value, vocabulary.TRUE.value),
            (vocabulary.TRUE.value, vocabulary.TRUE.value),
            (vocabulary.FALSE.value, vocabulary.FALSE.value),
        ]
        # Truth-input cells (col 2): originally rows 31 & 33 on a 64-tall subgrid. Scale
        # with H so small grids (e.g. 32×32) stay in [0, H) per copy.
        r_lo = (31 * H) // 64
        r_hi = (33 * H) // 64
        if r_hi >= H:
            r_hi = H - 1
        if r_lo >= H:
            r_lo = max(0, H - 2)
        if r_hi <= r_lo and H >= 2:
            r_hi = min(H - 1, r_lo + 1)
        for off, (a, b) in zip(offsets, seed_pairs):
            self.grid.grid[off + r_lo, 2] = a
            self.grid.grid[off + r_hi, 2] = b

        # Seeded TRUE/FALSE columns (per subgrid local coords) — for reward distance to input.
        self.truth_value_seed_col = 2
        self.truth_value_seed_rows = (r_lo, r_hi)

        self._subgrid_row_slices = self._compute_subgrid_row_slices(H)
        self.individual_grids = self._get_individual_grids()

        self.intended_evals_per_grid = [True, True, False, False]
        self.intended_output_neuron_to_truth_value_mappings_per_subgrid = [
            {vocabulary.OUTPUT_NEURON.value: vocabulary.TRUE.value},
            {vocabulary.OUTPUT_NEURON.value: vocabulary.TRUE.value},
            {vocabulary.OUTPUT_NEURON.value: vocabulary.FALSE.value},
            {vocabulary.OUTPUT_NEURON.value: vocabulary.FALSE.value},
        ]

        # Reset history so it starts from the post-stacking, post-seeding grid.
        self.history.clear()
        self.history.append(self.grid.grid.copy())

        self.grid_representations = self.generate_representations()

    def _compute_subgrid_row_slices(self, H: int):
        """Row slices for each of the 4 stacked copies (excluding the 3-row boundary blocks)."""
        slices = []
        cursor = 0
        for _ in range(4):
            slices.append(slice(cursor, cursor + H))
            cursor += H + 3  # H rows of subgrid + 3-row boundary block
        return slices

    def _get_individual_grids(self):
        return [self.grid.grid[s, :] for s in self._subgrid_row_slices]

    def generate_representations(self) -> list:
        """
        For each of the 4 stacked subgrids, segment objects per token. Each subgrid is
        segmented independently — the boundary rows ensure no cross-subgrid leakage.
        """
        grid_representations = []
        for s in self._subgrid_row_slices:
            subgrid = self.grid.grid[s, :]
            segmented_objects = {}
            for token_value in self.vocabulary_token_values:
                segmented_objects[token_value] = segment_object(subgrid, token_value, self.vocabulary)
            grid_representations.append({
                "grid": subgrid,
                "segmented_objects": segmented_objects,
            })
        self.grid_representations = grid_representations
        return grid_representations

    def _step_hooks(self):
        self.individual_grids = self._get_individual_grids()
        self.generate_representations()
