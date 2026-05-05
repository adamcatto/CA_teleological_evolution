from typing import Iterable, List

import numpy as np

from .grid import Grid


class Rule:
    """
    Update rule that transforms a specific 3x3 input pattern into a specific 3x3 output pattern.
    Operates on numpy arrays directly so it composes naturally with batched/tensor paths.
    """
    def __init__(
        self,
        name: str,
        description: str,
        input_grid: np.ndarray,
        output_grid: np.ndarray,
        vocabulary=None,
    ):
        assert input_grid.shape == output_grid.shape == (3, 3), "Input and output grids must be 3x3."
        self.name = name
        self.description = description
        self.input_grid = input_grid
        self.output_grid = output_grid
        self.vocabulary = vocabulary

    def apply(self, grid_array: np.ndarray) -> np.ndarray:
        """
        Apply the rule to every 3x3 window of `grid_array`. Match positions in row-major order;
        later matches overwrite earlier ones in their 3x3 footprint (last-writer-wins).
        """
        if grid_array.ndim != 2 or grid_array.shape[0] < 3 or grid_array.shape[1] < 3:
            return grid_array.copy()

        H, W = grid_array.shape
        shape = (H - 2, W - 2, 3, 3)
        strides = grid_array.strides + grid_array.strides
        patches = np.lib.stride_tricks.as_strided(grid_array, shape=shape, strides=strides)

        matches = (patches == self.input_grid).reshape(H - 2, W - 2, 9).all(axis=-1)

        new_grid = grid_array.copy()
        for i, j in np.argwhere(matches):
            new_grid[i:i + 3, j:j + 3] = self.output_grid
        return new_grid


class RuleSet:
    """
    Vectorized application of an ordered list of Rules. Semantics match calling each
    `Rule.apply` in sequence: rules apply in order; within a rule, matches apply in
    row-major order with last-writer-wins per cell.

    Match detection across all rules is fully vectorized via numpy broadcasting; only
    the per-match write loop remains in Python (and only for actual matches).
    """
    def __init__(self, rules: Iterable[Rule]):
        self.rules: List[Rule] = list(rules)
        if self.rules:
            self.inputs = np.stack([r.input_grid for r in self.rules], axis=0)   # (R, 3, 3)
            self.outputs = np.stack([r.output_grid for r in self.rules], axis=0) # (R, 3, 3)
        else:
            self.inputs = np.empty((0, 3, 3), dtype=int)
            self.outputs = np.empty((0, 3, 3), dtype=int)

    def __len__(self):
        return len(self.rules)

    def __iter__(self):
        return iter(self.rules)

    def apply(self, grid_array: np.ndarray) -> np.ndarray:
        """
        Apply all rules sequentially to `grid_array`, returning the new grid.
        Equivalent to: reduce(lambda g, r: r.apply(g), self.rules, grid_array.copy())
        """
        g = grid_array
        for rule in self.rules:
            g = rule.apply(g)
        return g

    def all_match_maps(self, grid_array: np.ndarray) -> np.ndarray:
        """
        Return a (R, H-2, W-2) bool array where entry [r, i, j] is True iff rule r matches
        the 3x3 patch with top-left at (i, j) in the *current* grid_array (no sequencing).
        Useful for analytics / debugging the rule set; not used by `apply` since rules
        must be sequenced for last-writer-wins semantics.
        """
        H, W = grid_array.shape
        shape = (H - 2, W - 2, 3, 3)
        strides = grid_array.strides + grid_array.strides
        patches = np.lib.stride_tricks.as_strided(grid_array, shape=shape, strides=strides)
        # patches: (H', W', 3, 3); inputs: (R, 3, 3) -> compare via broadcasting
        # Result: (R, H', W', 3, 3) -> all over (3, 3)
        return (patches[None] == self.inputs[:, None, None]).reshape(len(self.rules), H - 2, W - 2, 9).all(axis=-1)
