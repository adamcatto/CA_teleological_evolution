import numpy as np


class Grid:
    def __init__(self, grid: np.ndarray):
        assert grid.ndim == 2, "Grid must be 2D."
        self.grid = grid

    def copy(self) -> "Grid":
        return Grid(self.grid.copy())

    def get_all_3x3_subgrids(self) -> np.ndarray:
        """
        Vectorized extraction of all valid 3x3 subgrids from the main grid.
        Returns shape (N, 3, 3) where N = (rows-2) * (cols-2).
        """
        rows, cols = self.grid.shape
        if rows < 3 or cols < 3:
            return np.empty((0, 3, 3), dtype=self.grid.dtype)

        shape = (rows - 2, cols - 2, 3, 3)
        strides = self.grid.strides + self.grid.strides
        subgrids = np.lib.stride_tricks.as_strided(self.grid, shape=shape, strides=strides)
        return subgrids.reshape(-1, 3, 3)

    def to_one_hot(self, vocab_size: int) -> np.ndarray:
        """
        Return a one-hot encoding with shape (vocab_size, H, W). Float dtype so it can
        cross over to torch without copying semantics.
        """
        H, W = self.grid.shape
        oh = np.zeros((vocab_size, H, W), dtype=np.float32)
        oh[self.grid, np.arange(H)[:, None], np.arange(W)[None, :]] = 1.0
        return oh
