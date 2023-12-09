from typing import Optional
import numpy as np

class MinMaxClusting:

    def __init__(self) -> None:
        pass

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None) -> None:
        pass

    def _minmax(self, x: np.ndarray, grid_idx: np.ndarray):

        dist = np.zeros((len(x), len(grid_idx) + 1), dtype=float)
        for j, idx in enumerate(grid_idx):
            for k, coord in enumerate(x):
                dist[:, j] = np.linalg.norm(x - x[idx])
        min_dist = np.zeros(len(x))
        for i in range(len(x)):
            min_dist[i] = np.min(dist[i])
        return np.argmax(min_dist)
    