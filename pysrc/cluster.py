from typing import Optional
import numpy as np
from rich.progress import track

from .utils._cluster import _euclidean_distance, _euclidean_distance_period

class MinMaxClustering:
    """MinMaxClustering Class"""

    def __init__(self, n_cluster: int, init_grid_idx: list, period: Optional[list] = None) -> None:

        self.n_cluster = n_cluster
        self.selected_idx_ = np.full(n_cluster, -1, dtype=int)
        self.selected_idx_[:len(init_grid_idx)] = init_grid_idx
        self._min_dist = None
        self.period = period
        if self.period is not None:
            self._distance = _euclidean_distance_period
        else:
            self._distance = _euclidean_distance


    @property
    def _ngrid(self):
        return np.sum(self.selected_idx_ != -1)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None) -> None:
        """Fit the data."""

        self._min_dist = np.full((len(X)), np.inf, dtype=float)
        for i, grid in track(enumerate(self.selected_idx_[1:]), total=self.n_cluster):
            # Calculate distance to the last grid
            dist = self._distance(X, X[self.selected_idx_[i]], self.period)
            self._min_dist[self._min_dist > dist] = dist[self._min_dist > dist]
            self._min_dist[self.selected_idx_[:self._ngrid]]  = -np.inf  # Exclude selected grid
            if grid != -1:
                # has been assigned
                continue
            self.selected_idx_[i + 1] = np.argmax(self._min_dist)  # Updata the index of the current grid

        self._post_init(X, y)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data."""
        if X.shape != self.support_.shape:
            raise ValueError("X has a different shape than during the fitting.")
        return X[self.support_]

    def _post_init(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.support_ = np.full(X.shape, False)
        self.support_[self.selected_idx_] = True
    