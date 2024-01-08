from typing import Optional
import numpy as np
from rich.progress import track

from .utils._cluster import _euclidean_distance, _euclidean_distance_period

class NearestNeighborClustering:
    """NearestNeighborClustering Class
    Assign descriptor to its nearest grid."""

    def __init__(self, period: Optional[np.ndarray] = None) -> None:

        self.labels_ = None
        self.period = period
        self._distance = _euclidean_distance_period if period is not None else _euclidean_distance

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fit the data. Generate the cluster center by FPS algorithm."""

        ngrid = len(X)
        self.grid_pos = X
        self.grid_npoints = np.zeros(ngrid, dtype=int)
        self.grid_weight = np.zeros(ngrid, dtype=float)
        self.grid_neighbour = {i: [] for i in range(ngrid)}

    def predict(self,
                X: np.ndarray,
                y: Optional[np.ndarray] = None,
                sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """Transform the data."""
        if sample_weight is None:
            sample_weight = np.ones(len(X)) / len(X)
        self.labels_ = []
        for i, point in track(enumerate(X), description='Assigning samples to grids...', total=len(X)):
            descriptor2grid = self._distance(point, self.grid_pos, self.period)
            self.labels_.append(np.argmin(descriptor2grid))
            self.grid_npoints[self.labels_[-1]] += 1
            self.grid_weight[self.labels_[-1]] += sample_weight[i]
            self.grid_neighbour[self.labels_[-1]].append(i)

        for key in self.grid_neighbour:
            self.grid_neighbour[key] = np.array(self.grid_neighbour[key])

        return self.labels_

class MaxMinClustering:
    """MinMaxClustering Class"""

    def __init__(self, n_cluster: int, init_grid_idx: list, period: Optional[list] = None) -> None:

        self.n_cluster = n_cluster
        self.selected_idx_ = np.full(n_cluster, -1, dtype=int)
        self.selected_idx_[:len(init_grid_idx)] = init_grid_idx
        self._min_dist = None
        self.labels_ = None
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
        """Fit the data. Generate the cluster center by FPS algorithm."""

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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Transform the data."""
        if X.shape != self.support_.shape:
            raise ValueError("X has a different shape than during the fitting.")

        self.labels_ = []
        for point in X:
            descriptor2grid = self._distance(point, self.cluster_centers_, self.period)
            self.labels_.append(np.argmin(descriptor2grid))

        return self.labels_

    def _post_init(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.support_ = np.full(X.shape, False)
        self.support_[self.selected_idx_] = True
        self.cluster_centers_ = X[self.selected_idx_]
    