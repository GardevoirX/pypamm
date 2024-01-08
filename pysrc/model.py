from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from pysrc.utils.dist import pammrij

@dataclass
class GaussianMixtureModel:
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    period: Optional[np.ndarray] = None

    def __post_init__(self):
        self.dimension = self.means.shape[1]
        self.cov_inv = np.linalg.inv(self.covariances)
        self.cov_det = np.linalg.det(self.covariances)
        self.norm = 1 / np.sqrt((2 * np.pi) ** self.dimension * self.cov_det)

    def __call__(self, x: np.ndarray, i: Optional[Union[int, list[int]]]=None):
        """
        Calculate the probability density function (PDF) value for a given input array.

        Parameters:
            x (np.ndarray): The input array for which the PDF is calculated. Once a point.
            i (Optional[int]): The index of the element in the PDF array to return. 
                If None, the sum of all elements is returned.

        Returns:
            float or np.ndarray: The PDF value(s) for the given input(s). 
                If i is None, the sum of all PDF values is returned. 
                If i is specified, the normalized value of the corresponding gaussian is returned.

        Raises:
            None

        Example:
            >>> obj = ClassName()
            >>> obj.__call__(x, i)
            0.123456789
        """

        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        if self.period is not None:
            xij = np.zeros(self.means.shape)
            xij = pammrij(self.period, xij, x, self.means)
        else:
            xij = x - self.means
        p = self.weights * self.norm * \
            np.exp(-0.5 * (xij[:, np.newaxis, :] @
                           self.cov_inv @ xij[:, :, np.newaxis])).reshape(-1)
        sum_p = np.sum(p)
        if i is None:
            return sum_p

        return np.sum(p[i]) / sum_p
