import numpy as np
from numba import jit

def pammr2(period, xi, xj):

    if len(xi.shape) == 1:
        xi = xi[np.newaxis, :]

    xij = np.zeros(xi.shape, dtype=float)
    xij = pammrij(period, xij, xi, xj)

    return np.sum(xij**2, axis=1)

@jit(nopython=True)
def pammrij(period, xij, xi, xj):

    period_feature = period > 0
    xij = xi - xj
    xij[:, period_feature] -= np.round(xij[:, period_feature]/period[period_feature]) * period[period_feature]

    return xij

def mahalanobis(period: np.ndarray, x: np.ndarray, y: np.ndarray, cov_inv: np.ndarray):
    """
    Calculates the Mahalanobis distance between two vectors.

    Args:
        period (np.ndarray): An array of periods for each dimension of the grid.
        x (np.ndarray): An array of vectors to be localized.
        y (np.ndarray): An array of target vectors representing the grid.
        cov_inv (np.ndarray): The inverse of the covariance matrix.

    Returns:
        float: The Mahalanobis distance.

    """
    
    x, cov_inv = _mahalanobis_preprocess(x, cov_inv)
    return _mahalanobis(period, x, y, cov_inv)

def _mahalanobis_preprocess(x: np.ndarray, cov_inv: np.ndarray):

    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    if len(cov_inv.shape) == 2:
        cov_inv = cov_inv[np.newaxis, :, :]

    return x, cov_inv

@jit(nopython=True, fastmath=True)
def _mahalanobis(period: np.ndarray, x: np.ndarray, y: np.ndarray, cov_inv: np.ndarray):

    xy = np.zeros(x.shape, dtype=float)
    tmpv = np.zeros(x.shape, dtype=float)
    xcx = np.zeros(x.shape[0], dtype=float)
    xy = pammrij(period, xy, x, y)
    if cov_inv.shape[0] == 1:
        # many samples and one cov
        tmpv = xy.dot(cov_inv[0])
    else:
        # many samples and many cov
        for i in range(x.shape[0]):
            tmpv[i] = np.dot(xy[i], cov_inv[i])
    for i in range(x.shape[0]):
        xcx[i] = np.dot(xy[i], tmpv[i].T)

    return xcx