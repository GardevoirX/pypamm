import numpy as np
from numba import jit

def pammr2(period: np.ndarray, xi: np.ndarray, xj: np.ndarray):
    """
    Calculates the period-concerned squared distance.
    
    Args:
        period (np.ndarray): An array of periods for each dimension of the points.
        -1 stands for not periodic.
        xi (np.ndarray): An array of point coordinates. It can also contain many points.
        Shape: (n_points, n_dimensions)
        xj (np.ndarray): An array of point coordinates. It can only contain one point.

    Returns:
        np.ndarray: An array of squared distances. Shape: (n_points)
    """

    if len(xi.shape) == 1:
        xi = xi[np.newaxis, :]

    xij = np.zeros(xi.shape, dtype=float)
    xij = pammrij(period, xij, xi, xj)

    return np.sum(xij**2, axis=1)

@jit(nopython=True)
def pammrij(period: np.ndarray, xij: np.ndarray, xi: np.ndarray, xj: np.ndarray):
    """
    Calculates the period-concerned position vector.
    Args:
        period (np.ndarray): An array of periods for each dimension of the points.
        -1 stands for not periodic.
        xij (np.ndarray): An array for storing the result.
        xi (np.ndarray): An array of point coordinates. It can also contain many points.
        Shape: (n_points, n_dimensions)
        xj (np.ndarray): An array of point coordinates. It can only contain one point.

    Returns:
        xij (np.ndarray): An array of position vectors. Shape: (n_points, n_dimensions)
    """

    period_feature = period > 0
    xij = xi - xj
    xij[:, period_feature] -= np.round(xij[:, period_feature]/period[period_feature]) * period[period_feature]

    return xij

def get_squared_dist_matrix(positions: np.ndarray, period: np.ndarray):
    """
    Generates the squared distance matrix between given positions using the PAMMR2 algorithm.
    
    Parameters:
        positions (np.ndarray): An array of point positions.
        period (np.ndarray): An array of period values.
        
    Returns:
        np.ndarray: The squared distance matrix between the positions.
    """

    ngrid = len(positions)
    dist_matrix = np.zeros((ngrid, ngrid))
    for i in range(ngrid):
        dist_matrix[i, i:] = \
            pammr2(period, positions[i:], positions[i])
        dist_matrix[i:, i] = dist_matrix[i, i:]
    np.fill_diagonal(dist_matrix, np.inf)

    return dist_matrix

def mahalanobis(period: np.ndarray, x: np.ndarray, y: np.ndarray, cov_inv: np.ndarray):
    """
    Calculates the Mahalanobis distance between two vectors.

    Args:
        period (np.ndarray): An array of periods for each dimension of vectors.
        x (np.ndarray): An array of vectors to be localized.
        y (np.ndarray): An array of target vectors.
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