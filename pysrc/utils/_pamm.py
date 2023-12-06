import numpy as np
from numba import jit
from scipy.special import logsumexp as LSE

@jit(nopython=True)
def gs_next(idx: int, probs: np.ndarray, n_shells: int, distmm: np.ndarray, gabriel: np.ndarray):
    """Find next cluster in Gabriel graph."""

    ngrid = len(probs)
    neighs = np.copy(gabriel[idx])
    for _ in range(1, n_shells):
        nneighs = np.full(ngrid, False)
        for j in range(ngrid):
            if neighs[j]:
                nneighs |= gabriel[j]
        neighs |= nneighs

    next_idx = idx
    dmin = np.inf
    for j in range(ngrid):
        if probs[j] > probs[idx] and \
            distmm[idx, j] < dmin and \
            neighs[j]:
            next_idx = j
            dmin = distmm[idx, j]

    return next_idx

@jit(nopython=True)
def qs_next(idx:int, idxn: int, probs: np.ndarray, distmm: np.ndarray, lambda_: float):
    """Find next cluster with respect to qscut(lambda_)."""

    ngrid = len(probs)
    dmin = np.inf
    next_idx = idx
    if probs[idxn] > probs[idx]:
        next_idx = idxn
    for j in range(ngrid):
        if probs[j] > probs[idx] and \
            distmm[idx, j] < dmin and \
            distmm[idx, j] < lambda_:
            next_idx = j
            dmin = distmm[idx, j]

    return next_idx

def logsumexp(v1: np.ndarray, probs: np.ndarray, clusterid: int):

    mask = v1 == clusterid
    probs = np.copy(probs)
    probs[~mask] = -np.inf

    return LSE(probs)

def getidmax(v1: np.ndarray, probs: np.ndarray, clusterid: int):

    tmpv = np.copy(probs)
    tmpv[v1 != clusterid] = -np.inf
    return np.argmax(tmpv)

@jit(nopython=True)
def oas(cov, n, D):

    tr = np.trace(cov)
    tr2 = tr ** 2
    tr_cov2 = np.trace(cov ** 2)
    phi = ((1 - 2 / D) * tr_cov2 + tr2) / ((n + 1 - 2 / D) * tr_cov2 - tr2 / D)

    return (1 - phi) * cov + np.eye(D) * phi * tr /D # np.diag([phi * tr /D for i in range(D)])

@jit(nopython=True)
def covariance(grid_pos: np.ndarray, period: np.ndarray,
               grid_weight: np.ndarray, totw: float):
    """
    Calculate the covariance matrix for a given set of grid positions and weights.

    Parameters:
        grid_pos (np.ndarray): An array of shape (nsample, dimension)
        representing the grid positions.
        period (np.ndarray): An array of shape (dimension,)
        representing the periodicity of each dimension.
        grid_weight (np.ndarray): An array of shape (nsample,)
        representing the weights of the grid positions.
        totw (float): The total weight.

    Returns:
        cov (np.ndarray): The covariance matrix of shape (dimension, dimension).

    Note:
        The function assumes that the grid positions, weights, and total weight are provided correctly.
        The function handles periodic and non-periodic dimensions differently to calculate the covariance matrix.
    """

    nsample = grid_pos.shape[0]
    dimension = grid_pos.shape[1]
    xm = np.zeros(dimension)
    xxm = np.zeros((nsample, dimension))
    xxmw = np.zeros((nsample, dimension))
    for i in range(dimension):
        if period[i] > 0:
            sumsin = np.sum(grid_weight * np.sin(grid_pos[:, i]) *\
                            (2 * np.pi) / period[i]) / totw
            sumcos = np.sum(grid_weight * np.cos(grid_pos[:, i]) *\
                            (2 * np.pi) / period[i]) / totw
            xm[i] = np.arctan2(sumsin, sumcos)
        else:
            xm[i] = np.sum(grid_pos[:, i] * grid_weight) / totw
        xxm[:, i] = grid_pos[:, i] - xm[i]
        if period[i] > 0:
            xxm[:, i] -= np.round(xxm[:, i] / period[i]) * period[i]
    xxmw = xxm * grid_weight.reshape(-1, 1) / totw
    cov = xxmw.T.dot(xxm)
    cov /= 1 - sum((grid_weight / totw) ** 2)

    return cov

def read_period(period_text: str, dimension: int):

    if period_text is None:
        return np.full(dimension, -1)

    period = []
    for num in period_text.split(','):
        period.append(eval(num))
        if period[-1] == 6.28:
            period[-1] = 2 * np.pi
        elif period[-1] == 3.14:
            period[-1] = np.pi

    return np.array(period)

def effdim(cov):
    """
    Calculate the effective dimension of a covariance matrix.

    Parameters:
        cov (ndarray): The covariance matrix.

    Returns:
        float: The effective dimension of the covariance matrix.
    """

    eigval = np.linalg.eigvals(cov)
    eigval /= sum(eigval)
    eigval *= np.log(eigval)
    eigval[np.isnan(eigval)] = 0.

    return np.exp(-sum(eigval))

@jit(nopython=True)
def localization(period: np.ndarray,
                 grid_pos: np.ndarray,
                 target_grid_pos: np.ndarray,
                 grid_weight: np.ndarray,
                 s2: float):
    """
    Calculates the localization of a set of vectors in a grid.

    Args:
        period (np.ndarray): An array of periods for each dimension of the grid.
        x (np.ndarray): An array of vectors to be localized.
        y (np.ndarray): An array of target vectors representing the grid.
        grid_weight (np.ndarray): An array of weights for each target vector.
        s2 (float): The scaling factor for the squared distance.

    Returns:
        tuple: A tuple containing two numpy arrays:
            wl (np.ndarray): An array of localized weights for each vector.
            num (np.ndarray): The sum of the localized weights.

    """

    dimension = len(period)
    xy = grid_pos - target_grid_pos
    for i in range(dimension):
        if period[i] <= 0:
            continue
        # scaled length
        xy[:, i] /= period[i]
        # Finds the smallest separation between the images of the vector elements
        xy[:, i] -= np.round(xy[:, i])
        # Rescale back the length
        xy[:, i] *= period[i]
    wl = np.exp(-0.5 / s2 * np.sum(xy**2, axis=1)) * grid_weight
    num = np.sum(wl)

    return wl, num
