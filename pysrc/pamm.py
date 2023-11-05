import os
import warnings
import numpy as np
from numba import jit
from rich.progress import track
from skmatter.feature_selection import FPS
from scipy.spatial.distance import cdist
from sklearn.covariance import OAS
from typing import Optional

def read_period(period_text):

    if period_text is None:
        return None

    period = []
    for num in period_text.split(','):
        period.append(eval(num))
        if period[-1] == 6.28: period[-1] = 2 * np.pi
        elif period[-1] == 3.14: period[-1] = np.pi

    return np.array(period)

def pammr2(period, xi, xj):

    if len(xi.shape) == 1:
        xi = xi[np.newaxis, :]

    xij = np.zeros(xi.shape, dtype=float)
    xij = pammrij(period, xij, xi, xj)

    return np.sum(xij**2, axis=1)

@jit(nopython=True, fastmath=True)
def pammrij(period, xij, xi, xj):

    period_feature = period > 0
    xij = xi - xj
    xij[:, period_feature] -= np.round(xij[:, period_feature]/period[period_feature]) * period[period_feature]

    return xij

def covariance(grid_pos: np.ndarray, period: np.ndarray,
               grid_weight: np.ndarray, totw: float):
    """
    Calculate the covariance matrix for a given set of grid positions and weights.

    Parameters:
        grid_pos (np.ndarray): An array of shape (nsample, dimension) representing the grid positions.
        period (np.ndarray): An array of shape (dimension,) representing the periodicity of each dimension.
        grid_weight (np.ndarray): An array of shape (nsample,) representing the weights of the grid positions.
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

def effdim(cov):

    eigval = np.linalg.eigvals(cov)
    eigval /= sum(eigval)
    eigval *= np.log(eigval)
    eigval[np.isnan(eigval)] = 0.
    
    return np.exp(-sum(eigval))

def localization(period: np.ndarray, grid_pos: np.ndarray, target_grid_pos: np.ndarray, grid_weight: np.ndarray, s2: float):
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
        if period[i] <= 0: continue
        # scaled length
        xy[:, i] /= period[i]
        # Finds the smallest separation between the images of the vector elements
        xy[:, i] -= np.round(xy[:, i])
        # Rescale back the length
        xy[:, i] *= period[i]
    wl = np.exp(-0.5 / s2 * np.sum(xy**2, axis=1)) * grid_weight
    num = np.sum(wl)

    return wl, num

def oas(cov, n, D):

    tr = np.trace(cov)
    tr2 = tr ** 2
    tr_cov2 = np.trace(cov ** 2)
    phi = ((1 - 2 / D) * tr_cov2 + tr2) / ((n + 1 - 2 / D) * tr_cov2 - tr2 / D)
    
    return (1 - phi) * cov + np.diag([phi * tr /D for i in range(D)])

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

    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    if len(cov_inv.shape) == 2:
        cov_inv = cov_inv[np.newaxis, :, :]
    xy = np.zeros(x.shape)
    xy = pammrij(period, xy, x, y)
    if cov_inv.shape[0] == 1:
        tmpv = xy.dot(cov_inv[0])
    else:
        tmpv = np.array([xy[i].dot(cov_inv[i]) for i in range(x.shape[0])])
    xcx = np.array([xy[i].dot(tmpv[i].T) for i in range(x.shape[0])])

    return xcx

class PAMM:
    def __init__(self, dimension, alpha:float = 1., 
                 fpost:bool = False, seed:int = 12345, qs:float = 1., 
                 nmsopt:int = 0, ngrid:int = -1, bootstrap:int = 0, 
                 fspread:float = -1., fpoints:float = 0.15, 
                 period_text:Optional[str]= None, zeta:float = 0.,
                 thrmerg:float = 0.8, thrpcl:float = 0., outputfile:str = 'out',
                 savegrid:bool = False, gridfile:Optional[str] = None, 
                 savevor:bool = False, saveneigh:bool = False, 
                 neighfile:Optional[str] = None, gs:int = -1, 
                 weighted:bool = False, verbose:bool = False) -> None:
        
        self.dimension = dimension
        self.alpha = alpha
        self.fpost = fpost # cluster file
        self.seed = seed
        self.qs = qs
        self.nmsopt = nmsopt
        self.ngrid = ngrid
        self.bootstrap = bootstrap
        self.fspread = fspread
        self.fpoints = fpoints
        self.period = read_period(period_text)
        self.zeta = zeta
        self.thrmerg = thrmerg
        self.thrpcl = thrpcl
        self.outputfile = outputfile
        self.savegrid = savegrid
        self.gridfile = gridfile
        self.savevor = savevor
        self.saveneigh = saveneigh
        self.neighfile = neighfile
        self.gs = gs
        self.weighted = weighted
        self.verbose = verbose

        if self.qs < 0:
            raise ValueError('The QS scaling should be positive')
        if self.bootstrap < 0:
            raise ValueError('The number of iterations should be positive')
        if len(self.period) != self.dimension:
            raise ValueError('Check the number of periodic dimensions!')
        
    def run(self, descriptorfile):

        self.nsamples, self.descriptor, self.weight, self.totw = \
            self._read_descriptor_and_weights(descriptorfile)
        self.totw = 1
        if self.ngrid == -1:
        # If not specified, the number of voronoi polyhedra
        # are set to the square root of the total number of points
            self.ngrid = int(np.sqrt(self.nsamples))
        self.local_dimension = np.zeros(self.ngrid)
        if self.gridfile:
            igrid = self.from_gridfile()
        else:
            igrid = self.cal_grid()
        self.grid_pos = self.descriptor[igrid]
        self.grid_npoints, self.grid_weight, grid_neighbour = self.clustering()
        dist_matrix = self.get_grid_dist_matrix()
        gabriel = self.get_gabriel_graph(dist_matrix)
        h_invs, normkernels, qscut2 = self.computes_localization(igrid, dist_matrix)
        prob = self.computes_kernel_density_estimation(h_invs, normkernels, igrid, grid_neighbour)
        return prob
        
    def post_processing(self, clusterfile:str):

        raise NotImplementedError
    
    def _read_descriptor_and_weights(self, descriptorfile:str):
        '''Read the descriptor file provided.

        Args:
            descriptorfile: a string, the name of your file containing
            your descriptors and weights (if needed)
        
        Returns:
            nsamples: int, the number of samples
            descriptor: np.ndarray, the descriptor
            weight: np.ndarray, the weight
            totw: float, the sum of weight array
        '''

        content = np.loadtxt(descriptorfile)
        assert self.dimension + self.weighted == content.shape[1], \
               'Please check the number of columns of your descriptor' \
               'file. It does not equal the number of descriptor ' \
               'plusing weight (if needed in your input arguments).'
        nsamples = content.shape[0]
        descriptor = content[:, :self.dimension]
        if not self.weighted:
            weight = np.full(nsamples, 1, dtype=float)
        else:
            weight = content[:, -1]
        totw = np.sum(weight)
        weight /= totw
        
        return nsamples, descriptor, weight, totw
    
    def from_gridfile(self):

        igrid = np.loadtxt(self.gridfile, dtype=int)[:self.ngrid]
        igrid -= 1 #  fortran's array index starts from 1

        return igrid
    
    def cal_grid(self):

        if self.verbose:
            print(f'NSamples: {self.nsamples}')
            print(f'Selecting: {self.ngrid} points using MINMAX')
        fps = FPS(n_to_select=self.ngrid)
        fps.fit(self.descriptor.T)
        igrid = fps.selected_idx_

        return igrid
    def clustering(self):

        grid_npoints = np.zeros(self.ngrid, dtype=int)
        grid_weight = np.zeros(self.ngrid)
        grid_neighbour = {i: [] for i in range(self.ngrid)}

        # assign samples to its cloeset grid
        dist_grid2descriptor = []
        for descriptor in self.descriptor:
            dist_grid2descriptor.append(pammr2(self.period, self.grid_pos, descriptor))
        dist_grid2descriptor = np.array(dist_grid2descriptor).T
        labels = np.argmin(dist_grid2descriptor, axis=0)
        for ipoint, label in enumerate(labels):
            grid_npoints[label] += 1
            grid_weight[label] += self.weight[ipoint]
            grid_neighbour[label].append(ipoint)
        
        assert np.sum(grid_weight == 0) == 0, \
            "Error: voronoi has no points associated with" \
            "- probably two points are perfectly overlapping"

        for key in grid_neighbour:
            grid_neighbour[key] = np.array(grid_neighbour[key])

        return grid_npoints, grid_weight, grid_neighbour
    
    def get_grid_dist_matrix(self):
        # TODO: take PBC into consideration
        if self.verbose:
            print(" Precalculate distance matrix between grid points")

        dist_matrix = np.zeros((self.ngrid, self.ngrid))
        for i in range(self.ngrid):
            dist_matrix[i, i] = np.inf
            dist_matrix[i, i + 1:] = \
                pammr2(self.period, self.grid_pos[i + 1:], self.grid_pos[i])
            dist_matrix[i + 1:, i] = dist_matrix[i, i + 1:]

        return dist_matrix
        #return cdist(self.descriptor[self.igrid], self.descriptor[self.igrid])
    
    def get_gabriel_graph(self, dist_matrix2: np.ndarray):

        gabriel = np.full((self.ngrid, self.ngrid), True)
        for i in range(self.ngrid):
            gabriel[i, i] = False
            for j in range(i, self.ngrid):
                if np.sum(dist_matrix2[i] + dist_matrix2[j] < dist_matrix2[i, j]):
                    gabriel[i, j] = False
                    gabriel[j, i] = False

        if self.saveneigh:
            neigh_file = os.path.basename(self.outputfile) + '.neigh'
            with open(neigh_file, 'w') as wfl:
                for i in range(self.ngrid):
                    print(' '.join(gabriel[i, :]), file=wfl)
        return gabriel
    
    def computes_localization(self, igrid: np.ndarray, mindist: np.ndarray):
        delta = 1 / self.nsamples
        # only one of the methods can be used at a time
        if self.fspread > 0:
            self.fpoints = -1.
        cov = covariance(self.grid_pos, self.period, self.grid_weight, 1.0)
        print(f'Global eff. dim. {effdim(cov)}')

        if self.period is not None:
            tune = sum(self.period ** 2)
        else:
            tune = np.trace(cov)
        sigma2 = np.full(self.ngrid, tune)

        # initialize the localization based on fraction of data spread
        if self.fspread > 0:
            sigma2 *= self.fspread ** 2
        if self.verbose:
            print('Estimating kernel density bandwidths')
        ndescriptor = self.descriptor.shape[1]
        flocal = np.zeros(self.ngrid)
        h_invs = np.zeros((self.ngrid, ndescriptor, ndescriptor))
        normkernels = np.zeros(self.ngrid)
        qscut2 = np.zeros(self.ngrid)
        for i in range(self.ngrid):
            if self.verbose and (i % 100 == 0):
                print(f'  {i} / {self.ngrid}')
            wlocal, flocal[i] = localization(self.period, self.grid_pos, self.grid_pos[i], self.grid_weight, sigma2[i])
            if self.fpoints > 0:
                sigma2, flocal, wlocal = self._localization_based_on_fraction_of_points(sigma2, flocal, i, delta, tune, igrid)
            else:
                sigma2, flocal, wlocal = self._localization_based_on_fraction_of_spread(sigma2, flocal, i, mindist, igrid)
            h_invs[i], normkernels[i], qscut2[i] = \
                self._bandwidth_estimation_from_localization(wlocal, flocal, i, )

        qscut2 *= self.qs ** 2

        return h_invs, normkernels, qscut2
    
    def _localization_based_on_fraction_of_points(self, sigma2, flocal, idx, delta, tune, igrid):

        lim = self.fpoints
        if lim <= self.grid_weight[idx]:
            lim = self.grid_weight[idx] + delta
            warnings.warn(" Warning: localization smaller than voronoi, increase grid size (meanwhile adjusted localization)!")
        while flocal[idx] < lim:
            sigma2[idx] += tune
            wlocal, flocal[idx] = localization(self.period, self.grid_pos, self.grid_pos[idx], self.grid_weight, sigma2[idx])
        j = 1
        while True:
            if flocal[idx] > lim:
                sigma2[idx] -= tune / 2 ** j
            else:
                sigma2[idx] += tune / 2 ** j
            wlocal, flocal[idx] = localization(self.period, self.grid_pos, self.grid_pos[idx], self.grid_weight, sigma2[idx])
            if abs(flocal[idx] - lim) < delta:
                break
            j += 1

        return sigma2, flocal, wlocal
    
    def _localization_based_on_fraction_of_spread(self, sigma2, flocal, idx, mindist, igrid):
        
        if sigma2[idx] < mindist[idx]:
            sigma2[idx] = mindist[idx]
            wlocal, flocal[idx] = localization(self.period, self.descriptor, self.descriptor[igrid[idx]], self.grid_weight, sigma2[idx])

        return sigma2, flocal, wlocal
    
    def _bandwidth_estimation_from_localization(self, wlocal, flocal, idx):

        cov_i = covariance(self.grid_pos, self.period, wlocal, flocal[idx])
        nlocal = flocal[idx] * self.nsamples
        self.local_dimension[idx] = effdim(cov_i)
        cov_i = oas(cov_i, nlocal, self.descriptor.shape[1])
        # cov_i_inv = np.linalg.inv(cov_i)
        h = (4. / (self.local_dimension[idx] + 2.)) ** (2. / (self.local_dimension[idx] + 4.)) * nlocal ** (-2. / (self.local_dimension[idx] + 4.)) * cov_i
        h_inv = np.linalg.inv(h)
        sign, det = np.linalg.slogdet(h)
        logdet_h = det
        normkernel = self.dimension * np.log(2 * np.pi) + logdet_h
        qscut2 = np.trace(cov_i)

        return  h_inv, normkernel, qscut2
    
    def computes_kernel_density_estimation(self, h_inv: np.ndarray, normkernel: np.ndarray, igrid: np.ndarray, neighbour: dict):

        if self.verbose:
            print('Computing kernel density on reference points')
        d = self.descriptor.shape[1]
        kdecut2 = 9 * (np.sqrt(d) + 1) ** 2
        prob = np.full(self.ngrid, -np.inf)
        for i in range(self.ngrid):
            if self.verbose and (i % 100 == 0):
                print(f'  {i} / {self.ngrid}')
                dummd1s = mahalanobis(self.period, self.grid_pos, self.grid_pos[i], h_inv)
            for j, dummd1 in enumerate(dummd1s):
                if dummd1 > kdecut2:
                    lnk = -0.5 * (normkernel[j] + dummd1) + np.log(self.grid_weight[j])
                    prob[i] = _update_prob(prob[i], lnk)
                else:
                    neighbours = neighbour[j][neighbour[j] != igrid[i]]
                    dummd1s = mahalanobis(self.period, self.descriptor[neighbours], self.grid_pos[i], h_inv[j])
                    lnks = -0.5 * (normkernel[j] + dummd1s) + np.log(self.weight[neighbours])
                    prob[i] = _update_probs(prob[i], lnks)

        prob -= np.log(self.totw)

        return prob
    
@jit(nopython=True)
def _update_probs(prob_i: float, lnks: np.ndarray):

    for lnk in lnks:
        prob_i = _update_prob(prob_i, lnk)

    return prob_i

@jit(nopython=True)
def _update_prob(prob_i: float, lnk: float):

    if prob_i > lnk:
        return prob_i + np.log(1 + np.exp(lnk - prob_i))
    else:
            return lnk + np.log(1 + np.exp(prob_i - lnk))