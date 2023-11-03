import os
import warnings
import numpy as np
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

    return period

def pammr2(period, xi, xj):
    xij = xi - xj
    for d, p in enumerate(period):
        xij[d] = xij[d] - round(xij[d]/p) * p
    return np.sum(xij**2)

def covariance(grid_pos: np.ndarray, period: np.ndarray,
               grid_weight: np.ndarray, totw: float):

    nsample = grid_pos.shape[0]
    dimension = grid_pos.shape[1]
    xm = np.zeros(dimension)
    xxm = np.zeros((nsample, dimension))
    xxmw = np.zeros((nsample, dimension))
    for i in range(dimension):
        if period[i] > 0:
            sumsin = np.sum(grid_weight * np.sim(grid_pos[:, i]) *\
                            (2 * np.pi) / period[i]) / totw
            sumcos = np.sum(grid_weight * np.sim(grid_pos[:, i]) *\
                            (2 * np.pi) / period[i]) / totw
            xm[i] = np.arctan2(sumsin, sumcos)
        else:
            xm[i] = np.sum(grid_pos[:, i] * grid_weight) / totw
        xxm[:, i] = grid_pos[:, i] - xm[i]
        if period[i] > 0:
            xxm[:, i] -= round(xxm[:, i] / period[i]) * period[i]
        xxmw[:, i] = xxm[:, i] * grid_weight /totw
    
        cov = xxm.dot(xxmw.T)
        cov /= 1 - sum(grid_weight / totw) ** 2

    return cov

def effdim(cov):

    eigval = np.linalg.eigvals(cov)
    eigval /= sum(eigval)
    eigval *= np.log(eigval)
    eigval[np.isnan(eigval)] = 0.
    
    return np.exp(-sum(eigval))

def localization(period, x: np.ndarray, y: np.ndarray, grid_weight: np.ndarray, s2: float):

    dimension = len(period)
    xy = np.zeros(x.shape)
    for i in range(dimension):
        if period[i] <= 0: continue
        # scaled length
        xy[i, :] = x[i, :] - y[i]
        # Finds the smallest separation between the images of the vector elements
        xy[i, :] -= round(xy[i, :])
        # Rescale back the length
        xy[i, :] *= period[i]
    wl = np.exp(-0.5 / s2 * np.sum(xy**2, axis=0)) * grid_weight
    num = np.sum(wl)

    return wl, num


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
        
        self.deminsion = dimension
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
        if len(self.period) != self.deminsion:
            raise ValueError('Check the number of periodic dimensions!')
        
    def run(self, descriptorfile):

        self.nsamples, self.descriptor, self.weight, self.totw = \
            self._read_descriptor_and_weights(descriptorfile)
        if self.ngrid == -1:
        # If not specified, the number of voronoi polyhedra
        # are set to the square root of the total number of points
            self.ngrid = int(np.sqrt(self.nsamples))
        self.local_dimension = np.zeros(self.ngrid)
        self.weight /= self.totw
        if self.gridfile:
            igrid = self.from_gridfile()
        else:
            igrid = self.cal_grid()
        self.grid_pos = self.descriptor[igrid]
        self.grid_npoints, self.grid_weight = self.clustering(igrid)
        dist_matrix = self.get_grid_dist_matrix(igrid)
        gabriel = self.get_gabriel_graph(dist_matrix)

        return dist_matrix, gabriel
        
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
        assert self.deminsion + self.weighted == content.shape[1], \
               'Please check the number of columns of your descriptor' \
               'file. It does not equal the number of descriptor ' \
               'plusing weight (if needed in your input arguments).'
        nsamples = content.shape[0]
        descriptor = content[:, :self.deminsion]
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
        #grid = fps.transform(self.descriptor.T)
        igrid = fps.selected_idx_

        return igrid
    def clustering(self, igrid: np.ndarray):

        grid_npoints = np.zeros(self.ngrid, dtype=int)
        grid_weight = np.zeros(self.ngrid)

        # assign samples to its cloeset grid
        dist_grid2descriptor = cdist(self.descriptor[igrid], self.descriptor)
        labels = np.argmin(dist_grid2descriptor, axis=0)
        for ipoint, label in enumerate(labels):
            grid_npoints[label] += 1
            grid_weight[label] += self.weight[ipoint]
        
        assert np.sum(grid_weight == 0) == 0, \
            "Error: voronoi has no points associated with" \
            "- probably two points are perfectly overlapping"

        return grid_npoints, grid_weight
    
    def get_grid_dist_matrix(self, igrid: np.ndarray):
        # TODO: take PBC into consideration
        if self.verbose:
            print(" Precalculate distance matrix between grid points")

        grid = self.descriptor[igrid]
        dist_matrix = np.zeros((self.ngrid, self.ngrid))
        for i in range(self.ngrid):
            dist_matrix[i, i] = np.inf
            for j in range(i + 1, self.ngrid):
                dist_matrix[i, j] = pammr2(self.period, grid[i], grid[j])
                dist_matrix[j, i] = dist_matrix[i, j]
                if (i==0) and (j==1):
                    print(grid[i])
                    print(grid[j])
                    print(dist_matrix[i, j])

        return dist_matrix
        #return cdist(self.descriptor[self.igrid], self.descriptor[self.igrid])
    
    def get_gabriel_graph(self, dist_matrix2: np.ndarray):

        gabriel = np.full((self.ngrid, self.ngrid), True)
        for i in range(self.ngrid):
            gabriel[i, i] = False
            for j in range(i, self.ngrid):
                if not gabriel[i, j]:
                    continue
                for k in range(self.ngrid):
                    if dist_matrix2[i, j] >= dist_matrix2[i, k] + dist_matrix2[j, k]:
                        gabriel[i, j] = False
                        gabriel[j, i] = False
                        break

        if self.saveneigh:
            neigh_file = os.path.basename(self.outputfile) + '.neigh'
            with open(neigh_file, 'w') as wfl:
                for i in range(self.ngrid):
                    print(' '.join(gabriel[i, :]))
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
        sigma2 = tune

        # initialize the localization based on fraction of data spread
        if self.fspread > 0:
            sigma2 *= self.fspread ** 2
        if self.verbose:
            print('Estimating kernel density bandwidths')
        flocal = np.zeros(self.ngrid)
        for i in range(self.ngrid):
            if self.verbose and (i % 100 == 0):
                print(f'  {i} / {self.ngrid}')
            wlocal, flocal[i] = localization(self.period, self.descriptor, self.descriptor[igrid[i]], self.grid_weight, sigma2[i])
            if self.fpoints > 0:
                sigma2, flocal = self._localization_based_on_fraction_of_points(sigma2, flocal, i, delta, tune, igrid)
            else:
                sigma2, flocal = self._localization_based_on_fraction_of_spread(sigma2, flocal, i, mindist, igrid)
            self._bandwidth_esitimation_from_localization()

        qscut2 *= self.qs ** 2
        return
    
    def _localization_based_on_fraction_of_points(self, sigma2, flocal, idx, delta, tune, igrid):

        lim = self.fpoints
        if lim <= self.grid_weight[idx]:
            lim = self.grid_weight[idx] + delta
            warnings.warn(" Warning: localization smaller than voronoi, increase grid size (meanwhile adjusted localization)!")
        while flocal[idx] < lim:
            sigma2[idx] += tune
            wlocal, flocal[idx] = localization(self.period, self.descriptor, self.descriptor[igrid[idx]], self.grid_weight, sigma2[idx])
        j = 1
        while True:
            if flocal[idx] > lim:
                sigma2[idx] -= tune / 2 ** j
            else:
                sigma2[idx] += tune / 2 ** j
            wlocal, flocal[idx] = localization(self.period, self.descriptor, self.descriptor[igrid[idx]], self.grid_weight, sigma2[idx])
            if abs(flocal[idx] - lim) < delta:
                break
            j += 1

        return sigma2, flocal
    
    def _location_based_on_fraction_of_spread(self, sigma2, flocal, idx, mindist, igrid):
        
        if sigma2[idx] < mindist[idx]:
            sigma2[idx] = mindist[idx]
            wlocal, flocal[idx] = localization(self.period, self.descriptor, self.descriptor[igrid[idx]], self.grid_weight, sigma2[idx])

        return sigma2, flocal
    
    def _bandwidth_estimation_from_localization(self, wlocal, flocal, idx, ):

        cov_i = covariance(self.grid_pos, self.period, wlocal, flocal[idx])
        nlocal = flocal[idx] * self.nsamples
        self.local_dimension[idx] = effdim(cov_i)
        cov_i = OAS().fit(cov_i).covariance_
        cov_i_inv = np.linalg.inv(cov_i)
        h = (4. / (self.local_dimension[idx] + 2.)) ** (2. / (self.local_dimension[idx] + 4.)) * nlocal ** (-2. / (self.local_dimension[idx] + 4.)) * cov_i
        h_inv[idx] = np.linalg.inv(h)
        logdet_h[idx] = np.linalg.slogdet(h)[1]
        normkernel[idx] = self.deminsion * 2 * np.pi + logdet_h[idx]
        qscut2[idx] = np.trace(cov_i)

        return 

