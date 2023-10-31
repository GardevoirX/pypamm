import os
import numpy as np
from skmatter.feature_selection import FPS
from scipy.spatial.distance import cdist
from typing import Optional

def read_period(period_text):
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

def covariance(dimension: int, x: np.ndarray, period: np.ndarray,
               grid_weight: np.ndarray, totw: float):

    xm = np.zeros(dimension)
    xxm = np.zeros((dimension, len(x)))
    xxmw = np.zeros((dimension, len(x)))
    for i in range(dimension):
        if period[i] > 0:
            sumsin = np.sum(grid_weight * np.sim(x[i, :]) *\
                            (2 * np.pi) / period[i]) / totw
            sumcos = np.sum(grid_weight * np.sim(x[i, :]) *\
                            (2 * np.pi) / period[i]) / totw
            xm[i] = np.arctan2(sumsin, sumcos)
        else:
            xm[i] = np.sum(x[i, :] * w) / wnorm
        xxm[i, :] = x[i, :] - xm[i]
        if period[i] > 0:
            xxm[i, :] -= round(xxm[i, :] / period[i]) * period[i]
        xxmw[i, :] = xxm[i, :] * grid_weight /totw
        

    return

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
        self.weight /= self.totw
        if self.gridfile:
            igrid = self.from_gridfile()
        else:
            igrid = self.cal_grid()
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

        if self.ngrid == -1:
        # If not specified, the number of voronoi polyhedra
        # are set to the square root of the total number of points
            self.ngrid = int(np.sqrt(nsamples))
        
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
    
    def computes_localization(self):
        delta = 1 / self.nsamples
        # only one of the methods can be used at a time
        if self.fspread > 0:
            self.fpoints = -1.

        return
