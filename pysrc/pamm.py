import os
import warnings
from typing import Optional
import numpy as np
from numba import jit
from scipy.special import logsumexp as LSE
from skmatter.feature_selection import FPS
from pysrc.utils.dist import pammr2, mahalanobis

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

@jit(nopython=True)
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

@jit(nopython=True)
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

@jit(nopython=True)
def oas(cov, n, D):

    tr = np.trace(cov)
    tr2 = tr ** 2
    tr_cov2 = np.trace(cov ** 2)
    phi = ((1 - 2 / D) * tr_cov2 + tr2) / ((n + 1 - 2 / D) * tr_cov2 - tr2 / D)

    return (1 - phi) * cov + np.eye(D) * phi * tr /D # np.diag([phi * tr /D for i in range(D)])

def logsumexp(v1: np.ndarray, probs: np.ndarray, clusterid: int):

    mask = v1 == clusterid
    if np.any(mask):
        return LSE(probs[mask])
    else:
        return -np.inf

def getidmax(v1: np.ndarray, probs: np.ndarray, clusterid: int):

    tmpv = np.copy(probs)
    tmpv[v1 != clusterid] = -np.inf
    return np.argmax(tmpv)


class PAMM:
    def __init__(self, dimension, alpha:float = 1.,
                 fpost:bool = False, seed:int = 12345, qs:float = 1.,
                 nmsopt:int = 0, ngrid:int = -1,
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
        self.fspread = fspread
        self.fpoints = fpoints
        self.period = read_period(period_text, self.dimension)
        self.periodic = False if period_text is None else True
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
        #if self.bootstrap < 0:
        #    raise ValueError('The number of iterations should be positive')
        if len(self.period) != self.dimension:
            raise ValueError('Check the number of periodic dimensions!')

    def run(self, descriptors: np.ndarray, weights: Optional[np.ndarray] = None):

        self.nsamples, self.descriptor, self.weight, self.totw = \
            self._read_descriptor_and_weights(descriptors, weights)
        self.totw = 1
        if self.ngrid == -1:
        # If not specified, the number of voronoi polyhedra
        # are set to the square root of the total number of points
            self.ngrid = int(np.sqrt(self.nsamples))
        self.pabserr = np.full(self.ngrid, 0.0)
        self.prelerr = np.full(self.ngrid, np.inf)
        self.local_dimension = np.zeros(self.ngrid)
        if self.gridfile:
            igrid = self.from_gridfile()
        else:
            igrid = self.cal_grid()
        self.grid_pos = self.descriptor[igrid]
        self.grid_npoints, self.grid_weight, self.grid_neighbour = self.clustering()
        dist_matrix = self.get_grid_dist_matrix()
        self.dist_matrix = dist_matrix
        self.mindist = np.min(dist_matrix, axis=1)
        idmindist = np.argmin(dist_matrix, axis=1)
        self.gabriel = self.get_gabriel_graph(dist_matrix)
        h_invs, normkernels, qscut2, self.sigma2, self.flocal, self.h_tr_normed = \
            self.computes_localization(self.mindist)
        self.h_inv = h_invs
        self.normkernel = normkernels
        self.idmindist = idmindist
        self.qscut2 = qscut2
        self.probs = \
            self.computes_kernel_density_estimation(h_invs, normkernels, 
                                                    igrid, self.grid_neighbour)
        cluster_centers, idxroot = \
            self.quick_shift(self.probs, dist_matrix, idmindist, qscut2)
        cluster_centers, idxroot = \
            self.post_process(cluster_centers, idxroot, self.probs)
        self.cluster_centers = cluster_centers
        self.idxroot = idxroot

        return cluster_centers

    def post_processing(self, clusterfile:str):

        raise NotImplementedError

    def _read_descriptor_and_weights(self, descriptors: np.ndarray, weights: Optional[np.ndarray]):
        '''Read the descriptor file provided.

        Args:
            descriptor: a np.ndarray, the coordinates of your sample points
            weights: a np.ndarray, the weights of your sample points
        
        Returns:
            nsamples: int, the number of samples
            descriptor: np.ndarray, the descriptor
            weight: np.ndarray, the weight
            totw: float, the sum of weight array
        '''

        # Sanity check
        assert self.dimension == descriptors.shape[1], \
               'Please check the number of columns of your descriptor.' \
               'It does not equal the number of descriptors. '
        if self.weighted and weights is None:
            raise ValueError('Please provide the weight array.')

        nsamples = descriptors.shape[0]
        if weights is None:
            weights = np.ones(nsamples)
        totw = np.sum(weights)
        weights /= totw

        return nsamples, descriptors, weights, totw

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

        labels = []
        grid_npoints = np.zeros(self.ngrid, dtype=int)
        grid_weight = np.zeros(self.ngrid)
        grid_neighbour = {i: [] for i in range(self.ngrid)}

        # assign samples to its cloeset grid
        for descriptor in self.descriptor:
            descriptor2grid = pammr2(self.period, descriptor, self.grid_pos)
            labels.append(np.argmin(descriptor2grid))

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
        if self.verbose:
            print(" Precalculate distance matrix between grid points")

        dist_matrix = np.zeros((self.ngrid, self.ngrid))
        for i in range(self.ngrid):
            dist_matrix[i, i] = np.inf
            dist_matrix[i, i + 1:] = \
                pammr2(self.period, self.grid_pos[i + 1:], self.grid_pos[i])
            dist_matrix[i + 1:, i] = dist_matrix[i, i + 1:]

        return dist_matrix

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

    def computes_localization(self, mindist: np.ndarray):
        delta = 1 / self.nsamples
        # only one of the methods can be used at a time
        if self.fspread > 0:
            self.fpoints = -1.
        cov = covariance(self.grid_pos, self.period, self.grid_weight, 1.0)
        print(f'Global eff. dim. {effdim(cov)}')

        if self.periodic:
            tune = sum(self.period ** 2)
        else:
            tune = np.trace(cov)
        sigma2 = np.full(self.ngrid, tune, dtype=float)

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
        h_tr_normed = np.zeros(self.ngrid)
        for i in range(self.ngrid):
            if self.verbose and (i % 100 == 0):
                print(f'  {i} / {self.ngrid}')
            wlocal, flocal[i] = localization(self.period, self.grid_pos, self.grid_pos[i], self.grid_weight, sigma2[i])
            if self.fpoints > 0:
                sigma2, flocal, wlocal = self._localization_based_on_fraction_of_points(sigma2, flocal, i, delta, tune)
            else:
                if sigma2[i] < flocal[i]:
                    sigma2, flocal, wlocal = \
                        self._localization_based_on_fraction_of_spread(sigma2, flocal, i, mindist)
            h_invs[i], normkernels[i], qscut2[i], h_tr_normed[i] = \
                self._bandwidth_estimation_from_localization(wlocal, flocal, i)

        qscut2 *= self.qs ** 2

        return h_invs, normkernels, qscut2, sigma2, flocal, h_tr_normed

    def _localization_based_on_fraction_of_points(self, sigma2, flocal, idx, delta, tune):

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
    
    def _localization_based_on_fraction_of_spread(self, sigma2, flocal, idx, mindist):

        sigma2[idx] = mindist[idx]
        wlocal, flocal[idx] = localization(self.period, self.descriptor, self.grid_pos, self.grid_weight, sigma2[idx])

        return sigma2, flocal, wlocal

    def _bandwidth_estimation_from_localization(self, wlocal, flocal, idx):

        cov_i = covariance(self.grid_pos, self.period, wlocal, flocal[idx])
        nlocal = flocal[idx] * self.nsamples
        self.local_dimension[idx] = effdim(cov_i)
        cov_i = oas(cov_i, nlocal, self.descriptor.shape[1])
        # cov_i_inv = np.linalg.inv(cov_i)
        h = (4. / (self.local_dimension[idx] + 2.)) ** (2. / (self.local_dimension[idx] + 4.)) * nlocal ** (-2. / (self.local_dimension[idx] + 4.)) * cov_i
        h_tr_normed = np.trace(h) / h.shape[0]
        h_inv = np.linalg.inv(h)
        _, logdet_h = np.linalg.slogdet(h)
        normkernel = self.dimension * np.log(2 * np.pi) + logdet_h
        qscut2 = np.trace(cov_i)

        return  h_inv, normkernel, qscut2, h_tr_normed
    
    def computes_kernel_density_estimation(self, h_inv: np.ndarray, normkernel: np.ndarray, igrid: np.ndarray, neighbour: dict):

        if self.verbose:
            print('Computing kernel density on reference points')
        d = self.descriptor.shape[1]
        self.kdecut2 = 9 * (np.sqrt(d) + 1) ** 2
        prob = np.full(self.ngrid, -np.inf)
        for i in range(self.ngrid):
            if self.verbose and (i % 100 == 0):
                print(f'  {i} / {self.ngrid}')
            dummd1s = mahalanobis(self.period, self.grid_pos, self.grid_pos[i], h_inv)

            for j, dummd1 in enumerate(dummd1s):
                
                if dummd1 > self.kdecut2:
                    lnk = -0.5 * (normkernel[j] + dummd1) + np.log(self.grid_weight[j])
                    prob[i] = _update_prob(prob[i], lnk)
                else:
                    neighbours = neighbour[j][neighbour[j] != igrid[i]]
                    dummd1s = mahalanobis(self.period, self.descriptor[neighbours], self.grid_pos[i], h_inv[j])
                    lnks = -0.5 * (normkernel[j] + dummd1s) + np.log(self.weight[neighbours])
                    prob[i] = _update_probs(prob[i], lnks)

        prob -= np.log(self.totw)

        return prob
    
    def quick_shift(self, probs: np.ndarray, dist_matrix: np.ndarray, idmindist: np.ndarray, qscut2: np.ndarray):

        if self.verbose:
            print("Starting Quick-Shift")
        idxroot = np.full(self.ngrid, -1, dtype=int)
        for i in range(self.ngrid):
            if idxroot[i] != -1:
                continue
            if self.verbose and (i % 1000 == 0):
                print(f'  {i} / {self.ngrid}')
            qspath = np.zeros(self.ngrid, dtype=int)
            qspath[0] = i
            counter = 0
            while qspath[counter] != idxroot[qspath[counter]]:
                if self.gs > 0:
                    idxroot[qspath[counter]] = self.gs_next(qspath[counter], probs, self.gs, dist_matrix)
                else:
                    idxroot[qspath[counter]] = self.qs_next(qspath[counter], idmindist[qspath[counter]], probs, dist_matrix, qscut2[qspath[counter]])
                if idxroot[idxroot[qspath[counter]]] != -1:
                    break
                counter += 1
                qspath[counter] = idxroot[qspath[counter - 1]]
                
            for j in range(counter):
                idxroot[qspath[j]] = idxroot[idxroot[qspath[counter]]]
        cluster_centers = np.concatenate(np.argwhere(idxroot == np.arange(self.ngrid)))

        return cluster_centers, idxroot
    
    def gs_next(self, idx: int, probs: np.ndarray, n_shells: int, distmm: np.ndarray):

        neighs = np.copy(self.gabriel[idx])
        for i in range(1, n_shells):
            nneighs = np.full(self.ngrid, False)
            for j in range(self.ngrid):
                if neighs[j]:
                    nneighs |= self.gabriel[j]
            neighs |= nneighs
        gs_next = idx
        dmin = np.inf
        for j in range(self.ngrid):
            if probs[j] > probs[idx] and \
               distmm[idx, j] < dmin and \
               neighs[j]:
                gs_next = j
                dmin = distmm[idx, j]

        return gs_next
    
    def qs_next(self, idx:int, idxn: int, probs: np.ndarray, distmm: np.ndarray, lambda_: float):

        dmin = np.inf
        qs_next = idx
        if probs[idxn] > probs[idx]:
            qs_next = idxn
        for j in range(self.ngrid):
            if probs[j] > probs[idx] and \
               distmm[idx, j] < dmin and \
               distmm[idx, j] < lambda_:
                qs_next = j
                dmin = distmm[idx, j]

        return qs_next
    
    def post_process(self, cluster_centers: np.ndarray, idxroot: np.ndarray, probs: np.ndarray):

        nk = len(cluster_centers)
        to_merge = np.full(nk, False)
        normpks = logsumexp(idxroot, probs, 1)
        for k in range(nk):
            dummd1 = np.exp(logsumexp(idxroot, probs, cluster_centers[k]) - normpks)
            to_merge[k] = dummd1 > self.thrpcl
        # merge the outliers
        for i in range(nk):
            if not to_merge[k]:
                continue
            dummd1yi1 = cluster_centers[i]
            dummd1 = np.inf
            for j in range(nk):
                if to_merge[k]:
                    continue
                dummd2 = pammr2(self.period, 
                                self.grid_pos[idxroot[dummd1yi1]], 
                                self.grid_pos[idxroot[j]])
                if dummd2 < dummd1:
                    dummd1 = dummd2
                    cluster_centers[i] = j
            idxroot[idxroot == dummd1yi1] = cluster_centers[i]
        if sum(to_merge) > 0:
            cluster_centers = np.concatenate(np.argwhere(idxroot == np.arange(self.ngrid)))
            if self.verbose:
                print(f'Nk-{len(cluster_centers)} clusters were merged'
                      f'into other clusters.')
            nk = len(cluster_centers)
            for i in range(nk):
                dummd1yi1 = cluster_centers[i]
                cluster_centers[i] = getidmax(idxroot, probs, cluster_centers[i])
                idxroot[idxroot == dummd1yi1] = cluster_centers[i]
        
        return cluster_centers, idxroot

    def bootstrap(self, nbootstrap: int):

        if nbootstrap == 0:
            for i in range(self.ngrid):
                self.prelerr[i] = np.log(
                    np.sqrt(((self.mindist[i] * 2 * np.pi) ** (-self.local_dimension[i]) / 
                             np.exp(self.probs[i]) - 1) / 
                             self.nsamples))
            self.pabserr = self.prelerr + self.probs
            return self.pabserr, self.prelerr
        
        prob_boot = np.full((nbootstrap, self.ngrid), -np.inf)
        bs = np.zeros((nbootstrap, self.ngrid))
        for n in range(nbootstrap):
            if self.verbose:
                print(f'Bootstrap {n + 1} / {nbootstrap}')
            n_bootsample = 0
            for j in range(self.ngrid):
                nsample = np.random.binomial(self.nsamples, self.grid_npoints[j] / self.nsamples)
                if nsample == 0:
                    continue
                dummd2 = np.log(nsample / self.grid_npoints[j]) * \
                    np.log(self.grid_weight[j])
                n_bootsample += nsample
                for i in range(self.ngrid):
                    dummd1 = mahalanobis(self.period, self.grid_pos[i], self.grid_pos[j], self.h_inv[j])
                    if dummd1 > self.kdecut2:
                        lnk = -0.5 * (self.normkernel[j] + dummd1) + dummd2
                        prob_boot[n, i] = _update_prob(prob_boot[n, i], lnk)
                    else:
                        select = np.random.choice(self.grid_npoints[j], nsample)
                        idx = self.grid_neighbour[j][select]
                        dummd1s = mahalanobis(self.period, self.descriptor[idx], self.grid_pos[i], self.h_inv[j])
                        lnks = -0.5 * (self.normkernel[j] + dummd1s) + np.log(self.weight[idx])
                        prob_boot[n, i] = _update_probs(prob_boot[n, i], lnks)
            prob_boot[n] -= np.log(self.totw) + np.log(n_bootsample / self.nsamples)
            cluster_centers, idxroot = self.quick_shift(prob_boot[n], self.dist_matrix, self.idmindist, self.qscut2)
            print(len(cluster_centers), 'cluster centers')
            for i in range(self.ngrid):
                bs[n, i] = np.argmin(np.abs(cluster_centers - idxroot[i]))
        for i in range(self.ngrid):
            for j in range(nbootstrap):
                self.pabserr[i] += np.exp(2 * _update_prob(prob_boot[j, i], self.probs[i]))
            self.pabserr[i] = np.log(np.sqrt(self.pabserr[i] / (nbootstrap - 1)))
            self.prelerr[i] = self.pabserr[i] - self.probs[i]

        return self.pabserr, self.prelerr, bs


    def get_output(self, outputfile: str):

        nfeature = self.descriptor.shape[1]
        with open(outputfile, 'w') as wfl:
            for i in range(self.ngrid):
                for j in range(nfeature):
                    wfl.write(f'{self.grid_pos[i][j]:>15.4e}')
                wfl.write(f'{np.argmin(self.cluster_centers - self.idxroot[i]):>15d}')
                wfl.write(f'{self.probs[i]:>15.4e}')
                wfl.write(f'{self.pabserr[i]:>15.4e}')
                wfl.write(f'{self.prelerr[i]:>15.4e}')
                wfl.write(f'{self.sigma2[i]:>15.4e}')
                wfl.write(f'{self.flocal[i]:>15.4e}')
                wfl.write(f'{self.grid_weight[i]:>15.4e}')
                wfl.write(f'{self.local_dimension[i]:>15.4e}')
                wfl.write(f'{self.h_tr_normed[i]:>15.4e}')
                wfl.write('\n')

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
