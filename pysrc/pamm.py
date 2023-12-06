import os
import warnings
from typing import Optional
import numpy as np
from numba import jit
from rich.progress import track
from skmatter.feature_selection import FPS
from pysrc.utils._pamm import effdim, localization, read_period, covariance, oas, logsumexp, getidmax, gs_next, qs_next
from pysrc.utils.dist import pammr2, pammrij, mahalanobis, get_squared_dist_matrix
from pysrc.utils.graph import get_gabriel_graph


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

        # only one of the methods can be used at a time
        if self.fspread > 0:
            self.fpoints = -1.
        self.kdecut2 = 9 * (np.sqrt(self.dimension) + 1) ** 2

        if self.qs < 0:
            raise ValueError('The QS scaling should be positive')
        #if self.bootstrap < 0:
        #    raise ValueError('The number of iterations should be positive')
        if len(self.period) != self.dimension:
            raise ValueError('Check the number of periodic dimensions!')

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None, 
            sample_weight: Optional[np.ndarray] = None):

        self.nsamples, self.descriptor, self.weight = \
            self._read_descriptor_and_weights(x, sample_weight)
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
        self.grid_npoints, self.grid_weight, self.grid_neighbour, self.iminij = self.clustering()
        if self.verbose:
            print(" Precalculate distance matrix between grid points")
        self.dist_matrix = get_squared_dist_matrix(self.grid_pos, self.period)
        self.mindist = np.min(self.dist_matrix, axis=1)
        idmindist = np.argmin(self.dist_matrix, axis=1)
        self.gabriel = get_gabriel_graph(self.dist_matrix)
        h_invs, normkernels, qscut2, self.sigma2, self.flocal, self.h_tr_normed = \
            self.computes_localization(self.mindist)
        self.h_inv = h_invs
        self.normkernel = normkernels
        self.idmindist = idmindist
        self.qscut2 = qscut2
        self.probs = \
            self.computes_kernel_density_estimation(h_invs, normkernels, \
                                                    igrid, self.grid_neighbour)
        cluster_centers, idxroot = \
            self.quick_shift(self.probs, self.dist_matrix, idmindist, qscut2)
        cluster_centers, idxroot = \
            self.post_process(cluster_centers, idxroot, self.probs)
        self.cluster_centers = cluster_centers
        self.idxroot = idxroot
        # self.generate_probability_model()

        return cluster_centers

    def _read_descriptor_and_weights(self, descriptors: np.ndarray, weights: Optional[np.ndarray]):
        '''Read the descriptor file provided.

        Args:
            descriptor: a np.ndarray, the coordinates of your sample points
            weights: a np.ndarray, the weights of your sample points
        
        Returns:
            nsamples: int, the number of samples
            descriptor: np.ndarray, the descriptor
            weight: np.ndarray, the weight
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

        return nsamples, descriptors, weights

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

        return grid_npoints, grid_weight, grid_neighbour, labels

    def computes_localization(self, mindist: np.ndarray):

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

        flocal = np.zeros(self.ngrid)
        h_invs = np.zeros((self.ngrid, self.dimension, self.dimension))
        normkernels = np.zeros(self.ngrid)
        qscut2 = np.zeros(self.ngrid)
        h_tr_normed = np.zeros(self.ngrid)
        for i in track(range(self.ngrid), description='Estimating kernel density bandwidths'):
            wlocal, flocal[i] = localization(self.period, self.grid_pos, self.grid_pos[i],
                                             self.grid_weight, sigma2[i])
            if self.fpoints > 0:
                sigma2, flocal, wlocal = \
                    self._localization_based_on_fraction_of_points(sigma2, flocal, i,
                                                                   1 / self.nsamples, tune)
            elif sigma2[i] < flocal[i]:
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
            warnings.warn(" Warning: localization smaller than voronoi,"
                          " increase grid size (meanwhile adjusted localization)!")
        while flocal[idx] < lim:
            sigma2[idx] += tune
            wlocal, flocal[idx] = localization(self.period, self.grid_pos, self.grid_pos[idx],
                                               self.grid_weight, sigma2[idx])
        j = 1
        while True:
            if flocal[idx] > lim:
                sigma2[idx] -= tune / 2 ** j
            else:
                sigma2[idx] += tune / 2 ** j
            wlocal, flocal[idx] = localization(self.period, self.grid_pos, self.grid_pos[idx],
                                               self.grid_weight, sigma2[idx])
            if abs(flocal[idx] - lim) < delta:
                break
            j += 1

        return sigma2, flocal, wlocal

    def _localization_based_on_fraction_of_spread(self, sigma2, flocal, idx, mindist):

        sigma2[idx] = mindist[idx]
        wlocal, flocal[idx] = localization(self.period, self.descriptor, self.grid_pos,
                                           self.grid_weight, sigma2[idx])

        return sigma2, flocal, wlocal

    def _bandwidth_estimation_from_localization(self, wlocal, flocal, idx):

        cov_i = covariance(self.grid_pos, self.period, wlocal, flocal[idx])
        nlocal = flocal[idx] * self.nsamples
        self.local_dimension[idx] = effdim(cov_i)
        cov_i = oas(cov_i, nlocal, self.dimension)
        h = (4. / (self.local_dimension[idx] + 2.)) ** (2. / (self.local_dimension[idx] + 4.))\
            * nlocal ** (-2. / (self.local_dimension[idx] + 4.)) * cov_i
        h_tr_normed = np.trace(h) / h.shape[0]
        h_inv = np.linalg.inv(h)
        _, logdet_h = np.linalg.slogdet(h)
        normkernel = self.dimension * np.log(2 * np.pi) + logdet_h
        qscut2 = np.trace(cov_i)

        return  h_inv, normkernel, qscut2, h_tr_normed

    def computes_kernel_density_estimation(self, h_inv: np.ndarray, normkernel: np.ndarray, igrid: np.ndarray, neighbour: dict):

        prob = np.full(self.ngrid, -np.inf)
        for i in track(range(self.ngrid), description='Computing kernel density on reference points'):
            dummd1s = mahalanobis(self.period, self.grid_pos, self.grid_pos[i], h_inv)
            for j, dummd1 in enumerate(dummd1s):
                if dummd1 > self.kdecut2:
                    lnk = -0.5 * (normkernel[j] + dummd1) + np.log(self.grid_weight[j])
                    prob[i] = _update_prob(prob[i], lnk)
                else:
                    neighbours = neighbour[j][neighbour[j] != igrid[i]]
                    dummd1s = mahalanobis(self.period, self.descriptor[neighbours],
                                          self.grid_pos[i], h_inv[j])
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
            qspath = []
            qspath.append(i)
            while qspath[-1] != idxroot[qspath[-1]]:
                if self.gs > 0:
                    idxroot[qspath[-1]] = gs_next(qspath[-1], probs, self.gs,
                                                  dist_matrix, self.gabriel)
                else:
                    idxroot[qspath[-1]] = qs_next(qspath[-1], idmindist[qspath[-1]],
                                                  probs, dist_matrix, qscut2[qspath[-1]])
                if idxroot[idxroot[qspath[-1]]] != -1:
                    break
                qspath.append(idxroot[qspath[-1]])
            idxroot[qspath] = idxroot[idxroot[qspath[-1]]]
        cluster_centers = np.concatenate(np.argwhere(idxroot == np.arange(self.ngrid)))

        return cluster_centers, idxroot

    def post_process(self, cluster_centers: np.ndarray, idxroot: np.ndarray, probs: np.ndarray):

        nk = len(cluster_centers)
        to_merge = np.full(nk, False)
        self.normpks = logsumexp(np.ones(len(idxroot)), probs, 1)
        for k in range(nk):
            dummd1 = np.exp(logsumexp(idxroot, probs, cluster_centers[k]) - self.normpks)
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

        with open(outputfile, 'w') as wfl:
            for i in range(self.ngrid):
                for j in range(self.dimension):
                    wfl.write(f'{self.grid_pos[i][j]:>15.4e}')
                wfl.write(f'{np.argmin(abs(self.cluster_centers - self.idxroot[i])):>15d}')
                wfl.write(f'{self.probs[i]:>15.4e}')
                wfl.write(f'{self.pabserr[i]:>15.4e}')
                wfl.write(f'{self.prelerr[i]:>15.4e}')
                wfl.write(f'{self.sigma2[i]:>15.4e}')
                wfl.write(f'{self.flocal[i]:>15.4e}')
                wfl.write(f'{self.grid_weight[i]:>15.4e}')
                wfl.write(f'{self.local_dimension[i]:>15.4e}')
                wfl.write(f'{self.h_tr_normed[i]:>15.4e}')
                wfl.write('\n')

    def generate_probability_model(self):

        n_cluster = len(self.cluster_centers)
        cluster_mean = np.zeros((n_cluster, self.dimension), dtype=float)
        cluster_cov = np.zeros((n_cluster, self.dimension, self.dimension), dtype=float)
        cluster_weight = np.zeros(n_cluster, dtype=float)

        for k in range(n_cluster):
            print(f'{self.cluster_centers[k]} {logsumexp(self.idxroot, self.probs, self.cluster_centers[k])}')
            cluster_mean[k] = self.grid_pos[self.cluster_centers[k]]
            cluster_weight[k] = np.exp(logsumexp(self.idxroot, self.probs, self.cluster_centers[k])
                                       - self.normpks)
            for _ in range(self.nmsopt):
                msmu = np.zeros(self.dimension, dtype=float)
                tmppks = -np.inf
                for i in range(self.ngrid):
                    dummd1 = mahalanobis(self.period, self.grid_pos[i],
                                         self.grid_pos[self.cluster_centers[k]],
                                         self.h_inv[self.cluster_centers[k]])
                    msw = -0.5 * (self.normkernel[self.cluster_centers[k]] + dummd1) + self.probs[i]
                    tmpmsmu = 0.
                    tmpmsmu = pammrij(self.period, tmpmsmu, self.grid_pos[i],
                                      self.grid_pos[self.cluster_centers[k]])
                    msmu += np.exp(msw) * tmpmsmu
                tmppks = _update_prob(tmppks, msw)
                cluster_mean[k] += msmu / np.exp(tmppks)
            if self.periodic:
                cluster_cov[k] = self._get_lcov_clusterp(self.ngrid, self.nsamples, self.grid_pos,
                                                         self.idxroot, self.grid_pos[k])
                if np.sum(self.idxroot == self.cluster_centers[k]) == 1:
                    cluster_cov[k] = self._get_lcov_clusterp(self.nsamples, self.nsamples,
                                                             self.descriptor,
                                                             self.iminij, self.grid_pos[k])
                    print('Warning: single point cluster!')
            else:
                cluster_cov[k] = self._get_lcov_cluster(self.ngrid, self.grid_pos,
                                                        self.idxroot, self.cluster_centers[k])
                if np.sum(self.idxroot == self.cluster_centers[k]) == 1:
                    cluster_cov[k] = self._get_lcov_cluster(self.nsamples, self.descriptor,
                                                            self.iminij, self.cluster_centers[k])
                    print('Warning: single point cluster!')
                cluster_cov[k] = oas(cluster_cov[k], logsumexp(self.idxroot, self.probs, self.cluster_centers[k]) * self.nsamples, self.dimension)

        with open(self.outputfile + '.pamm', 'w', encoding='utf-8') as wfl:
            wfl.write(f'# PAMMv2 clusters analysis. NSamples: {self.nsamples}'
                        f'Ngrid: {self.ngrid} QSLambda: {self.qs}\n')
            wfl.write('# Dimensionality/NClusters//Pk/Mean/Covariance/Period\n')
            self._write_clusters(wfl, n_cluster, cluster_weight, cluster_mean, cluster_cov)

    def _get_lcov_cluster(self, N: int, x: np.ndarray, clroots: np.ndarray, idcl: int):

        ww = np.zeros(N)
        normww = logsumexp(clroots, self.probs, idcl)
        ww[clroots == idcl] = np.exp(self.probs[clroots == idcl] - normww)
        cov = covariance(x, self.period, ww, np.sum(ww))

        return cov

    def _get_lcov_clusterp(self, N: int, Ntot: int, x: np.ndarray, clroots: np.ndarray, idcl: int):

        ww = np.zeros(N)
        totnormp = logsumexp(np.zeros(N), self.probs, 0)
        cov = np.zeros((self.dimension, self.dimension), dtype=float)
        xx = np.zeros(x.shape, dtype=float)
        ww[clroots == idcl] = np.exp(self.probs[clroots == idcl] - totnormp)
        ww *= Ntot
        nlk = np.sum(ww)
        for i in range(self.dimension):
            xx[:, i] = x[:, i] - round(x[:, i] / self.period[i]) * self.period[i]
            r2 = (np.sum(ww * np.cos(xx[:, i])) / nlk) ** 2 \
                + (np.sum(ww * np.sin(xx[:, i])) / nlk) ** 2
            re2 = (nlk / (nlk - 1)) * (r2 - (1 / nlk))
            cov[i, i] = 1 / (np.sqrt(re2) * (2 - re2) / (1 - re2))

        return cov

    def _write_clusters(self, wfl, n_cluster: int, cluster_weight,
                        cluster_mean: np.ndarray, cluster_cov: np.ndarray):

        wfl.write(f'{self.dimension} {n_cluster}\n')
        for k in range(n_cluster):
            wfl.write(f'{self.cluster_centers[k]:>15d}')
            wfl.write(f'{cluster_weight[k]:>15.8e}')
            for i in range(self.dimension):
                wfl.write(f' {cluster_mean[k, i]:>15.8e}')
            for i in range(self.dimension):
                for j in range(self.dimension):
                    wfl.write(f' {cluster_cov[k, i, j]:>15.8e}')
            if self.periodic:
                for i in range(self.dimension):
                    wfl.write(f' {self.period[i]:>15.8e}')
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
