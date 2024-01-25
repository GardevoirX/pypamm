import warnings
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from numba import jit
from rich.progress import track
from skmatter.feature_selection import FPS
from pysrc.clustering import NearestNeighborClustering
from pysrc.utils._pamm import *
from pysrc.utils.dist import pammr2, pammrij, mahalanobis, get_squared_dist_matrix
from pysrc.utils.graph import get_gabriel_graph


class PAMM:
    """Probabilistic Analysis of Molecular Motifs method.
    
    Args:
        descriptors (np.ndarray): An array of molecular descriptors.
        dimension (int): The number of dimensions.
        ngrid (int): The number of grid points in the descriptors.
        sample_weight (Optional[np.ndarray]): An array of weights for each sample.
        period_text (Optional[str]): A string of periods for each dimension.
        qs (float): Scaling factor used during the QS clustering.
        gs (int): The neighbor shell for gabriel shift.
        nmsopt (int): The number of mean-shift refinement steps.
        thrpcl (float): Clusters with a pk loewr than this value are merged with the NN.
        fspread (float): The fractional variance for bandwidth estimation.
        fpoints (float): The fractional number of grid points.
        outputfile (str): The name of the output file.
        seed (int): The seed for the random number generator.
        verbose (bool): Whether to print verbose output.
        
    Attributes:
        n_clusters (int): The number of clusters.
        cluster_centers (np.ndarray): The coordinates of the cluster centers.
        center_idx (np.ndarray): The indices of the cluster centers.
        labels_ (np.ndarray): The labels of each grid.
        """

    def __init__(self, descriptors:np.ndarray, dimension:int, ngrid:int,
                 sample_weight: Optional[np.ndarray] = None,
                 period_text:Optional[str]= None,
                 qs:float = 1., gs:int = -1,
                 nmsopt:int = 0, thrpcl:float = 0.,
                 fspread:float = -1., fpoints:float = 0.15,
                 outputfile:str = 'out', seed:int = 12345, verbose:bool = False):

        np.random.seed(seed)

        self.descriptor = descriptors
        self.nsamples = descriptors.shape[0]
        self.weight, self._totw = \
            self._read_weights(sample_weight)
        self.dimension = dimension
        self.qs = qs
        self.nmsopt = nmsopt
        self.ngrid = ngrid
        self.fspread = fspread
        self.fpoints = fpoints
        self.period = read_period(period_text, self.dimension)
        self.periodic = False if period_text is None else True
        self.thrpcl = thrpcl
        self.outputfile = outputfile
        self.gs = gs
        self.verbose = verbose

        # only one of the methods can be used at a time
        if self.fspread > 0:
            self.fpoints = -1.
        self.kdecut2 = 9 * (np.sqrt(self.dimension) + 1) ** 2
        self.local_dimension = np.zeros(self.ngrid)

        self.cluster_attributes = create_grid_attributes(self.ngrid)

        if self.qs < 0:
            raise ValueError('The QS scaling should be positive')
        if len(self.period) != self.dimension:
            raise ValueError('Check the number of periodic dimensions!')

    @property
    def grid_pos(self):
        """The coordinates of the grids"""
        return self.descriptor[self.grid_idx]

    @property
    def grid_weight(self):
        """The weights of the grids"""
        if np.any(np.isnan(self.cluster_attributes['weights'])):
            print('The clustering has not been performed yet')
        else:
            return self.cluster_attributes['weights'].to_numpy()

    @property
    def _dist_matrix(self):
        """The distance matrix between grid points"""
        if not hasattr(self, '__dist_matrix'):
            self.__dist_matrix = get_squared_dist_matrix(self.grid_pos, self.period)
        return self.__dist_matrix

    @property
    def _mindist(self):
        """The minimum distance between grid points"""
        if not hasattr(self, '__mindist'):
            self.__mindist = np.min(self._dist_matrix, axis=1)
        return self.__mindist

    @property
    def _idmindist(self):
        """The index of the minimum distance between grid points"""
        if not hasattr(self, '__idmindist'):
            self.__idmindist = np.argmin(self._dist_matrix, axis=1)
        return self.__idmindist

    @property
    def _gabriel(self):
        """The Gabriel graph"""
        if not hasattr(self, '__gabriel'):
            self.__gabriel = get_gabriel_graph(self._dist_matrix, self.ngrid)
        return self.__gabriel

    @property
    def labels_(self):
        """The labels of each grid"""
        if self.cluster_attributes['labels'][0] == -1:
            # A very simple check
            print('The clustering has not been performed yet')
        else:
            return self.cluster_attributes['labels'].to_numpy()

    @property
    def _probs(self):
        """The probabilities of each grid"""
        if np.isnan(self.cluster_attributes['probs'][0]):
            # A very simple check
            print('The clustering has not been performed yet')
        else:
            return self.cluster_attributes['probs'].to_numpy()

    def _pabserr(self):
        if np.isnan(self.cluster_attributes['pabserr'][0]):
            print('The clustering has not been performed yet')
        else:
            return self.cluster_attributes['pabserr'].to_numpy()

    def _prelerr(self):
        if np.isnan(self.cluster_attributes['prelerr'][0]):
            print('The clustering has not been performed yet')
        else:
            return self.cluster_attributes['prelerr'].to_numpy()

    @property
    def _sigma2(self):
        if np.isnan(self.cluster_attributes['sigma2'][0]):
            print('The clustering has not been performed yet')
        else:
            return self.cluster_attributes['sigma2'].to_numpy()

    @property
    def _flocal(self):
        if np.isnan(self.cluster_attributes['flocal'][0]):
            print('The clustering has not been performed yet')
        else:
            return self.cluster_attributes['flocal'].to_numpy()

    @property
    def _h_tr_normed(self):
        if np.isnan(self.cluster_attributes['h_trace_normed'][0]):
            print('The clustering has not been performed yet')
        else:
            return self.cluster_attributes['h_trace_normed'].to_numpy()

    @property
    def center_idx(self):
        """The indices of the cluster centers"""
        if not hasattr(self, '__center_idx'):
            self.__center_idx = np.unique(self.labels_)
        return self.__center_idx

    @property
    def n_clusters(self):
        """The number of clusters"""
        return len(self.center_idx)

    @property
    def cluster_centers(self):
        """The coordinates of the cluster centers"""
        return self.grid_pos[self.center_idx]

    @property
    def _normpks(self):
        if not hasattr(self, '__normpks'):
            self.__normpks = logsumexp(np.ones(self.ngrid), self._probs, 1)
        return self.__normpks

    def fit(self,
            X: np.ndarray,
            y: Optional[np.ndarray] = None):
        """Computing the PAMM algorithm."""

        self.grid_idx = X
        self._grid_npoints, self._grid_neighbour, self._sample_labels_ = self._assign_descriptors_to_grids()
        self._h_invs, self._normkernels, self._qscut2 = self._computes_localization(self._mindist)
        self._computes_kernel_density_estimation(self._h_invs, self._normkernels, \
                                                 self.grid_idx, self._grid_neighbour)
        center_idx, labels = \
            self._quick_shift(self._probs, self._dist_matrix, self._idmindist, self._qscut2)
        center_idx, labels = \
            self._post_process(center_idx, labels, self._probs)

        self.cluster_attributes['labels'] = labels

        #self.generate_probability_model()

        return self

    def _read_weights(self, weights:Optional[np.ndarray] = None) -> Tuple[int, np.ndarray]:
        '''Read the descriptor file provided.

        Args:
            weights: a np.ndarray, the weights of your sample points. Optional
        
        Returns:
            weight: np.ndarray, the weight
            totw: float, the total weight
        '''

        if weights is None:
            weights = np.ones(self.nsamples)
        totw = np.sum(weights)
        weights /= totw
        totw = 1.

        return weights, totw

    def _assign_descriptors_to_grids(self):

        assigner = NearestNeighborClustering(self.period)
        assigner.fit(self.grid_pos)
        labels = assigner.predict(self.descriptor, sample_weight=self.weight)
        grid_npoints = assigner.grid_npoints
        grid_neighbour = assigner.grid_neighbour
        self.cluster_attributes['weights'] = assigner.grid_weight

        return grid_npoints, grid_neighbour, labels

    def _computes_localization(self, mindist: np.ndarray):

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

        self.cluster_attributes['local_dimension'] = self.local_dimension
        self.cluster_attributes['sigma2'] = sigma2
        self.cluster_attributes['flocal'] = flocal
        self.cluster_attributes['h_trace_normed'] = h_tr_normed

        return h_invs, normkernels, qscut2

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

    def _computes_kernel_density_estimation(self,
                                            h_inv: np.ndarray,
                                            normkernel: np.ndarray,
                                            igrid: np.ndarray,
                                            neighbour: dict):

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

        prob -= np.log(self._totw)
        self.cluster_attributes['probs'] = prob

    def _quick_shift(self,
                     probs: np.ndarray,
                     dist_matrix: np.ndarray,
                     idmindist: np.ndarray,
                     qscut2: np.ndarray):

        idxroot = np.full(self.ngrid, -1, dtype=int)
        for i in track(range(self.ngrid), description='Quick-Shift'):
            if idxroot[i] != -1:
                continue
            qspath = []
            qspath.append(i)
            while qspath[-1] != idxroot[qspath[-1]]:
                if self.gs > 0:
                    idxroot[qspath[-1]] = gs_next(qspath[-1], probs, self.gs,
                                                  dist_matrix, self._gabriel)
                else:
                    idxroot[qspath[-1]] = qs_next(qspath[-1], idmindist[qspath[-1]],
                                                  probs, dist_matrix, qscut2[qspath[-1]])
                if idxroot[idxroot[qspath[-1]]] != -1:
                    break
                qspath.append(idxroot[qspath[-1]])
            idxroot[qspath] = idxroot[idxroot[qspath[-1]]]
        cluster_centers = np.concatenate(np.argwhere(idxroot == np.arange(self.ngrid)))

        return cluster_centers, idxroot

    def _post_process(self, cluster_centers: np.ndarray, idxroot: np.ndarray, probs: np.ndarray):

        nk = len(cluster_centers)
        to_merge = np.full(nk, False)
        for k in range(nk):
            dummd1 = np.exp(logsumexp(idxroot, probs, cluster_centers[k]) - self._normpks)
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
        """
        Generates a bootstrap estimate of the standard errors 
        for the given number of bootstrap iterations.

        Args:
            nbootstrap (int): The number of bootstrap iterations to perform.

        Returns:
            tuple: A tuple containing the bootstrap estimate of the absolute errors (pabserr), 
                   the bootstrap estimate of the relative errors (prelerr), 
                   and the bootstrap samples (bs).
        """


        if nbootstrap < 0:
            raise ValueError('The number of iterations should be non-negative')
        _prelerr = np.full(self.ngrid, np.inf)
        _pabserr = np.full(self.ngrid, 0.0)
        if nbootstrap == 0:
            for i in range(self.ngrid):
                _prelerr[i] = np.log(
                    np.sqrt(((self._mindist[i] * 2 * np.pi) ** (-self.local_dimension[i]) /
                             np.exp(self._probs[i]) - 1) /
                             self.nsamples))
            _pabserr = _prelerr + self._probs

        prob_boot = np.full((nbootstrap, self.ngrid), -np.inf)
        bs = np.zeros((nbootstrap, self.ngrid))
        for n in range(nbootstrap):
            if self.verbose:
                print(f'Bootstrap {n + 1} / {nbootstrap}')
            n_bootsample = 0
            for j in range(self.ngrid):
                nsample = np.random.binomial(self.nsamples, self._grid_npoints[j] / self.nsamples)
                if nsample == 0:
                    continue
                dummd2 = np.log(nsample / self._grid_npoints[j]) * \
                    np.log(self.grid_weight[j])
                n_bootsample += nsample
                for i in range(self.ngrid):
                    dummd1 = mahalanobis(self.period, self.grid_pos[i], self.grid_pos[j], self._h_invs[j])
                    if dummd1 > self.kdecut2:
                        lnk = -0.5 * (self._normkernels[j] + dummd1) + dummd2
                        prob_boot[n, i] = _update_prob(prob_boot[n, i], lnk)
                    else:
                        select = np.random.choice(self._grid_npoints[j], nsample)
                        idx = self._grid_neighbour[j][select]
                        dummd1s = mahalanobis(self.period, self.descriptor[idx], self.grid_pos[i], self._h_invs[j])
                        lnks = -0.5 * (self._normkernels[j] + dummd1s) + np.log(self.weight[idx])
                        prob_boot[n, i] = _update_probs(prob_boot[n, i], lnks)
            prob_boot[n] -= np.log(self._totw) + np.log(n_bootsample / self.nsamples)
            cluster_centers, idxroot = self._quick_shift(prob_boot[n], self._dist_matrix, self._idmindist, self._qscut2)
            print(len(cluster_centers), 'cluster centers')
            for i in range(self.ngrid):
                bs[n, i] = np.argmin(np.abs(cluster_centers - idxroot[i]))
        for i in range(self.ngrid):
            for j in range(nbootstrap):
                _pabserr[i] += np.exp(2 * _update_prob(prob_boot[j, i], self._probs[i]))
            _pabserr[i] = np.log(np.sqrt(_pabserr[i] / (nbootstrap - 1)))
            _prelerr[i] = _pabserr[i] - self._probs[i]

        self.cluster_attributes['pabserr'] = _pabserr
        self.cluster_attributes['prelerr'] = _prelerr

        return bs

    def get_output(self, outputfile: str):
        """
        Writes the output of the grid values, labels, probabilities,
        errors, and other attributes to a file.

        Parameters:
            outputfile (str): The name of the output file.

        Returns:
            None
        """

        with open(outputfile, 'w') as wfl:
            for i in range(self.ngrid):
                for j in range(self.dimension):
                    wfl.write(f'{self.grid_pos[i][j]:>15.4e}')
                wfl.write(f'{np.argmin(abs(self.center_idx - self.labels_[i])):>15d}')
                wfl.write(f'{self._probs[i]:>15.4e}')
                wfl.write(f'{self.pabserr[i]:>15.4e}')
                wfl.write(f'{self.prelerr[i]:>15.4e}')
                wfl.write(f'{self._sigma2[i]:>15.4e}')
                wfl.write(f'{self._flocal[i]:>15.4e}')
                wfl.write(f'{self.grid_weight[i]:>15.4e}')
                wfl.write(f'{self.local_dimension[i]:>15.4e}')
                wfl.write(f'{self._h_tr_normed[i]:>15.4e}')
                wfl.write('\n')

    def generate_probability_model(self):
        """
        Generates a probability model based on the given inputs.

        Parameters:
            None

        Returns:
            None
        """

        cluster_mean = np.zeros((self.n_clusters, self.dimension), dtype=float)
        cluster_cov = np.zeros((self.n_clusters, self.dimension, self.dimension), dtype=float)
        cluster_weight = np.zeros(self.n_clusters, dtype=float)

        for k in range(self.n_clusters):
            cluster_mean[k] = self.grid_pos[self.center_idx[k]]
            cluster_weight[k] = np.exp(logsumexp(self.labels_, self._probs, self.center_idx[k])
                                       - self._normpks)
            for _ in range(self.nmsopt):
                msmu = np.zeros(self.dimension, dtype=float)
                tmppks = -np.inf
                for i in range(self.ngrid):
                    dummd1 = mahalanobis(self.period, self.grid_pos[i],
                                         self.grid_pos[self.center_idx[k]],
                                         self._h_invs[self.center_idx[k]])
                    msw = -0.5 * (self._normkernels[self.center_idx[k]] + dummd1) + self._probs[i]
                    tmpmsmu = 0.
                    tmpmsmu = pammrij(self.period, tmpmsmu, self.grid_pos[i],
                                      self.grid_pos[self.center_idx[k]])
                    msmu += np.exp(msw) * tmpmsmu
                tmppks = _update_prob(tmppks, msw)
                cluster_mean[k] += msmu / np.exp(tmppks)
            cluster_cov[k] = self._update_cluster_cov(k)

        '''with open(self.outputfile + '.pamm', 'w', encoding='utf-8') as wfl:
            wfl.write(f'# PAMMv2 clusters analysis. NSamples: {self.nsamples}'
                        f'Ngrid: {self.ngrid} QSLambda: {self.qs}\n')
            wfl.write('# Dimensionality/NClusters//Pk/Mean/Covariance/Period\n')
            self._write_clusters(wfl, self.n_clusters, cluster_weight, cluster_mean, cluster_cov)'''

        return cluster_weight, cluster_mean, cluster_cov

    def _update_cluster_cov(self, k: int):

        if self.periodic:
            cov = self._get_lcov_clusterp(self.ngrid, self.nsamples, self.grid_pos,
                                          self.labels_, self.center_idx[k], self._probs)
            if np.sum(self.labels_ == self.center_idx[k]) == 1:
                cov = self._get_lcov_clusterp(self.nsamples, self.nsamples,
                                              self.descriptor, self._sample_labels_,
                                              self.center_idx[k], self.weight)
                print('Warning: single point cluster!')
        else:
            cov = self._get_lcov_cluster(self.ngrid, self.grid_pos,
                                         self.labels_, self.center_idx[k], self._probs)
            if np.sum(self.labels_ == self.center_idx[k]) == 1:
                cov = self._get_lcov_cluster(self.nsamples, self.descriptor,
                                             self._sample_labels_, self.center_idx[k], self.weight)
                print('Warning: single point cluster!')
            cov = oas(cov, logsumexp(self.labels_, self._probs, self.center_idx[k]) * self.nsamples, self.dimension)

        return cov

    def _get_lcov_cluster(self, N: int, x: np.ndarray, clroots: np.ndarray, idcl: int, probs: np.ndarray):

        ww = np.zeros(N)
        normww = logsumexp(clroots, probs, idcl)
        ww[clroots == idcl] = np.exp(probs[clroots == idcl] - normww)
        cov = covariance(x, self.period, ww, np.sum(ww))

        return cov

    def _get_lcov_clusterp(self, N: int, Ntot: int, x: np.ndarray, clroots: np.ndarray, idcl: int, probs: np.ndarray):

        ww = np.zeros(N)
        totnormp = logsumexp(np.zeros(N), probs, 0)
        cov = np.zeros((self.dimension, self.dimension), dtype=float)
        xx = np.zeros(x.shape, dtype=float)
        ww[clroots == idcl] = np.exp(probs[clroots == idcl] - totnormp)
        ww *= Ntot
        nlk = np.sum(ww)
        for i in range(self.dimension):
            xx[:, i] = x[:, i] - np.round(x[:, i] / self.period[i]) * self.period[i]
            r2 = (np.sum(ww * np.cos(xx[:, i])) / nlk) ** 2 \
                + (np.sum(ww * np.sin(xx[:, i])) / nlk) ** 2
            re2 = (nlk / (nlk - 1)) * (r2 - (1 / nlk))
            cov[i, i] = 1 / (np.sqrt(re2) * (2 - re2) / (1 - re2))

        return cov

    def _write_clusters(self, wfl, n_cluster: int, cluster_weight,
                        cluster_mean: np.ndarray, cluster_cov: np.ndarray):

        wfl.write(f'{self.dimension} {n_cluster}\n')
        for k in range(n_cluster):
            wfl.write(f'{self.center_idx[k]:>15d}')
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
