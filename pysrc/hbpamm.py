import warnings
import MDAnalysis as mda
import numpy as np

from MDAnalysis.analysis.distances import distance_array

MAXPARS = 4

def parse_cell(box) -> np.ndarray:
    if not box is None:
        return [eval(num) for num in box.split(',')]
    else:
        parse_xyz_cell_comment()
        
def parse_xyz_cell_comment():
    raise NotImplementedError


class HBPAMM:
    def __init__(self, ta, td, th, clusterfile=None, box=None, alpha=1,
                 cutoff=5., vghb='1', delta=1, zeta=0, dosad=False,
                 npt_mode=False, weighted=False) -> None:

        # some default value of parameters
        self.filename = None
        self.vta = []
        self.vtd = []
        self.vth = []
        self.nk = -1 # number of gaussians in the mixture
        self.dopamm = False

        self.ta = ta.split(',')
        self.td = td.split(',')
        self.th = th.split(',')
        self._too_many_types_warning(self.ta, 'acceptor')
        self._too_many_types_warning(self.td, 'donor')
        self._too_many_types_warning(self.th, 'hydrogen')
        self.clusterfile = clusterfile
        if not clusterfile is None:
            self.dopamm = True
        self.cell = parse_cell(box)
        self.alpha = alpha
        self.mucutoff = cutoff
        self.vghb = [eval(num) for num in vghb.split(',')]
        self.delta = delta
        self.zeta = zeta
        self.dosad = dosad
        self.npt_mode = npt_mode
        self.weighted = weighted

        if self.dosad and (not self.dopamm):
            raise NotImplementedError("Error: cannot compute lifetime statistics without cluster data!")
        #self.icell = inv(self.cell)
        
    def run(self, trajectory: str):
        feature_collection = []
        weight_collection = []
        u = mda.Universe(trajectory)
        self.masktypes = self._assign_atomtypes(u)
        if self.dopamm:
            self._read_gaussian_parameters()
        if not self.npt_mode:
            self.cell = [self.cell for i in range(len(u.trajectory))]
        else:
            raise NotImplementedError
        for iframe in range(0, len(u.trajectory), self.delta):
            h_atom, donor, acceptor = self._assign_atomtypes(u)
            h2a_dist = distance_array(u.trajectory[iframe]._pos[h_atom],
                                     u.trajectory[iframe]._pos[acceptor],
                                     self.cell[iframe])
            h2d_dist = distance_array(u.trajectory[iframe]._pos[h_atom],
                                     u.trajectory[iframe]._pos[donor],
                                     self.cell[iframe])
            a2d_dist = distance_array(u.trajectory[iframe]._pos[acceptor],
                                     u.trajectory[iframe]._pos[donor],
                                     self.cell[iframe]) # r
            mu = h2d_dist.reshape(h2d_dist.shape[0], 1, h2d_dist.shape[1]) + \
                 h2a_dist.reshape(h2a_dist.shape[0], h2a_dist.shape[1], 1)
            nu = h2d_dist.reshape(h2d_dist.shape[0], 1, h2d_dist.shape[1]) - \
                 h2a_dist.reshape(h2a_dist.shape[0], h2a_dist.shape[1], 1)
            # first dimension for H, second dimension for acceptor, third for donor
            had_pairs = [(iH, iacceptor, idonor) 
                         for iH, iacceptor, idonor in np.argwhere(mu < self.mucutoff)
                         if acceptor[iacceptor] != donor[idonor]]
            feature = np.array([(nu[(iH, iacceptor, idonor)],
                                 mu[(iH, iacceptor, idonor)],
                                 a2d_dist[(iacceptor, idonor)])
                                 for (iH, iacceptor, idonor) in had_pairs])
            if self.weighted:
                weight = 1 / ((feature[:, 1] - feature[:, 0]) *
                              (feature[:, 1] + feature[:, 0]) *
                              feature[:, 2])
            else:
                weight = None
            feature_collection.append(feature)
            weight_collection.append(weight)
            if self.dopamm:
                # 418-430
                raise NotImplementedError
            else:
                for ipair, _ in enumerate(feature):
                    print(f'{feature[ipair, 0]:.8f} '
                          f'{feature[ipair, 1]:.8f} '
                          f'{feature[ipair, 2]:.8f} ',
                          end='')
                    if not weight is None:
                        print(f'{weight[ipair]:.8f}', end='')
                    print()
            if self.dopamm:
                # 441-470
                raise NotImplementedError

        return np.concatenate(feature_collection), \
               np.concatenate(weight_collection)

    def _too_many_types_warning(self, types: list, descriptor: str):
        if len(types) > MAXPARS:
            warnings.warn(f'Too many {descriptor} types specified on command line,'         
                          'exceeding the limitation of the Fortran version.')

    def _assign_atomtypes(self, u: mda.Universe) -> (np.ndarray, np.ndarray, np.ndarray):
        h_atom = u.select_atoms(f'name {" ".join(self.th)}').ids - 1
        donor= u.select_atoms(f'name {" ".join(self.td)}').ids - 1
        acceptor = u.select_atoms(f'name {" ".join(self.ta)}').ids - 1

        return h_atom, donor, acceptor

    def _read_gaussian_patameters(self):

        with open(self.clusterfile, 'r') as rfl:
            content = rfl.readlines()
        read_parameters = False
        cluster_counter = 0
        for line in content:
            if line[0] == '#':
                continue
            if read_parameters:
                parameters = np.array([eval(num) for num in line.split()])
                cluster_D[cluster_counter] = parameters[0]
                cluster_mean[cluster_counter] = parameters[1:1+D]
                cluster_cov[cluster_counter] = parameters[1+D:].reshape((D, D))
                cluster_det[cluster_counter] = np.linalg.det(cluster_cov[cluster_counter])
                cluster_counter += 1
            else:
                D, Nk = [eval(num) for num in line.split()]
                assert D == 3, "Only 3D descriptors are supported"
                cluster_D = np.zeros(Nk, dtype=float)
                cluster_mean = np.zeros((Nk, D), dtype=float)
                cluster_cov = np.zeros((Nk, D, D), dtype=float)
                cluster_det = np.zeros(Nk, dtype=float)
                cluster_icov = np.zeros((Nk, D, D), dtype=float)
                read_parameters = True
        cluster_lnorm = np.log(1 / np.sqrt((2 * np.pi) ** cluster_D * cluster_det))

        raise NotImplementedError