import numpy as np

from toolbox.optim import _GradientDescent
from toolbox.logger import Logger
log = Logger('SeqDICOD')


class SeqDICOD(_GradientDescent):
    '''Convolutional Sparse coding by coordinate descent
    '''
    def __init__(self, pb, M=None, debug=0, **kwargs):
        '''Coordinate descent algorithm

        Parameters
        ----------
        pb: _Problem
            variable and function holder for the Problem
            to solve
        M: int, optional (default : 10S)
            Chunk size for the sequential DICOD
            The default behavior is to take 4 time the size
            of the dicitonary
        debug: int, optional (default: 0)
            Verbosity level, set to 0 for no output
        '''
        log.set_level(max(3-debug, 1)*10)
        debug = max(debug-1, 0)
        super(SeqDICOD, self).__init__(
            pb, debug=debug, **kwargs)
        if 'name' not in kwargs.keys():
            self.name = 'DICODS_' + str(self.id)

        self.M = M

    def _init_algo(self):
        '''Precompute some quantities that are used across iterations
        '''
        # Compute the different constant of the problem
        self.K, self.d, self.s = self.pb.D.shape
        self.L = self.pb.x.shape[1] - self.s + 1

        # set up the chunk size
        self.n_chunk = self.M
        if self.n_chunk is None:
            self.n_chunk = self.L//(4*self.s)

        # Precompute the norm of the dictionary elements and
        # their cross correlation
        self.alpha_k = np.sum(np.mean(self.pb.D*self.pb.D, axis=1),
                              axis=1).reshape((-1, 1))
        self.alpha_k += (self.alpha_k == 0)
        self.pb.compute_DD()
        self.DD = self.pb.DD

        # compute the initial valut for _beta
        self._beta = self.pb.grad()

        # Init runing variables
        self.current_chunk = 0
        self.chunk_size = self.L // self.n_chunk
        log.debug('Chunck size: {}'.format(self.chunk_size))

        self.dz = []

    def p_update(self):
        '''Chose the best update in one chunk and perform it
        '''
        # Extract the chunk
        m0 = self.current_chunk*self.chunk_size
        m1 = (self.current_chunk+1)*self.chunk_size
        self.current_chunk += 1
        if m1 >= self.L:
            self.current_chunk = 0

        Bz = -self._beta[:, m0:m1]
        pt = self.pb.pt[:, m0:m1]
        Z = self.pb.prox(Bz, self.pb.lmbd)
        Z /= self.alpha_k

        # select best coordinate descent
        i0 = np.argmax(abs(Z-pt))
        i0 = np.unravel_index(i0, pt.shape)
        dz = pt[i0] - Z[i0]
        i1 = (i0[0], i0[1]+m0)
        self.pb.pt[i1] = Z[i0]

        self._update_beta(dz, i1[0], i1[1])
        self.dz += [abs(dz)]
        if len(self.dz) > self.n_chunk:
            del self.dz[0]
        return np.max(self.dz)

    def _update_beta(self, dz, k, t):
        '''Update the univariates optim solution
        '''
        pz = self._beta[k, t]
        off = max(0, self.s-t-1)
        d = max(0, t-self.s+1)
        ll = len(self._beta[k, d:t+self.s])
        self._beta[:, d:t+self.s] -= self.DD[:, k, off:off+ll]*dz
        self._beta[k, t] = pz
