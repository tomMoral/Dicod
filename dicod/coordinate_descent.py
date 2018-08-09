import logging
import numpy as np

from ._lasso_solver import _LassoSolver

log = logging.getLogger('dicod')


class CoordinateDescent(_LassoSolver):
    '''Convolutional Coordinate Descent Algorithm [Kavukcuoglu2010]

    Parameters
    ----------
    pb: _Problem
        variable and function holder for the Problem
        to solve
    debug: int, optional (default: 0)
        Verbosity level, set to 0 for no output
    '''

    def _init_algo(self):
        '''Precompute some quantities that are used across iterations
        '''
        self.K, self.d, self.s = self.pb.D.shape
        self.L = self.pb.x.shape[1] - self.s + 1
        self._beta = self.pb.grad()
        self.alpha_k = np.sum(np.mean(self.pb.D*self.pb.D, axis=1),
                              axis=1).reshape((-1, 1))
        self.alpha_k += (self.alpha_k == 0)
        self.DD = self.pb.DD

    def p_update(self):
        '''Chose the best update and perform it
        '''
        Bz = -self._beta
        Z = self.pb.prox(Bz, self.pb.lmbd)
        Z /= self.alpha_k

        # select best coordinate descent
        i0 = np.argmax(abs(Z-self.pb.pt))
        i0 = np.unravel_index(i0, self.pb.pt.shape)
        dz = self.pb.pt[i0] - Z[i0]
        self.pb.pt[i0] = Z[i0]

        self._update_beta(dz, i0[0], i0[1])
        return abs(dz)

    def _update_beta(self, dz, k, t):
        '''Update the univariates optim solution
        '''
        pz = self._beta[k, t]
        off = max(0, self.s-t-1)
        d = max(0, t-self.s+1)
        ll = len(self._beta[k, d:t+self.s])
        self._beta[:, d:t+self.s] -= self.DD[:, k, off:off+ll]*dz
        self._beta[k, t] = pz
