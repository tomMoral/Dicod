import logging
import numpy as np
from numpy.linalg import LinAlgError

from ._lasso_solver import _LassoSolver

log = logging.getLogger('dicod')


class FSS(_LassoSolver):
    """Algorithm solving a sparse coding problem with
    a feature sign search (based on Ng_05_sparse)

    Usage
    -----
    pb = _Problem()
    fss = FeatureSignSearch(pb)
    finished = False
    while not finished:
        finished = fss.p_update()
    """
    def __init__(self, pb, n_zero_coef=100, relax=1e-8,
                 debug=0, **kwargs):
        '''Feature Sign Search Algorihtm

        Parameters
        ----------
        pb: inherits from _Problem
            object to hold the different variables of the problem
        n_zero_coef: int, optional (default: 100)
            Maximal number of newly active coefficients
        relax: float, optional (default: 1e-8)
            Threshold to consider there is no movement anymore
        '''
        super(FSS, self).__init__(problem=pb, debug=debug,
                                  **kwargs)
        self.relax = relax
        self.n_zero_coef = n_zero_coef
        self.name = 'FSS_'+str(self.id)

    def p_update(self):
        '''One step for the Algorihtm
        Update the active set and solve the
        quadratic subproblem
        '''
        self._active_set()
        log.debug('Active Set: {:5}'.format(
                  len(self.active_indices)))
        dz = self._subproblem()
        self.dcost_dz = self.pb.grad()
        return dz

    def _init_algo(self):
        '''Precompute some quantities that are used across iterations
        '''
        log.debug('Initiate FFS...')
        _, self.d, self.s = self.pb.D.shape
        self.L = len(self.pb.x) - self.s + 1

        self.pb.compute_DD()
        self.DD = self.pb.DD
        self.B = np.array([np.mean([np.convolve(dk, xk, 'valid')
                                   for dk, xk in zip(d, self.pb.x)], axis=0)
                           for d in self.pb.D[:, :, ::-1]])
        self.active_indices = []
        for i, j in zip(*(self.pb.pt.nonzero())):
            self.active_indices.append((i, j))
        self.dcost_dz = self.pb.grad()

    def _stop(self, dz):
        '''Stopping criterion - Stop if all the 0 coef have a small gradient
        or if another stopping criterion is specified
        '''
        finished = super(FSS, self)._stop(dz)
        stop = (abs(self.dcost_dz[self.pb.pt == 0]) < self.pb.lmbd).all()

        if stop:
            log.info('FeatureSignSearch reach optimal solution')
        self.finished = finished | stop
        return self.finished

    def _active_set(self):
        '''Select a subset of the coefficient to add to the
        active indices based on the gradient
        '''
        r = self.dcost_dz.shape[1]
        compt = 0

        # Sort the dcost_dz to select the biggest gradient
        tmp = abs(self.dcost_dz)
        tmp = (tmp[:, :-2] <= tmp[:, 1:-1]) & (tmp[:, 2:] <= tmp[:, 1:-1])
        d = tmp.shape[0]
        tmp = np.c_[np.zeros((d, 1)), tmp, np.zeros((d, 1))]
        tmp = np.ones(tmp.shape) + 0.2*tmp
        # coefficients first
        order_grad = abs(self.dcost_dz*tmp).flatten().argsort()[::-1]
        # t = np.sum(abs(self.dcost_dz) > self.pb.lmbd)
        # p = abs(self.dcost_dz.flatten())[order_grad[:t]]
        # p = p / p.sum()
        # order_grad = np.random.choice(order_grad[:t], t,
        #                               replace=False, p=p)

        # Select 0 coefficient with gradient over gamma
        for k in order_grad:
            i = k // r
            j = k % r
            if (self.pb.pt[i, j] == 0 and
                    abs(self.dcost_dz[i, j]) > self.pb.lmbd):
                self.active_indices += [(i, j)]
                compt += 1
            if (compt == self.n_zero_coef or
                    abs(self.dcost_dz[i, j]) < self.pb.lmbd):
                break

    def _subproblem(self):
        '''Solve a QP-subproblem for the active set
        in acordance with the sign vector computed before
        '''
        dz = 0
        self.loop = True
        self._create_subproblem()
        while self.loop:
            log.debug('Number of active coefficients: {:5}'.format(
                len(self.active_indices)))

            # Compute the solution of the quadratic subproblem
            z_new = self._sol_SP()
            self.loop &= (z_new != self.pt_fs).any()
            z_sol = self._discrete_linesearch(z_new)

            # Update solution
            for k, (i, j) in enumerate(self.active_indices):
                dz += (self.pb.pt[i, j] - z_sol[k])**2
                self.pb.pt[i, j] = z_sol[k]

            #shrink the subproblem
            self._shrink_SP(z_sol)

            # Supproblem stopping criterion
            grad_fs = abs(2*self.A_fs.dot(self.pt_fs) + self.b_fs +
                          self.pb.lmbd*self.pt_sign)
            self.loop &= (grad_fs > self.relax).any()
        return np.sqrt(dz)

    def _create_subproblem(self):
        '''Create a QP-subproblem for the active set
        '''
        num_activ = len(self.active_indices)
        # Compute the quadratic subproblem
        self.A_fs = np.zeros((num_activ, num_activ))
        self.b_fs = np.zeros(num_activ)
        self.pt_fs = np.zeros(num_activ)
        self.pt_sign = np.zeros(num_activ)
        for i, (ind_i1, ind_t1) in enumerate(self.active_indices):
            for j, (ind_i2, ind_t2) in enumerate(self.active_indices):
                if 0 <= (ind_t2-ind_t1+(self.s-1)) < 2*self.s-1:
                    Dv = self.DD[ind_i1][ind_i2][ind_t1-ind_t2+(self.s-1)]
                else:
                    Dv = 0
                self.A_fs[i, j] = Dv

            self.b_fs[i] = -2*self.B[ind_i1, ind_t1]
            self.pt_fs[i] = self.pb.pt[ind_i1, ind_t1]
            self.pt_sign[i] = ((np.sign(self.pt_fs[i]) +
                               (self.pt_fs[i] == 0) *
                               (2*(self.dcost_dz[ind_i1][ind_t1] < 0)-1)))

    def _sol_SP(self):
        '''Solution of the QP-subproblem at one step
        '''
        try:
            sol = -.5*np.linalg.inv(self.A_fs).dot(
                self.b_fs + self.pb.lmbd*self.pt_sign)
            assert np.max(sol) <= 1e7
            return sol
        except (LinAlgError, AssertionError, ValueError):
            log.warning('Singular value in QP-subproblem, '
                        'using approximation')
            n = self.A_fs.shape[0]
            return -.5*np.linalg.inv(self.A_fs+1e-7*np.eye(n)).dot(
                self.b_fs + self.pb.lmbd*self.pt_sign)

    def _shrink_SP(self, z_sol):
        '''Select non zero subproblem
        '''
        ind = z_sol.nonzero()[0]
        self.active_indices = [self.active_indices[i] for i in ind]
        self.A_fs = self.A_fs[ind][:, ind]
        self.b_fs = self.b_fs[ind]
        self.pt_fs = z_sol[ind]
        self.pt_sign = np.sign(self.pt_fs)

    def _obj_qp(self, S):
        '''Compute the cost of a quadratic subproblem
        '''
        return (S.T.dot(self.A_fs.dot(S)) + self.b_fs.dot(S) +
                self.pb.lmbd * abs(S).sum())

    def _discrete_linesearch(self, z_new):
        '''Perform a discrete line search between z and z_new
        '''
        # Find all 0 crossing on the segment [S, S_new]
        # Add some value for the cases where some
        # coefficients where equal to 0
        l = 0.1
        crossing = set([1, 0])
        for i in range(len(z_new)):
            if np.sign(z_new[i]) != self.pt_sign[i]:
                if self.pt_fs[i] != 0:
                    crossing.add(self.pt_fs[i]/(self.pt_fs[i]-z_new[i]))
                elif l > 1e-20:
                    crossing.add(l)
                    l /= 2
        crossing = list(crossing)

        # Choose the best out of the crossing point
        dz = z_new-self.pt_fs
        l_z1 = [self.pt_fs+lc*dz for lc in crossing]
        l_z1 = [z*(np.sign(z) == self.pt_sign) for z in l_z1]
        z_cost = [self._obj_qp(z) for z in l_z1]
        i0 = len(z_cost)-1 - np.argmin(z_cost[::-1])
        if not crossing[i0]:
            crossing = np.r_[0, np.logspace(-20, -1)]
            l_z1 = [self.pt_fs+lc*dz for lc in crossing]
            l_z1 = [z*(np.sign(z) == self.pt_sign) for z in l_z1]
            z_cost = [self._obj_qp(z) for z in l_z1]
            i0 = len(z_cost)-1 - np.argmin(z_cost[::-1])
            if not crossing[i0]:
                self.grad_fs1 = (abs(2*self.A_fs.dot(self.pt_fs) + self.b_fs +
                                 self.pb.lmbd*self.pt_sign))
                self.grad_fs2 = (abs(2*self.A_fs.dot(z_new) + self.b_fs +
                                 self.pb.lmbd*self.pt_sign))
                self.keep = [crossing, z_cost, z_new, self.pt_fs, self.A_fs,
                             self.b_fs, self.pt_sign]
                log.debug('No update', )
                import IPython
                IPython.embed()
                self.loop = False
        return l_z1[i0]
