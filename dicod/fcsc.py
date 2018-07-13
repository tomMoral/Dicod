import logging
import numpy as np
from numpy import linalg as LA
from numpy.fft import rfft as fft, irfft as ifft


from ._gradient_descent import _GradientDescent
from .multivariate_convolutional_coding_problem import next_fast_len


log = logging.getLogger('dicod')


l1 = lambda x: np.sum(np.abs(x))
l2 = lambda x: np.sum(x * x)


MU_MAX = 1e5


class FCSC(_GradientDescent):
    """Fast Iterative Soft Thresholding Algorithm

    Parameters
    ----------
    problem: _Problem
        problem we aim to solve
    f_theta: str or float, optional (default: 'fista')
        update of the momentum coefficient.
        If f_theta is a float, use this as a fix rate
        if {'k2', 'fista'} define a theta_k and the coef
        is (theta_k-1)/theta_(k+1).
    eta: float > 1, optional (default: 1.1)
        Line search parameter. Greater mean quicker
        convergence but more instable.
    fix: bool, optional (default: false)
        If set to True, use a fix update step size.
    debug: int, optional (default: 0)
        Set verbosity level
    """
    def __init__(self, problem, tau=1.05, debug=0, tol2=1e-16, **kwargs):
        super(FCSC, self).__init__(
            problem, debug=debug, **kwargs)

        self.tau = tau
        self.tol = tol2

        self.name = 'FCSC_' + str(self.id)

    def _init_algo(self):
        X = self.pb.x  # d, T
        D = self.pb.D  # K, d, S
        self.z = self.pb.pt  # K, L

        self.K, self.d = K, _ = D.shape[0], self.pb.d  # noqa: F841
        self.z_slice = [slice(0, d) for d in self.z.shape]
        self.beta = self.pb.lmbd

        # Init momentum scaling term
        self.mu_t = 1

        shape = X.shape[-1] + D.shape[-1] - 1

        # Frequential domain representation
        self.fft_shape = fft_shape = next_fast_len(shape)
        Xh = fft(X, n=fft_shape)
        w = Xh.shape[-1]
        self.z_fft_shape = (K, ) + Xh.shape[-1:]
        self.D_fft = D_fft = fft(D, n=fft_shape)

        # Precompute constants to accelerate frequency domain computations
        self.DtD_fft = (D_fft[:, None].conj() * D_fft[None]).mean(axis=2)
        self.Dtx_fft = np.mean(D_fft.conj() * Xh[None], axis=1)  # K, w

        self.Id = 1e-10 * np.eye(K)

        # Precompute eigen decomposition to accelerate the inversion of A
        # self.eigh = LA.eigh(I[None, :, :]+DtD_fft.swapaxes(0, 1).T)
        # self.eigh += (LA.inv(self.eigh[1]),)

        self.z_fft = LA.solve((self.Id[None, :, :] +
                               self.DtD_fft.swapaxes(0, 1).T),
                              self.Dtx_fft.T).T
        assert self.z_fft.shape == (K, w), (self.z_fft.shape, K, w)
        self.z = ifft(self.z_fft)[self.z_slice]

        self.t = np.copy(self.z)
        self.t_fft = np.copy(self.z_fft)

        self.lambda_t = np.zeros_like(self.z)
        self.lambda_fft = np.zeros_like(self.z_fft)

    def p_update(self):
        '''Update the pt in the objective direction
        '''
        self._z_subproblem()

        self._t_subproblem()
        self.lambda_t += self.z - self.t
        self.lambda_fft = fft(self.lambda_t, n=self.fft_shape).reshape(
            (self.K, -1))

        self.mu_t = min(self.tau * self.mu_t, MU_MAX)
        self.pb.pt = self.z
        return ((self.z - self.t)**2).mean()

    def _z_subproblem(self):
        A = self.mu_t * np.eye(self.K) + np.transpose(self.DtD_fft,
                                                      axes=(2, 0, 1))
        b = (self.Dtx_fft + self.mu_t * (self.t_fft - self.lambda_fft)).T
        self.z_fft = LA.solve(A, b).T
        self.z = ifft(self.z_fft.reshape(self.z_fft_shape)
                      )[self.z_slice]

    def _t_subproblem(self):
        if self.mu_t > 0:
            self.t = self.lambda_t + self.z
            self.t[:] = self._prox(self.t, self.beta / self.mu_t)
        else:
            self.t[:] = 0

        self.t_fft = fft(self.t, n=self.fft_shape).reshape((self.K, -1))

    @staticmethod
    def _prox(z, mu):
        return np.sign(z) * np.clip(abs(z) - mu, 0, np.inf)

    def _line_search(self):
        '''Line search for the maximal possible update
        '''
        L = self.L
        lmbd = self.pb.lmbd
        grad = self.pb.grad(self.yn)
        if self.fixe:
            return self.pb.prox(self.yn - self.alpha * grad,
                                lmbd * self.alpha)

        fy = self.pb.x - self.pb.reconstruct(self.yn)
        fy = np.sum(fy * fy) / 2

        def prox(L):
            return self.pb.prox(self.yn - grad / L, lmbd / L)

        def diff_y(x):
            return x - self.yn

        def Q(x, dx, L):
            return (fy + (dx * grad).sum() + L / 2 * l2(dx) + lmbd * l1(x))

        def cond(x, dx, L):
            return (self.pb.cost(x) <= Q(x, dx, L))

        pz = prox(L)
        dz = diff_y(pz)

        while not cond(pz, dz, L) and not np.isclose(dz, 0).all():
            log.debug(str(self.pb.cost(pz)), Q(pz, dz, L))
            L *= self.eta
            pz = prox(L)
            dz = diff_y(pz)

        self.L = L
        return pz

    def _theta(self):
        '''Update the momentum coefficient
        '''
        if self.f_theta == 'k2':
            return 2 / (self.t + 3)
        elif self.f_theta == 'fista':
            return (1 + np.sqrt(1 + 4 * self.tk * self.tk)) / 2
        elif type(self.f_theta) == float:
            return 1 - self.f_theta
        return 0.2
