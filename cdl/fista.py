import numpy as np


from toolbox.optim import _GradientDescent
from toolbox.logger import Logger

log = Logger(name='Fista')

l1 = lambda x: np.sum(np.abs(x))
l2 = lambda x: np.sum(x*x)


class FISTA(_GradientDescent):
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
    def __init__(self, problem, f_theta='fista',
                 eta=1.2, fixe=False, debug=0, **kwargs):
        log.set_level(max(3-debug, 1)*10)
        debug = max(debug-1, 0)
        super(FISTA, self).__init__(
            problem, debug=debug, **kwargs)

        self.eta = eta
        self.fixe = fixe

        # Theta coefficient for the momentum and update rule
        self.tk = 1
        self.f_theta = f_theta

        self.name = 'Fista_'+str(self.id)

    def _init_algo(self):
        self.pb.compute_DD()
        self.alpha = 1/self.pb.L
        self.L = self.pb.L/10
        self.yn = self.pb.pt

    def p_update(self):
        '''Update the pt in the objective direction
        '''
        xk1 = self.pb.pt

        self.pb._update(self._line_search())

        #Update momentum information
        dpt = (self.pb.pt - xk1)
        tk1 = self._theta()
        self.yn = self.pb.pt + (self.tk-1)/tk1*dpt
        self.tk = tk1

        # Return dz
        #return np.max(abs(dpt))
        return np.sqrt(np.sum(dpt*dpt))

    def _line_search(self):
        '''Line search for the maximal possible update
        '''
        L = self.L
        lmbd = self.pb.lmbd
        grad = self.pb.grad(self.yn)
        if self.fixe:
            return self.pb.prox(self.yn - self.alpha*grad,
                                lmbd*self.alpha)

        fy = self.pb.x - self.pb.reconstruct(self.yn)
        fy = np.sum(fy*fy)/2

        p = lambda l: self.pb.prox(self.yn - grad/l, lmbd/l)
        d = lambda x: x - self.yn
        Q = lambda x, dx, l: (fy + (dx*grad).sum() + l/2*l2(dx) +
                              lmbd*l1(x))
        cond = lambda x, dx, l: (self.pb.cost(x) <= Q(x, dx, l))

        pz = p(L)
        dz = d(pz)

        while not cond(pz, dz, L) and not np.isclose(dz, 0).all():
            log.debug(str(self.pb.cost(pz)), Q(pz, dz, L))
            L *= self.eta
            pz = p(L)
            dz = d(pz)

        self.L = L
        return pz

    def _theta(self):
        '''Update the momentum coefficient
        '''
        if self.f_theta == 'k2':
            return 2/(self.t+3)
        elif self.f_theta == 'fista':
            return (1 + np.sqrt(1+4*self.tk*self.tk))/2
        elif type(self.f_theta) == float:
            return 1-self.f_theta
        return 0.2
