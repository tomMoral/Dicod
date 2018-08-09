import logging
import numpy as np
from time import time

from .utils import CostCurve, get_log_rate

log = logging.getLogger('dicod')
if len(log.handlers) == 0:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    log.addHandler(handler)


class _LassoSolver(object):
    '''Generic functionalities for Lasso solvers

    Parameters
    ----------
    max_iter: int (default: 1e6)
        maximal number of iteration for the considered method.
    timeout : float (default: 40)
        stop the computations after this number of seconds.
    stop : str (default: 'dz')
        stopping criterion. Value should be one of {'none', 'dz'}.
            - 'none': stop only with timeout and max_iter.
            - 'dz': stop when the update value dz if less than tol.
    tol : float (default: 1e-10)
        tolerance level for the stopping criterion.
    logging : boolean (default: False)
        If set to True, compute and store the cost_curve.
    log_rate : numeric or str (default: 'log1.6')
        scheduler to log the cost during the different iterations.
    name : str (default: None)
        name of the solver.
    debug : int,
        verbosity of the logger.
    '''

    id_solver = 0

    def __init__(self, max_iter=1e6, timeout=40, stop='dz', tol=1e-10,
                 logging=False, log_rate='log1.6', name=None, debug=0):
        log.setLevel(max(3 - debug, 1) * 10)

        # Logging system
        self.logging = logging
        self.log_rate = get_log_rate(log_rate)

        # Stopping criterions
        self.tol = tol
        self.stop = stop
        self.timeout = timeout
        self.max_iter = max_iter

        # Name for debugging
        self.id = _LassoSolver.id_solver
        _LassoSolver.id_solver += 1
        self.name = name if name else 'Solver' + str(self.id)

    def solve(self, problem, **kwargs):
        self.pb = problem
        self.alpha = 1 / problem.L
        self.change_param(**kwargs)
        return self

    def __repr__(self):
        return self.name

    def _init_algo(self):
        pass

    def update(self):
        '''Update the parameters given

        Parameters
        ----------
        grad: list, optional (default: None)
            list of the gradient for each parameters
        '''
        if self.finished:
            return True
        if self.it == 0:
            self.start()
        self.it += 1

        # Perform the computation for the iteration
        t_start_it = time()
        dz = self.p_update()
        self.time += time() - t_start_it
        if self.logging and self.it >= self.next_log:
            self.record(self.it, self.time, self.pb.cost())
        stop = self._stop(dz)
        if stop:
            self.end()
        return stop

    def p_update(self):
        '''Update the parameters
        '''
        grad = self.pb.grad(self.pb.pt)
        self.p_grad = grad
        self.pb -= self.alpha * grad
        return np.sum(abs(grad))

    def _stop(self, dz):
        '''Implement stopping criterion
        '''

        if self.it >= self.max_iter or self.time >= self.timeout:
            self.finished = True
            log.info("{} - Stop - Reach timeout or maxiter"
                     "".format(self.__repr__()))
            return True

        if self.stop == 'none':
            return False

        # If |x_n-x_n-1| < tol, stop
        if self.stop == 'dz' and dz < self.tol:
            self.finished = True
            log.info('{} - Stop - No advance X'.format(self.__repr__()))
            return self.finished

        # Other stopping criterion
        if self.stop == 'up5':
            return False
        else:
            return False

    def start(self):
        log.info('{} - Start'.format(self))
        self.reset()
        self.t_start = time()
        self._init_algo()
        self.t_init = time() - self.t_start
        self.time += self.t_init
        if self.logging:
            self.record(self.it, self.time, self.pb.cost())

    def end(self):
        self.runtime = time()-self.t_start
        self.cost = self.pb.cost()
        if self.logging:
            self.record(self.it, self.time, self.cost)
        log.debug('{} - End - iteration {}, time {:.4}s'
                  .format(self, self.it, self.time))
        log.debug('Total time: {:.4}s'.format(self.runtime))

    def record(self, it, time, cost):
        self.cost_curve.iterations.append(it + 1)
        self.cost_curve.times.append(time)
        self.cost_curve.pobj.append(cost)
        self.next_log = self.log_rate(it + 1)

    def reset(self):
        # initiate loop variables
        self.it = 0
        self.time = 0
        self.finished = False
        self.cost_curve = CostCurve([], [], [])

    def fit(self, pb):
        self.reset()
        self.pb = pb
        stop = False
        while not stop:
            stop = self.update()
