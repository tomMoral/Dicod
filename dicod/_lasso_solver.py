import logging
import numpy as np
from math import sqrt
from time import time

from .utils import get_log_rate

log = logging.getLogger('dicod')


class _LassoSolver(object):
    """Class to hold gradient descent properties"""

    id_solver = 0

    def __init__(self, stop='', tol=1e-10, graphical_cost=None,
                 name=None, debug=0, logging=False,
                 log_rate='log1.6', max_iter=1e6, timeout=40):
        '''Generic functionalities for Lasso solvers

        Parameters
        ----------
        param: list of the Parameters
        alpha: learning rate controler
        max_iter: int (default: 1e6)
            maximal number of iteration for the considered method.
        '''
        log.setLevel(max(3-debug, 1)*10)
        debug = max(debug-1, 0)

        self.id = _LassoSolver.id_solver
        _LassoSolver.id_solver += 1

        self.stop = stop
        self.tol = tol

        # Logging system
        self.logging = logging
        self.log_rate = get_log_rate(log_rate)
        self.max_iter = max_iter
        self.timeout = timeout

        self.name = name if name is not None else '_GD' + str(self.id)
        self.graph_cost = None

        self.reset()

    def set_param(self, stop='', tol=1e-10, graphical_cost=None,
                  name=None, debug=0, logging=False,
                  log_rate='log1.6', max_iter=1000, timeout=40):
        if debug > 0:
            log.set_level(10)

        self.stop = stop
        self.tol = tol

        # Logging system
        self.logging = logging
        self.log_rate = get_log_rate(log_rate)
        self.max_iter = max_iter
        self.timeout = timeout

        self.name = name if name is not None else '_GD' + str(self.id)
        self.graph_cost = None
        if graphical_cost is not None:
            self.graph_cost = dict(name=graphical_cost, curve=self.name)

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
        if self.iteration == 0:
            self.start()
        self.iteration += 1
        dz = self.p_update()
        self.t = time() - self.t_start
        if self.iteration >= self.next_log and self.logging:
            log.log_obj(name='cost' + str(self.id), obj=np.copy(self.pb.pt),
                        iteration=self.iteration, fun=self.pb.cost,
                        graph_cost=self.graph_cost, time=self.t,
                        levl=50)
            self.next_log = self.log_rate(self.iteration)
        stop = self._stop(dz)
        if stop:
            self.end()
        return stop

    def p_update(self):
        '''Update the parameters
        '''
        grad = self.pb.grad(self.pb.pt)
        self.p_grad = grad
        lr = self._get_lr()
        self.pb -= lr*grad
        return lr*np.sum(grad)

    def _get_lr(self):
        '''Auxillary funciton, return the learning rate
        '''
        lr = self.alpha
        if self.decreasing_rate == 'sqrt':
            lr /= sqrt(self.iteration)
        elif self.decreasing_rate == 'linear':
            lr /= self.iteration
        elif self.decreasing_rate == 'k2':
            lr *= 2/(self.iteration+2)
        elif hasattr(self.decreasing_rate, '__call__'):
            lr *= self.decreasing_rate(self.iteration)
        return lr

    def _stop(self, dz):
        '''Implement stopping criterion
        '''

        if self.iteration >= self.max_iter or self.t >= self.timeout:
            self.finished = True
            log.info("{} - Stop - Reach timeout or maxiter"
                     "".format(self.__repr__()))
            return True

        if self.stop == 'none':
            return False

        # If |x_n-x_n-1| < tol, stop
        if dz < self.tol:
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
        self._init_algo()
        if self.logging:
            log.log_obj(name='cost'+str(self.id), obj=np.copy(self.pb.pt),
                        iteration=0.7, fun=self.pb.cost,
                        graph_cost=self.graph_cost, time=time()-self.t_start)
        self.next_log = self.log_rate(self.iteration)

    def end(self):
        self.runtime = time()-self.t_start
        if self.logging:
            log.log_obj(name='cost'+str(self.id), obj=self.pb.pt,
                        iteration=self.iteration, fun=self.pb.cost,
                        graph_cost=self.graph_cost, time=self.t)
        log.debug('{} - End - iteration {}, time {:.4}s'
                  .format(self, self.iteration, self.t))
        log.debug('Total time: {:.4}s'.format(self.runtime))

    def reset(self):
        # initiate loop variables
        self.finished = False
        self.iteration = 0
        self.t_start = time()
        self.t = 0

    def fit(self, pb):
        self.pb = pb
        self.reset()
        stop = False
        while not stop:
            stop = self.update()
