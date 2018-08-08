import logging
import numpy as np
from math import sqrt
from time import time

log = logging.getLogger('dicod')


def get_log_rate(lr):
    if lr[:3] == 'log':
        inc = float(lr[3:]) if len(lr[3:]) > 0 else 2
        log_rate = (lambda s: lambda t: s*t + (t == 0))(inc)
    elif lr[:3] == 'lin':
        inc = int(lr[3:]) if len(lr[3:]) > 0 else 10
        log_rate = (lambda s: lambda t: t+s)(inc)
    elif lr == 'none':
        log_rate = lambda t: 1e300
    elif type(lr) is int or type(lr) is float:
        inc = log_rate
        log_rate = (lambda s: lambda t: s*t + (t == 0))(inc)
    else:
        assert False, "{} is not a log rate".format(lr)

    return log_rate


class _GradientDescent(object):
    """Class to hold gradient descent properties"""

    id_gd = 0

    def __init__(self, problem, decreasing_rate='sqrt',
                 stop='', tol=1e-10, graphical_cost=None,
                 name=None, debug=0, logging=False,
                 log_rate='log1.6', i_max=1e6, t_max=40):
        '''Gradient Descent handeler

        Parameters
        ----------
        param: list of the Parameters
        alpha: learning rate controler
        grad: function computing the gradient
            given the current parameters
        decreasing_rate: {'sqrt', 'linear'} deacreasing rate
            for the learning rate
        '''
        log.setLevel(max(3-debug, 1)*10)
        debug = max(debug-1, 0)

        self.id = _GradientDescent.id_gd
        _GradientDescent.id_gd += 1

        self.pb = problem
        self.decreasing_rate = decreasing_rate
        self.stop = stop
        self.tol = tol

        # Logging system
        self.logging = logging
        self.log_rate = get_log_rate(log_rate)
        self.i_max = i_max
        self.t_max = t_max

        self.name = name if name is not None else '_GD' + str(self.id)
        self.graph_cost = None

        self.reset()

    def set_param(self, decreasing_rate='sqrt',
                  stop='', tol=1e-10, graphical_cost=None,
                  name=None, debug=0, logging=False,
                  log_rate='log1.6', i_max=1000, t_max=40):
        if debug > 0:
            log.set_level(10)

        self.decreasing_rate = decreasing_rate
        self.stop = stop
        self.tol = tol

        # Logging system
        self.logging = logging
        self.log_rate = get_log_rate(log_rate)
        self.i_max = i_max
        self.t_max = t_max

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

        if self.iteration >= self.i_max or self.t >= self.t_max:
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
