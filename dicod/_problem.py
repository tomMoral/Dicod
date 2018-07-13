import numpy as np


class ImplementationError(Exception):
    """Implementation Error"""
    def __init__(self, msg, cls):
        super(ImplementationError, self).__init__()
        self.msg = msg
        self.cls = cls

    def __repr__(self):
        return self.cls+'-'+self.msg


class _Problem(object):
    """Meta class to handle optimisation problem"""
    def __init__(self, x0=None, size=None):
        super(_Problem, self).__init__()
        self.x0 = x0
        if self.x0 is None:
            assert size is not None, 'No size for the Problem'
            self.x0 = np.zeros(size)
        self.sizes = self.x0.shape
        self.L = 10
        self.reset()

    def cost(self):
        raise ImplementationError('cost not implemented', self.__class__)

    def grad(self, pt=None):
        raise ImplementationError('grad not implemented', self.__class__)

    def prox(self, pt=None):
        raise ImplementationError('prox not implemented', self.__class__)

    def reset(self):
        self.pt = np.copy(self.x0)

    def __iadd__(self, update):
        self.pt += update

    def __isub__(self, update):
        self.pt -= update

    def __setitem__(self, k, update):
        self.pt[k] = update

    def __getitem__(self, k):
        return self.pt[k]

    def _update(self, update):
        self.pt = update
