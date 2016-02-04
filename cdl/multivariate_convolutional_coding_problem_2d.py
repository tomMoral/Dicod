import numpy as np
from scipy.signal import convolve2d

from toolbox.optim.problem import _Problem


class MultivariateConvolutionalCodingProblem2D(_Problem):
    '''Convolutional Sparse coding by coordinate descent

    Parameters
    ----------
    D: array-like (n_dict (K), n_dim(d), height(h_dic), width(w_dic))
        dictionary for the coding problem
    x: array-like (n_dim(d), height(h_sig), width(w_sig))
        signal to encode on the dictionary
    lmbd: float, optional (default: 0.1)
        control the sparsity of the solution
    z0: array-like (K, h_sig-h_dic+1, w_sig-w_dic+1), optional (default: None)
        initial code, if not set, start with the 0 serie
    nonneg: bool, optional (default: False)
        Use the proximal operator to shrink in a non-negative
        way.
    '''
    def __init__(self, D, x, lmbd=0.1, z0=None, nonneg=False, **kwargs):
        self.D = np.array(D)
        self.x = np.array(x)
        if self.D.ndim == 3:
            self.D = self.D[:, :, None]
        if self.D.ndim == 2:
            self.D = self.D[:, None, None]
        if self.x.ndim == 2:
            self.x = self.x[:, None]
        if self.x.ndim == 1:
            self.x = self.x[None, None]
        size = None
        if z0 is None:
            size = (self.D.shape[0], self.x.shape[-2]-self.D.shape[-2]+1,
                    self.x.shape[-1]-self.D.shape[-1]+1)
        super(MultivariateConvolutionalCodingProblem2D, self).__init__(
            z0, size=size, **kwargs)
        self.M, self.d, self.h_dic, self.w_dic = self.D.shape
        self.lmbd = lmbd
        self.nonneg = False
        self.d = self.x.shape[0]

    def compute_DD(self, DD=None):
        self.DD = DD
        if self.DD is None:
            self.DD = np.mean([[[convolve2d(dk0, dk1, mode='full')
                                 for dk0, dk1 in zip(d0, d1)]
                                for d1 in self.D]
                               for d0 in self.D[:, :, ::-1, ::-1]], axis=2)
        self.L = np.sqrt(np.sum(self.DD*self.DD, axis=0).sum(axis=0)).sum()
        return self.DD

    def update_D(self, dD, D=None, DD=None):
        if D is None:
            D = self.D
            if dD is not None:
                D = D + dD
            #D /= np.sqrt(np.sum(D*D, axis=2))[:, :, np.newaxis]
        self.D = D
        self.L = np.sqrt(np.mean(self.D*self.D))
        if DD is None:
            self.compute_DD()
        else:
            self.DD = DD
        return self.D, self.DD

    def Er(self, pt):
        '''Commpute the reconstruction error
        '''
        res = self.x - self.reconstruct(pt)
        return (res*res).sum()/(2*self.d)

    def cost(self, pt=None):
        '''Compute the cost at the given point
        '''
        if pt is None:
            pt = self.pt
        return (self.Er(pt) + self.lmbd*np.sum(abs(pt)))/np.prod(pt.shape[-2:])

    def grad(self, pt=None):
        '''Compute the gradient at the given point
        '''
        if pt is None:
            pt = self.pt
        residual = self.reconstruct(pt) - self.x
        _grad = np.mean([[np.convolve(dk, rk, mode='valid')
                         for dk, rk in zip(Dm, residual)]
                        for Dm in self.D[:, :, ::-1]], axis=1)
        return _grad

    def prox(self, pt=None, lmbd=None):
        '''Compute the proximal operator at the given point
        Can pass an additional argument to compute it for different lambda
        '''
        if pt is None:
            pt = self.pt
        if lmbd is None:
            lmbd = self.lmbd
        if self.nonneg:
            return np.maximum(pt-lmbd, 0)
        else:
            return np.sign(pt)*np.maximum(abs(pt)-lmbd, 0)

    def grad_D(self, pt):
        residual = self.reconstruct(pt) - self.x
        self._grad_D = [[np.convolve(z, rk, mode='valid')
                         for rk in residual]
                        for z in pt[:, ::-1]]
        self._grad_D = np.array(self._grad_D)
        return self._grad_D

    def reconstruct(self, pt):
        '''Reconstruct the signal from the given code
        '''
        return np.sum([[convolve2d(dk, zm) for dk in Dm]
                       for Dm, zm in zip(self.D, pt)], axis=0)
