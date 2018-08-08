import numpy as np


class _SparseCoding(object):
    """Metaclass for sparse coding"""
    def __init__(self, D, alpha=0.1, max_iter=100,
                 conv=False, **kwargs):
        self.D = D  # Dictionaries
        self.K, self.s = D.shape
        self.alpha = alpha  # Sparsity importance
        self.max_iter = max_iter
        self.conv = conv
        self.z = None

    def fit_transform(self, Y, *args, **kwargs):
        self.fit(Y, *args, **kwargs)
        return self.transform(Y, *args, **kwargs)

    def fit(self, Y, *args, **kwargs):
        pass

    def transform(self, Y, *args, **kwargs):
        raise SparseCodingImplementationError(
            'transform not implemented for {}'.format(self.__class__))

    def dictionary_update(self, D):
        '''Update the dicitonary used to encode
        '''
        self.D = D

    def reconstruct(self):
        '''Reconstruct the signal from the code and the dictionary
        If conv is true, use a convolutional dictionary coding
        '''
        if self.z is None:
            raise SparseCodingImplementationError(
                "No code has been computed")
        if self.conv:
            return np.sum([np.convolve(dk, zk)
                           for dk, zk in zip(self.D, self.z)], axis=0)
        else:
            return self.D.T.dot(self.z)

    def _fobj(self, Y):
        residual = Y - self.reconstruct()
        return (np.sum(residual*residual) +
                abs(self.z).sum()*self.alpha)


class SparseCodingImplementationError(Exception):
    """Error for in the implementation of a _Decomposition object"""
    def __init__(self, msg):
        super(SparseCodingImplementationError, self).__init__()
        self.msg = msg

    def __str__(self):
        return repr(self.msg)


class _Decompose(object):
    """Abstract class for decomposition algorithm"""
    def __init__(self, *args, **kwargs):
        pass

    def fit_transform(self, X, y=None, *args):
        self.fit(X, y)
        return self.transform(X)

    def fit(self, X, y, *args):
        raise DecomposeImplementationError('fit not implemented for {}'
                                           ''.format(self.__class__))

    def transform(self, X, *args):
        raise DecomposeImplementationError('transform not implemented for {}'
                                           ''.format(self.__class__))


class DecomposeImplementationError(Exception):
    """Error for in the implementation of a _Decomposition object"""
    def __init__(self, msg):
        super(DecomposeImplementationError, self).__init__()
        self.msg = msg

    def __str__(self):
        return repr(self.msg)


#from .convolutionalFeatSignSearch import FeatureSignSearch
#from .convolutionalPursuit import ConvolutionalPursuit
