import numpy as np
from numpy.fft import rfft as fft, irfft as ifft

from toolboxTom.optim.problem import _Problem
from scipy.signal import fftconvolve


class MultivariateConvolutionalCodingProblem(_Problem):
    '''Convolutional Sparse coding by coordinate descent

    Parameters
    ----------
    D: array-like (n_dict (M), n_dim(d), length(K))
        dictionary for the coding problem
    x: array-like (n_dim(d), length(T))
        signal to encode on the dictionary
    lmbd: float, optional (default: 0.1)
        control the sparsity of the solution
    z0: array-like (M, T-K+1), optional (default: None)
        initial code, if not set, start with the 0 serie
    nonneg: bool, optional (default: False)
        Use the proximal operator to shrink in a non-negative
        way.
    '''
    def __init__(self, D, x, lmbd=0.1, z0=None, nonneg=False, **kwargs):
        self.D = np.array(D)
        self.x = np.array(x)
        if self.D.ndim == 2:
            self.D = self.D[:, None]
        if self.x.ndim == 1:
            self.x = self.x[None, :]
        size = None
        if z0 is None:
            size = (self.D.shape[0], self.x.shape[-1] - self.D.shape[-1] + 1)
        super(MultivariateConvolutionalCodingProblem, self).__init__(
            z0, size=size, **kwargs)
        self.M, self.d, self.K = self.D.shape
        self.lmbd = lmbd
        self.nonneg = False
        self.d = self.x.shape[0]

        self._compute_constant()

    def compute_DD(self, DD=None):
        self.DD = DD
        if self.DD is None:
            self.DD = np.mean([[[fftconvolve(dk, dk1)
                                 for dk, dk1 in zip(d, d1)]
                                for d1 in self.D]
                               for d in self.D[:, :, ::-1]], axis=2)

        return self.DD

    def _compute_constant(self):
        """Precompute fft of X and D to fasten the gradient computations"""
        p, X_shape = self.x.shape[0], self.x.shape[-1]
        fft_shape = X_shape + self.D.shape[-1] - 1

        # Frequential domain representation
        self.fft_shape = fft_shape = next_fast_len(int(fft_shape))
        self.X_fft = X_fft = fft(self.x, n=fft_shape)
        self.D_fft = D_fft = fft(self.D, n=fft_shape)

        # Reshape so that all the variableas have the same dimensions
        # [K, p, T]
        self.X_fft = X_fft = self.X_fft[None]

        # Precompute constants to accelerate frequency domain computations
        self.DtD_fft = (D_fft[:, None].conj() * D_fft[None]
                        ).mean(axis=2, keepdims=False)
        self.DtX_fft = (D_fft.conj() * X_fft).mean(axis=1, keepdims=False)

        # Lipchitz constant
        self.L = np.linalg.norm(self.DtD_fft, axis=(0, 1), ord=2).max()

        # Store extra dimensions
        self.T = p * np.prod(X_shape)

    def update_D(self, dD, D=None):
        if D is None:
            D = self.D
            if dD is not None:
                D = D + dD
        self.D = D
        self._compute_constant()
        return self.D

    def Er(self, pt):
        '''Commpute the reconstruction error
        '''
        res = self.x - self.reconstruct(pt)
        return (res * res).sum() / (2 * self.d)

    def cost(self, pt=None):
        '''Compute the cost at the given point
        '''
        if pt is None:
            pt = self.pt
        return self.Er(pt) + self.lmbd * np.sum(abs(pt))

    def grad_slow(self, pt=None):
        '''Compute the gradient at the given point
        '''
        if pt is None:
            pt = self.pt
        residual = self.reconstruct(pt) - self.x
        _grad = np.mean([[fftconvolve(rk, dk, mode='valid')
                         for dk, rk in zip(Dm, residual)]
                        for Dm in self.D[:, :, ::-1]], axis=1)
        return _grad

    def grad(self, pt=None):
        '''Compute the gradient at the given point
        '''
        if pt is None:
            pt = self.pt
        z = pt
        z_slice = [slice(0, d) for d in z.shape]
        z_fft = fft(z, n=self.fft_shape)
        Gh = np.sum(self.DtD_fft * z_fft[None], axis=1)
        Gh -= self.DtX_fft
        out = ifft(Gh).real[z_slice]
        return out

    def prox(self, pt=None, lmbd=None):
        '''Compute the proximal operator at the given point
        Can pass an additional argument to compute it for different lambda
        '''
        if pt is None:
            pt = self.pt
        if lmbd is None:
            lmbd = self.lmbd
        if self.nonneg:
            return np.maximum(pt - lmbd, 0)
        else:
            return np.sign(pt) * np.maximum(abs(pt) - lmbd, 0)

    def grad_D(self, pt):
        residual = self.reconstruct(pt) - self.x
        self._grad_D = [[fftconvolve(z, rk, mode='valid')
                         for rk in residual]
                        for z in pt[:, ::-1]]
        self._grad_D = np.array(self._grad_D)
        return self._grad_D

    def reconstruct(self, pt):
        '''Reconstruct the signal from the given code
        '''
        return np.sum([[fftconvolve(dk, zm) for dk in Dm]
                       for Dm, zm in zip(self.D, pt)], axis=0)


def next_fast_len(target):
    """
    Find the next fast size of input data to `fft`, for zero-padding, etc.

    SciPy's FFTPACK has efficient functions for radix {2, 3, 4, 5}, so this
    returns the next composite of the prime factors 2, 3, and 5 which is
    greater than or equal to `target`. (These are also known as 5-smooth
    numbers, regular numbers, or Hamming numbers.)

    Parameters
    ----------
    target : int
        Length to start searching from.  Must be a positive integer.

    Returns
    -------
    out : int
        The first 5-smooth number greater than or equal to `target`.

    Examples
    --------
    On a particular machine, an FFT of prime length takes 133 ms:

    >>> from scipy import fftpack
    >>> min_len = 10007  # prime length is worst case for speed
    >>> a = np.random.randn(min_len)
    >>> b = fftpack.fft(a)

    Zero-padding to the next 5-smooth length reduces computation time to
    211 us, a speedup of 630 times:

    >>> fftpack.helper.next_fast_len(min_len)
    10125
    >>> b = fftpack.fft(a, 10125)

    Rounding up to the next power of 2 is not optimal, taking 367 us to
    compute, 1.7 times as long as the 5-smooth size:

    >>> b = fftpack.fft(a, 16384)

    """
    hams = (8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48,
            50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128,
            135, 144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 250,
            256, 270, 288, 300, 320, 324, 360, 375, 384, 400, 405, 432, 450,
            480, 486, 500, 512, 540, 576, 600, 625, 640, 648, 675, 720, 729,
            750, 768, 800, 810, 864, 900, 960, 972, 1000, 1024, 1080, 1125,
            1152, 1200, 1215, 1250, 1280, 1296, 1350, 1440, 1458, 1500, 1536,
            1600, 1620, 1728, 1800, 1875, 1920, 1944, 2000, 2025, 2048, 2160,
            2187, 2250, 2304, 2400, 2430, 2500, 2560, 2592, 2700, 2880, 2916,
            3000, 3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750, 3840,
            3888, 4000, 4050, 4096, 4320, 4374, 4500, 4608, 4800, 4860, 5000,
            5120, 5184, 5400, 5625, 5760, 5832, 6000, 6075, 6144, 6250, 6400,
            6480, 6561, 6750, 6912, 7200, 7290, 7500, 7680, 7776, 8000, 8100,
            8192, 8640, 8748, 9000, 9216, 9375, 9600, 9720, 10000)

    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target - 1)):
        return target

    # Get result quickly for small sizes, since FFT itself is similarly fast.
    if target <= hams[-1]:
        from scipy.fftpack.helper import bisect_left
        return hams[bisect_left(hams, target)]

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            # Quickly find next power of 2 >= quotient
            p2 = 2**((quotient - 1).bit_length())

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match
