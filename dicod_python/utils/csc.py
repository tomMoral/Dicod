"""Helper functions for Convolutional Sparse Coding.

author : thomas.moreau@inria.fr
"""

import numpy as np
from scipy.signal import fftconvolve


def compute_norm_atoms(D):
    """Compute the norm of the atoms

    Parameters
    ----------
    D : ndarray, shape (n_atoms, n_channels, height_atom, width_atom)
        Current dictionary for the sparse coding
    """
    # Average over the channels and sum over the size of the atom
    norm_atoms = np.sum(D * D, axis=(1, 2, 3), keepdims=True)
    norm_atoms += (norm_atoms == 0)
    return norm_atoms[:, 0]


def compute_DtD(D):
    """Compute the transpose convolution between the atoms

    Parameters
    ----------
    D : ndarray, shape (n_atoms, n_channels, height_atom, width_atom)
        Current dictionary for the sparse coding
    """
    # Average over the channels
    DtD = np.sum([[[fftconvolve(di_p, dj_p, mode='full')
                    for di_p, dj_p in zip(di, dj)]
                   for dj in D]
                  for di in D[:, :, ::-1, ::-1]], axis=2)
    return DtD


def soft_thresholding(x, mu, positive=False):
    """Soft-thresholding operator

    Parameters
    ----------
    x : ndarray
        Variable on which the soft-thresholding is applied.
    mu : float
        Threshold of the operator
    positive : boolean
        If set to True, apply the soft-thresholding with positivity constraint.
    """
    if positive:
        return np.maximum(x - mu, 0)

    return np.sign(x) * np.maximum(abs(x) - mu, 0)


def reconstruct(z_hat, D):
    X_hat = np.sum([[fftconvolve(z_k, d_kp) for d_kp in d_k]
                    for z_k, d_k in zip(z_hat, D)], axis=0)
    return X_hat


def cost(X, z_hat, D, lmbd):
    res = X - reconstruct(z_hat, D)
    return 0.5 * np.sum(res ** 2) + lmbd * abs(z_hat).sum()
