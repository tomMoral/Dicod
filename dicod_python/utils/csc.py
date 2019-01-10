"""Helper functions for Convolutional Sparse Coding.

Author : tommoral <thomas.moreau@inria.fr>
"""

import numpy as np
from scipy.signal import fftconvolve


def compute_norm_atoms(D):
    """Compute the norm of the atoms

    Parameters
    ----------
    D : ndarray, shape (n_atoms, n_channels, *atom_shape)
        Current dictionary for the sparse coding
    """
    # Average over the channels and sum over the size of the atom
    sum_axis = tuple(range(1, D.ndim))
    norm_atoms = np.sum(D * D, axis=sum_axis, keepdims=True)
    norm_atoms += (norm_atoms == 0)
    return norm_atoms[:, 0]


def compute_DtD(D):
    """Compute the transpose convolution between the atoms

    Parameters
    ----------
    D : ndarray, shape (n_atoms, n_channels, *atom_shape)
        Current dictionary for the sparse coding
    """
    # Average over the channels
    flip_axis = tuple(range(2, D.ndim))
    DtD = np.sum([[[fftconvolve(di_p, dj_p, mode='full')
                    for di_p, dj_p in zip(di, dj)]
                   for dj in D]
                  for di in np.flip(D, axis=flip_axis)], axis=2)
    return DtD


def compute_ztz(z, atom_shape, padding_shape=None):
    """
    ztz.shape = n_atoms, n_atoms, 2 * atom_shape - 1
    z.shape = n_atoms, n_times - n_times_atom + 1)
    """
    # TODO: benchmark the cross correlate function of numpy
    n_atoms, *_ = z.shape
    flip_axis = tuple(range(1, z.ndim))
    ztz_shape = (n_atoms, n_atoms) + tuple([
        2 * size_atom_ax - 1 for size_atom_ax in atom_shape
    ])

    if padding_shape is None:
        padding_shape = [(size_atom_ax - 1, size_atom_ax - 1)
                         for size_atom_ax in atom_shape]

    padding_shape = np.asarray([(0, 0)] + padding_shape, dtype='i')
    inner_slice = (Ellipsis,) + tuple([
        slice(size_atom_ax - 1, - size_atom_ax + 1)
        for size_atom_ax in atom_shape])

    z_pad = np.pad(z, padding_shape, mode='constant')
    z = z_pad[inner_slice]

    # compute the cross correlation between z and z_pad
    ztz = np.array([[fftconvolve(z_pad_k0, z_k, mode='valid')
                     for z_k in z]
                    for z_pad_k0 in np.flip(z_pad, axis=flip_axis)])
    assert ztz.shape == ztz_shape, (ztz.shape, ztz_shape)
    return ztz


def compute_ztX(z, X):
    """
    z.shape = n_atoms, n_times - n_times_atom + 1)
    X.shape = n_channels, n_times
    ztX.shape = n_atoms, n_channels, n_times_atom
    """
    n_atoms, *valid_shape = z.shape
    n_channels, *sig_shape = X.shape
    atom_shape = tuple(
        [size_ax - size_valid_ax + 1
         for size_ax, size_valid_ax in zip(sig_shape, valid_shape)])

    ztX = np.zeros((n_atoms, n_channels, *atom_shape))
    for k, *pt in zip(*z.nonzero()):
        pt = tuple(pt)
        X_slice = (Ellipsis,) + tuple([
            slice(v, v + size_atom_ax)
            for v, size_atom_ax in zip(pt, atom_shape)
        ])
        ztX[k] += z[k][pt] * X[X_slice]

    return ztX


def soft_thresholding(x, mu, positive=False):
    """Soft-thresholding point-wise operator

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
