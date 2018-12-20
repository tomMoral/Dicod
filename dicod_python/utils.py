import numpy as np
from scipy.signal import fftconvolve


NEIGHBOR_POS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
                (1, -1), (1, 0), (1, 1)]


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def get_neighbors(i, grid_shape):
    """Return a list of existing neighbors for a given cell in a grid

    Parameters
    ----------
    i : int
        index of the cell in the grid
    grid_shape : 2-tuple
        Size of the considered grid.

    Return
    ------
    neighbors : list
        List with 8 elements. Return None if the neighbor does not exist and
        the ravel indice of the neighbor if it exists.
    """
    height, width = grid_shape
    assert 0 <= i < height * width
    h_cell, w_cell = i // height, i % height

    neighbors = [None] * 8
    for i, (dh, dw) in enumerate(NEIGHBOR_POS):
        h_neighbor = h_cell + dh
        w_neighbor = w_cell + dw
        has_neighbor = 0 <= h_neighbor < height
        has_neighbor &= 0 <= w_neighbor < width
        if has_neighbor:
            neighbors[i] = h_neighbor * width + w_neighbor

    return neighbors


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
