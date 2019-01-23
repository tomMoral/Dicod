# Authors: Thomas Moreau <thomas.moreau@inria.fr>

import numpy as np

from .optim import fista

from .loss_and_gradient import compute_objective
from .loss_and_gradient import gradient_d


def prox_d(D, step_size=0, return_norm=False):
    sum_axis = tuple(range(1, D.ndim))
    if D.ndim == 3:
        norm_d = np.maximum(1, np.linalg.norm(D, axis=sum_axis, keepdims=True))
    else:
        norm_d = np.sqrt(np.sum(D * D, axis=sum_axis, keepdims=True))
    D /= norm_d

    if return_norm:
        squeeze_axis = tuple(range(1, D.ndim))
        return D, norm_d.squeeze(axis=squeeze_axis)
    else:
        return D


def update_d(X, z, D_hat0, constants=None, step_size=None, max_iter=300,
             eps=None, momentum=False, verbose=0):
    """Learn d's in time domain.

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, *sig_shape)
        The data for sparse coding
    z : array, shape (n_trials, n_atoms, *valid_shape)
        Can also be a list of n_trials LIL-sparse matrix of shape
            (n_atoms, n_times - n_times_atom + 1)
        The code for which to learn the atoms
    D_hat0 : array, shape (n_atoms, n_channels, *atom_support)
        The initial atoms.
    constants : dict or None
        Dictionary of constants to accelerate the computation of the gradients.
        It should only be given for loss='l2' and should contain ztz and ztX.
    momentum : bool
        If True, use an accelerated version of the proximal gradient descent.
    verbose : int
        Verbosity level.

    Returns
    -------
    D_hat : array, shape (n_atoms, n_channels, n_times_atom)
        The atoms to learn from the data.
    """
    n_trials, n_channels, *sig_shape = X.shape
    n_atoms, n_channels, *atom_support = D_hat0.shape

    def objective(D, full=False):
        return compute_objective(D=D, constants=constants)

    def grad(D):
        grad = gradient_d(D=D, X=X, z=z, constants=constants)
        return grad

    def prox(D, step_size=0):
        D = prox_d(D)
        return D

    adaptive_step_size = True

    D_hat, pobj, step_size = fista(
        objective, grad, prox, x0=D_hat0, max_iter=max_iter,
        step_size=step_size, adaptive_step_size=adaptive_step_size,
        eps=eps, momentum=momentum, verbose=verbose, scipy_line_search=True,
        name="Update D")

    return D_hat, step_size
