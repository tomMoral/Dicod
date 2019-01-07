import os
import pytest
import numpy as np


from dicod_python.dicod import dicod
from dicod_python.utils import check_random_state
from dicod_python.coordinate_descent import _init_beta
from dicod_python.utils.csc import compute_ztz, compute_ztX

VERBOSE = 10
TEST_HOSTFILE = os.path.join(os.path.dirname(__file__), 'hostfile')


@pytest.mark.parametrize('n_jobs', [2, 4, 6])
@pytest.mark.parametrize('signal_shape, atom_shape', [((800,), (50,)),
                                                      ((100, 100), (10, 8))])
def test_stopping_criterion(n_jobs, signal_shape, atom_shape):
    tol = 1
    reg = 1
    n_atoms = 10
    n_channels = 3

    rng = check_random_state(42)

    X = rng.randn(n_channels, *signal_shape)
    D = rng.randn(n_atoms, n_channels, *atom_shape)
    sum_axis = tuple(range(1, D.ndim))
    D /= np.sqrt(np.sum(D * D, axis=sum_axis, keepdims=True))

    z_hat, *_ = dicod(X, D, reg, tol=tol, n_jobs=n_jobs,
                      hostfile=TEST_HOSTFILE, verbose=VERBOSE)

    beta, dz_opt = _init_beta(X, D, reg, z_i=z_hat)
    assert abs(dz_opt).max() < tol


@pytest.mark.parametrize('valid_shape, atom_shape', [((500,), (30,)),
                                                     ((72, 60), (10, 8))])
def test_ztz(valid_shape, atom_shape):
    tol = .5
    reg = 1
    n_atoms = 7
    n_channels = 5
    random_state = None

    sig_shape = tuple([
        (size_valid_ax + size_atom_ax - 1)
        for size_atom_ax, size_valid_ax in zip(atom_shape, valid_shape)])

    rng = check_random_state(random_state)

    X = rng.randn(n_channels, *sig_shape)
    D = rng.randn(n_atoms, n_channels, *atom_shape)
    D /= np.sqrt(np.sum(D * D, axis=(1, 2), keepdims=True))

    z_hat, ztz, ztX = dicod(X, D, reg, tol=tol, n_jobs=4, return_ztz=True,
                            hostfile=TEST_HOSTFILE, verbose=VERBOSE)

    ztz_full = compute_ztz(z_hat, atom_shape)
    assert np.allclose(ztz_full, ztz)

    ztX_full = compute_ztX(z_hat, X)
    assert np.allclose(ztX_full, ztX)
