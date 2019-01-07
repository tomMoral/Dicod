#
# Authors : Tommoral <thomas.moreau@inria.fr>
#
import time
import numpy as np
from scipy.signal import fftconvolve


from .utils import check_random_state
from .utils import debug_flags as flags
from .utils.segmentation import Segmentation
from .utils.csc import soft_thresholding, reconstruct
from .utils.csc import compute_DtD, compute_norm_atoms


def coordinate_descent(X_i, D, lmbd, n_seg='auto', tol=1e-5,
                       strategy='greedy', max_iter=100000,
                       z_positive=False, timing=False,
                       random_state=None, verbose=0):
    """Coordinate Descent Algorithm for 2D convolutional sparse coding.

    Parameters
    ----------
    X_i : ndarray, shape (n_channels, *sig_shape)
        Image to encode on the dictionary D
    z_i : ndarray, shape (n_atoms, *valid_shape)
        Warm start value for z_hat
    D : ndarray, shape (n_atoms, n_channels, *atom_shape)
        Current dictionary for the sparse coding
    lmbd : float
        Regularization parameter
    n_seg : int or { 'auto' }
        Number of segments to use for each dimension. If set to 'auto' use
        segments of twice the size of the dictionary.
    tol : float
        Tolerance for the minimal update size in this algorithm.
    strategy : str in { 'greedy' | 'random' }
        Coordinate selection scheme for the coordinate descent. If set to
        'greedy', the coordinate with the largest value for dz_opt is selected.
        If set to 'random, the coordinate is chosen uniformly on the segment.
    max_iter : int
        Maximal number of iteration run by this algorithm.
    z_positive : boolean
        If set to true, the activations are constrained to be positive.
    timing : boolean
        If set to True, log the cost and timing information.
    random_state : None or int or RandomState
        current random state to seed the random number generator.
    verbose : int
        Verbosity level of the algorithm.

    Return
    ------
    z_hat : ndarray, shape (n_atoms, *valid_shape)
        Activation associated to X_i for the given dictionary D
    """
    n_channels, *sig_shape = X_i.shape
    n_atoms, n_channels, *atom_shape = D.shape
    valid_shape = tuple([
        size_ax - size_atom_ax + 1
        for size_ax, size_atom_ax in zip(sig_shape, atom_shape)
    ])

    # compute sizes for the segments for LGCD
    if n_seg == 'auto':
        n_seg = []
        for axis_size, atom_size in zip(valid_shape, atom_shape):
            n_seg.append(max(axis_size // (2 * atom_size - 1), 1))
    segments = Segmentation(n_seg, signal_shape=valid_shape)

    # Pre-compute some quantities
    constants = {}
    constants['norm_atoms'] = compute_norm_atoms(D)
    constants['DtD'] = compute_DtD(D)

    # Initialization of the algorithm variables
    i_seg = -1
    accumulator = 0
    z_hat = np.zeros((n_atoms,) + valid_shape)
    n_coordinates = z_hat.size

    t_start = time.time()
    beta, dz_opt = _init_beta(X_i, D, lmbd, z_i=None, constants=constants,
                              z_positive=z_positive)
    for ii in range(max_iter):
        if ii % 1000 == 0 and verbose > 0:
            print("\rCD {:7.2%}".format(ii / max_iter), end='', flush=True)

        i_seg = segments.increment_seg(i_seg)
        if segments.is_active_segment(i_seg):
            k0, pt0, dz = _select_coordinate(dz_opt, segments, i_seg,
                                             strategy=strategy,
                                             random_state=random_state)
        else:
            k0, pt0, dz = None, None, 0

        accumulator = max(abs(dz), accumulator)

        # Update the selected coordinate and beta, only if the update is
        # greater than the convergence tolerance.
        if abs(dz) > tol:

            # update beta
            beta, dz_opt = coordinate_update(k0, pt0, dz, beta, dz_opt, z_hat,
                                             D, lmbd, constants, z_positive)
            touched_segs = segments.get_touched_segments(
                pt=pt0, radius=atom_shape)
            n_changed_status = segments.set_active_segments(touched_segs)

            if flags.CHECK_ACTIVE_SEGMENTS and n_changed_status:
                segments.test_active_segment(dz_opt, tol)

        elif strategy == "greedy":
            segments.set_inactive_segments(i_seg)

        if timing:
            # TODO: logging stuff
            pass

        # check stopping criterion
        if _check_convergence(segments, tol, ii, dz_opt, n_coordinates,
                              strategy, accumulator=accumulator):
            assert np.all(abs(dz_opt) <= tol)
            if verbose > 0:
                print("[LGCD:INFO] converged after {} iterations"
                      .format(ii + 1))

            break

    if verbose > 0:
        print("\r[LGCD:INFO] done in {:.3}s".format(time.time() - t_start))
    return z_hat


def _init_beta(X_i, D, lmbd, z_i=None, constants={}, z_positive=False):
    """Init beta with the gradient in the current point 0

    Parameters
    ----------
    X_i : ndarray, shape (n_channels, *sig_shape)
        Image to encode on the dictionary D
    z_i : ndarray, shape (n_atoms, *valid_shape)
        Warm start value for z_hat
    D : ndarray, shape (n_atoms, n_channels, *atom_shape)
        Current dictionary for the sparse coding
    lmbd : float
        Regularization parameter
    constants : dictionary, optional
        Pre-computed constants for the computations
    z_positive : boolean
        If set to true, the activations are constrained to be positive.
    """
    if 'norm_atoms' in constants:
        norm_atoms = constants['norm_atoms']
    else:
        norm_atoms = compute_norm_atoms(D)

    if z_i is not None and abs(z_i).sum() > 0:
        residual = reconstruct(z_i, D) - X_i
    else:
        residual = -X_i

    flip_axis = tuple(range(2, D.ndim))
    beta = np.sum(
        [[fftconvolve(dkp, res_p, mode='valid')
          for dkp, res_p in zip(dk, residual)]
         for dk in np.flip(D, flip_axis)], axis=1)

    if z_i is not None:
        assert z_i.shape == beta.shape
        for k, *pt in zip(*z_i.nonzero()):
            pt = tuple(pt)
            beta[k][pt] -= z_i[k][pt] * norm_atoms[k]

    dz_opt = soft_thresholding(-beta, lmbd, positive=z_positive) / norm_atoms

    if z_i is not None:
        dz_opt -= z_i

    return beta, dz_opt


def _select_coordinate(dz_opt, segments, i_seg, strategy='greedy',
                       random_state=None):
    """Pick a coordinate to update

    Parameters
    ----------
    dz_opt : ndarray, shape (n_atoms, *valid_shape)
        Difference between the current value and the optimal value for each
        coordinate.
    segments : dicod.utils.Segmentation
        Segmentation info for LGCD
    strategy : str in { 'greedy' | 'random' }
        Coordinate selection scheme for the coordinate descent. If set to
        'greedy', the coordinate with the largest value for dz_opt is selected.
        If set to 'random, the coordinate is chosen uniformly on the segment.
    random_state : None or int or RandomState
        current random state to seed the random number generator.
    """
    if strategy == 'random':
        rng = check_random_state(random_state)
        n_atoms, *valid_shape = dz_opt.shape
        k0 = rng.randint(n_atoms)
        pt0 = ()
        for size_valid_ax in valid_shape:
            v0 = rng.randint(size_valid_ax)
            pt0 = pt0 + (v0,)
        dz = dz_opt[k0][pt0]

    elif strategy == 'greedy':
        seg_slice = segments.get_seg_slice(i_seg)
        dz_opt_seg = dz_opt[seg_slice]
        i0 = abs(dz_opt_seg).argmax()
        k0, *pt0 = np.unravel_index(i0, dz_opt_seg.shape)
        pt0 = segments.get_global_coordinate(i_seg, pt0)
        dz = dz_opt[k0][pt0]
    else:
        raise ValueError("'The coordinate selection strategy should be in "
                         "{'greedy' | 'random' | 'cyclic'}. Got '{}'."
                         .format(strategy))
    return k0, pt0, dz


def coordinate_update(k0, pt0, dz, beta, dz_opt, z_hat, D, lmbd, constants,
                      z_positive, coordinate_exist=True):
    """Update the optimal value for the coordinate updates.

    Parameters
    ----------
    k0, pt0 : int, (int, int)
        Indices of the coordinate updated.
    dz : float
        Value of the update.
    beta, dz_opt : ndarray, shape (n_atoms, *valid_shape)
        Auxillary variables holding the optimal value for the coordinate update
    z_hat : ndarray, shape (n_atoms, *valid_shape)
        Value of the coordinate.
    D : ndarray, shape (n_atoms, n_channels, *atom_shape)
        Current dictionary for the sparse coding
    lmbd : float
        Regularization parameter
    constants : dictionary, optional
        Pre-computed constants for the computations
    z_positive : boolean
        If set to true, the activations are constrained to be positive.
    coordinate_exist : boolean
        If set to true, the coordinate is located in the updated part of beta.
        This option is only useful for DICOD.

    Return
    ------
    beta, dz_opt : ndarray, shape (n_atoms, *valid_shape)
        Auxillary variables holding the optimal value for the coordinate update
    """
    n_atoms, *valid_shape = beta.shape
    n_atoms, n_channels, *atom_shape = D.shape

    if 'DtD' in constants:
        DtD = constants['DtD']
    else:
        DtD = compute_DtD(D)
    if 'norm_atoms' in constants:
        norm_atoms = constants['norm_atoms']
    else:
        norm_atoms = compute_norm_atoms(D)

    # define the bounds for the beta update
    update_slice, DtD_slice = (Ellipsis,), (Ellipsis, k0)
    for v, size_atom_ax, size_valid_ax in zip(pt0, atom_shape, valid_shape):
        start_up_ax = max(0, v - size_atom_ax + 1)
        end_up_ax = min(size_valid_ax, v + size_atom_ax)
        update_slice = update_slice + (slice(start_up_ax, end_up_ax),)
        start_DtD_ax = max(0, size_atom_ax - 1 - v)
        end_DtD_ax = start_DtD_ax + (end_up_ax - start_up_ax)
        DtD_slice = DtD_slice + (slice(start_DtD_ax, end_DtD_ax),)

    # update beta
    if coordinate_exist:
        z_hat[k0][pt0] += dz
        beta_i0 = beta[k0][pt0]
    beta[update_slice] += DtD[DtD_slice] * dz

    # update dz_opt
    tmp = soft_thresholding(-beta[update_slice], lmbd,
                            positive=z_positive) / norm_atoms
    dz_opt[update_slice] = tmp - z_hat[update_slice]

    # If the coordinate exists, put it back to 0 update
    if coordinate_exist:
        beta[k0][pt0] = beta_i0
        dz_opt[k0][pt0] = 0

    return beta, dz_opt


def _check_convergence(segments, tol, iteration, dz_opt, n_coordinates,
                       strategy, accumulator=0):
    """Check convergence for the coordinate descent algorithm

    Parameters
    ----------
    segments : Segmentation
        Number of active segment at this iteration.
    tol : float
        Tolerance for the minimal update size in this algorithm.
    iteration : int
        Current iteration number
    dz_opt : ndarray, shape (n_atoms, *valid_shape)
        Difference between the current value and the optimal value for each
        coordinate.
    n_coordinates : int
        Number of coordinate in the considered problem.
    strategy : str in { 'greedy' | 'random' }
        Coordinate selection scheme for the coordinate descent. If set to
        'greedy', the coordinate with the largest value for dz_opt is selected.
        If set to 'random, the coordinate is chosen uniformly on the segment.
    accumulator : float, (default: 0)
        In the case of strategy 'random', accumulator should keep track of an
        approximation of max(abs(dz_opt)). The full convergence criterion will
        only be checked if accumulator <= tol.
    """
    # check stopping criterion
    if strategy == 'greedy':
        if not segments.exist_active_segment():
            return True
    else:
        # only check at the last coordinate
        if (iteration + 1) % n_coordinates == 0:
            accumulator *= 0
            if accumulator <= tol:
                return np.all(abs(dz_opt) <= tol)

    return False
