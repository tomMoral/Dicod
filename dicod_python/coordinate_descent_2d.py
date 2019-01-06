import numpy as np
from scipy.signal import fftconvolve


from .utils import debug_flags as flags
from .utils.segmentation import Segmentation
from .utils import check_random_state, NEIGHBOR_POS
from .utils.csc import soft_thresholding, reconstruct
from .utils.csc import compute_DtD, compute_norm_atoms


def coordinate_descent_2d(X_i, D, lmbd, n_seg='auto', tol=1e-5,
                          strategy='greedy', max_iter=100000,
                          z_positive=False, timing=False,
                          random_state=None, verbose=0):
    """Coordinate Descent Algorithm for 2D convolutional sparse coding.

    Parameters
    ----------
    X_i : ndarray, shape (n_channels, height, width)
        Image to encode on the dictionary D
    z_i : ndarray, shape (n_atoms, height_valid, width_valid)
        Warm start value for z_hat
    D : ndarray, shape (n_atoms, n_channels, height_atom, width_atom)
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
    z_hat : ndarray, shape (n_atoms, height_valid, width_valid)
        Activation associated to X_i for the given dictionary D
    """
    n_channels, height, width = X_i.shape
    n_atoms, n_channels, height_atom, width_atom = D.shape
    atom_shape = (height_atom, width_atom)

    height_valid = height - height_atom + 1
    width_valid = width - width_atom + 1
    valid_shape = (height_valid, width_valid)
    n_coordinates = n_atoms * height_valid * width_valid

    # compute sizes for the segments for LGCD
    if n_seg == 'auto':
        n_seg = []
        for axis_size, atom_size in zip(valid_shape, atom_shape):
            n_seg.append(max(axis_size // (2 * atom_size - 1), 1))
    segments = Segmentation(n_seg, signal_shape=valid_shape)

    # seg_shape, grid_seg, n_seg = get_seg_info(
    #     n_seg, height_valid, width_valid, atom_shape)

    # Pre-compute some quantities
    constants = {}
    constants['norm_atoms'] = compute_norm_atoms(D)
    constants['DtD'] = compute_DtD(D)

    # Initialization of the algorithm variables
    i_seg = -1
    accumulator = 0
    z_hat = np.zeros((n_atoms, height_valid, width_valid))

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
            # update the selected coordinate
            z_hat[(k0,) + pt0] += dz

            # update beta
            beta, dz_opt = _update_beta(k0, pt0, dz, beta, dz_opt, z_hat, D,
                                        lmbd, constants, z_positive)
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
                print("[Coordinate descent converged after {} iterations"
                      .format(ii + 1))

            break

    if verbose > 0:
        print("\r CD done   ")
    return z_hat


def get_seg_info(n_seg, height_valid, width_valid, atom_shape):
    """Compute the number of segment and their shapes for LGCD.

    Parameters
    ----------
    n_seg : int or { 'auto' }
        Number of segments to use for each dimension. If set to 'auto' use
        segments of twice the size of the dictionary.
    height_valid, width_valid : int
        Size of the considered signal.
    atom_shape : (int, int)
        Shape of the atoms in the dictionary.

    Return
    ------
    seg_shape : (int, int), shape of each segment.
    grid_seg : (int, int), number of segments on each dimension
    n_seg :  int, total number of segment used
    """
    height_atom, width_atom = atom_shape

    # compute sizes for the segments for LGCD
    if n_seg == 'auto':
        # use the default value for n_seg, ie 2x the size of D
        height_seg, width_seg = 2 * height_atom, 2 * width_atom
    else:
        height_seg = max(height_valid // n_seg, 1)
        width_seg = max(width_valid // n_seg, 1)

    height_n_seg = height_valid // height_seg
    width_n_seg = width_valid // width_seg

    grid_seg = (height_n_seg, width_n_seg)
    seg_shape = (height_seg, width_seg)

    return seg_shape, grid_seg, width_n_seg * height_n_seg


def _init_beta(X_i, D, lmbd, z_i=None, constants={}, z_positive=False):
    """Init beta with the gradient in the current point 0

    Parameters
    ----------
    X_i : ndarray, shape (n_channels, height, width)
        Image to encode on the dictionary D
    z_i : ndarray, shape (n_atoms, height_valid, width_valid)
        Warm start value for z_hat
    D : ndarray, shape (n_atoms, n_channels, height_atom, width_atom)
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

    beta = np.sum(
        [[fftconvolve(dkp, res_p, mode='valid')
          for dkp, res_p in zip(dk, residual)]
         for dk in D[:, :, ::-1, ::-1]], axis=1)

    if z_i is not None:
        for k, h, w in zip(*z_i.nonzero()):
            beta[k, h, w] -= z_i[k, h, w] * norm_atoms[k]

    dz_opt = soft_thresholding(-beta, lmbd, positive=z_positive) / norm_atoms

    if z_i is not None:
        dz_opt -= z_i

    return beta, dz_opt


def _select_coordinate(dz_opt, segments, i_seg, strategy='greedy',
                       random_state=None):
    """Pick a coordinate to update

    Parameters
    ----------
    dz_opt : ndarray, shape (n_atoms, height_valid, width_valid)
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
        n_atoms, height_valid, width_valid = dz_opt.shape
        k0 = rng.randint(n_atoms)
        h0 = rng.randint(height_valid)
        w0 = rng.randint(width_valid)
        dz = dz_opt[k0, h0, w0]
        pt0 = (h0, w0)

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


def _update_beta(k0, pt0, dz, beta, dz_opt, z_hat, D, lmbd, constants,
                 z_positive, coordinate_exist=True):
    """Update the optimal value for the coordinate updates.

    Parameters
    ----------
    k0, pt0 : int, (int, int)
        Indices of the coordinate updated.
    dz : float
        Value of the update.
    beta, dz_opt : ndarray, shape (n_atoms, height_valid, width_valid)
        Auxillary variables holding the optimal value for the coordinate update
    z_hat : ndarray, shape (n_atoms, height_valid, width_valid)
        Value of the coordinate.
    D : ndarray, shape (n_atoms, n_channels, height_atom, width_atom)
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
    beta, dz_opt : ndarray, shape (n_atoms, height_valid, width_valid)
        Auxillary variables holding the optimal value for the coordinate update
    """
    n_atoms, height_valid, width_valid = beta.shape
    n_atoms, n_channels, height_atom, width_atom = D.shape

    if 'DtD' in constants:
        DtD = constants['DtD']
    else:
        DtD = compute_DtD(D)
    if 'norm_atoms' in constants:
        norm_atoms = constants['norm_atoms']
    else:
        norm_atoms = compute_norm_atoms(D)

    # define the bounds for the beta update
    h0, w0 = pt0
    start_height_up = max(0, h0 - height_atom + 1)
    end_height_up = min(h0 + height_atom, height_valid)
    start_width_up = max(0, w0 - width_atom + 1)
    end_width_up = min(w0 + width_atom, width_valid)
    update_slice = (slice(None), slice(start_height_up, end_height_up),
                    slice(start_width_up, end_width_up))

    # update beta
    if coordinate_exist:
        beta_i0 = beta[k0, h0, w0]
    update_height = end_height_up - start_height_up
    offset_height = max(0, height_atom - h0 - 1)
    update_width = end_width_up - start_width_up
    offset_width = max(0, width_atom - w0 - 1)
    beta[update_slice] += (
        DtD[:, k0, offset_height:offset_height + update_height,
            offset_width:offset_width + update_width] * dz
    )

    # update dz_opt
    tmp = soft_thresholding(-beta[update_slice], lmbd,
                            positive=z_positive) / norm_atoms
    dz_opt[update_slice] = tmp - z_hat[update_slice]

    # If the coordinate exists, put it back to 0 update
    if coordinate_exist:
        beta[k0, h0, w0] = beta_i0
        dz_opt[k0, h0, w0] = 0

    return beta, dz_opt


def _is_interfering_update(t, t_atom, t_start_seg, t_end_seg):
    """Check if an update in t will interfer outside of a given segment

    Parameters
    ----------
    t : int
        indice of the update to check
    t_atom : int
        size of the atom used with the update
    t_start_seg : int
        indice where the considered segment begins
    t_end_seg : int
        indice where the considered segment ends
    """
    upstream_interference = (t - t_start_seg + 1) < t_atom
    downstream_interference = (t_end_seg - t) < t_atom
    return True, downstream_interference, upstream_interference


def get_interfering_neighbors(h0, w0, i_seg, seg_bounds, grid_shape,
                              atom_shape):
    """Get the list of neighboring segments affected by the given update.

    Parameters
    ---------
    h0, w0 : int
        Position of the update
    i_seg : int
        Indice of the current segment, updated with this update.
    seg_bounds : ((int, int), (int, int))
        Boundaries of the current segment
    grid_shape : (int, int)
        Size of the grid of segments.
    atom_shape : (int, int)
        Shape of the atoms in the dictionary.

    Return
    ------
    interfering_neighbors : list
        Indices of the neighbors affected by the given update if they exist, or
        -1 if they do not exist. This returns None when the segment is not
        touched by the given update.
    """
    grid_height, grid_width = grid_shape
    assert 0 <= i_seg < grid_height * grid_width
    h_cell, w_cell = i_seg // grid_width, i_seg % grid_width

    interfering_neighbors = [None] * 8
    h_interf = _is_interfering_update(h0, atom_shape[0], *seg_bounds[0])
    w_interf = _is_interfering_update(w0, atom_shape[1], *seg_bounds[1])

    for i, (dh, dw) in enumerate(NEIGHBOR_POS):
        h_neighbor = h_cell + dh
        w_neighbor = w_cell + dw
        has_neighbor = 0 <= h_neighbor < grid_height
        has_neighbor &= 0 <= w_neighbor < grid_width
        if h_interf[dh] and w_interf[dw]:
            if has_neighbor:
                interfering_neighbors[i] = h_neighbor * grid_width + w_neighbor
            else:
                interfering_neighbors[i] = -1

    return interfering_neighbors


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
    dz_opt : ndarray, shape (n_atoms, height_valid, width_valid)
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
            if flags.CHECK_ACTIVE_SEGMENTS:
                assert np.all(abs(dz_opt) <= tol)
            return True
    else:
        # only check at the last coordinate
        if (iteration + 1) % n_coordinates == 0:
            accumulator *= 0
            if accumulator <= tol:
                return np.all(abs(dz_opt) <= tol)

    return False


def get_seg_bounds(i_seg, grid_seg, seg_shape, worker_bounds=None):
    """Get the bound of a given segment

    Parameters
    ----------
    i_seg : int
        Current segment indice
    grid_seg : (int, int)
        number of segments on each dimension
    seg_shape : (int, int)
        Size of each segment
    worker_bounds : ((int, int), (int, int))
        Boundaries of the worker segment

    Return
    ------
    seg_bounds : ((int, int), (int, int)), segment bounds
    """

    height_n_seg, width_n_seg = grid_seg
    height_seg, width_seg = seg_shape

    if worker_bounds is None:
        h_start, h_end = 0, height_n_seg * height_seg
        w_start, w_end = 0, width_n_seg * width_seg
    else:
        (h_start, h_end), (w_start, w_end) = worker_bounds

    # Compute cartesian coordinate of the segment
    w_seg = i_seg % width_n_seg
    h_seg = i_seg // width_n_seg

    # Compute the bounds of the current segment
    seg_bounds = np.array(
        [[h_seg * height_seg, (h_seg+1) * height_seg],
         [w_seg * width_seg, (w_seg+1) * width_seg]])
    seg_bounds += [[h_start], [w_start]]

    # the last segment in each direction can be larger
    if (h_seg + 1) % height_n_seg == 0:
        seg_bounds[0][1] = h_end
    if (w_seg + 1) % width_n_seg == 0:
        seg_bounds[1][1] = w_end
    return seg_bounds
