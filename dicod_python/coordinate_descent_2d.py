import numpy as np
from scipy.signal import fftconvolve


from .utils import soft_thresholding, reconstruct
from .utils import compute_DtD, compute_norm_atoms
from .utils import check_random_state, NEIGHBOR_POS


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
    n_coordinates = n_atoms * height_valid * width_valid

    # compute sizes for the segments for LGCD
    seg_shape, grid_seg, n_seg = _get_seg_info(
        n_seg, height_valid, width_valid, atom_shape)

    # Pre-compute some quantities
    constants = {}
    constants['norm_atoms'] = compute_norm_atoms(D)
    constants['DtD'] = compute_DtD(D)

    # Initialization of the algorithm variables
    i_seg = 0
    accumulator = n_seg
    k0, h0, w0 = 0, -1, -1
    active_segments = np.array([True] * n_seg)
    seg_bounds = [[0, seg_shape[0]], [0, seg_shape[1]]]
    z_hat = np.zeros((n_atoms, height_valid, width_valid))

    beta, dz_opt = _init_beta(X_i, z_hat, D, lmbd, constants,
                              z_positive)
    for ii in range(max_iter):
        if ii % 100 == 0 and verbose > 0:
            print("\rCD {:7.2%}".format(ii / max_iter), end='', flush=True)
        k0, h0, w0, dz = _select_coordinate(dz_opt, seg_bounds,
                                            active_segments[i_seg],
                                            strategy=strategy,
                                            random_state=random_state)

        if strategy == 'random':
            # accumulate on all coordinates from the stopping criterion
            if ii % n_coordinates == 0:
                accumulator = 0
            accumulator = max(accumulator, abs(dz))

        # Update the selected coordinate and beta, only if the update is
        # greater than the convergence tolerance.
        if abs(dz) > tol:
            # update the selected coordinate
            z_hat[k0, h0, w0] += dz

            # update beta
            beta, dz_opt = _update_beta(dz, k0, h0, w0, beta, dz_opt, z_hat, D,
                                        lmbd, constants, z_positive)
            _update_active_segment(h0, w0, i_seg, active_segments, accumulator,
                                   seg_bounds, grid_seg, atom_shape)

        elif active_segments[i_seg] and strategy == "greedy":
            accumulator -= 1
            active_segments[i_seg] = False

        if timing:
            # TODO: logging stuff
            pass

        # check stopping criterion
        if _check_convergence(accumulator, tol, ii, n_coordinates, strategy):
            if verbose > 0:
                print("[Coordinate descent converged after {} iterations"
                      .format(ii + 1))
            break

        i_seg, seg_bounds = _next_seg(i_seg, seg_bounds, grid_seg, seg_shape)

    if verbose > 0:
        print("\r CD done   ")
    return z_hat


def _get_seg_info(n_seg, height_valid, width_valid, atom_shape):
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

    height_n_seg = height_valid // height_seg + (height_valid % height_seg != 0)
    width_n_seg = width_valid // width_seg + (width_valid % width_seg != 0)

    grid_seg = (height_n_seg, width_n_seg)
    seg_shape = (height_seg, width_seg)

    return seg_shape, grid_seg, width_n_seg * height_n_seg


def _init_beta(X_i, z_i, D, lmbd, constants, z_positive):
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

    nnz = z_i.nonzero()
    if len(nnz[0]) > 0:
        residual = reconstruct(z_i, D) - X_i
    else:
        residual = -X_i

    beta = np.sum(
        [[fftconvolve(dkp, res_p, mode='valid')
          for dkp, res_p in zip(dk, residual)]
         for dk in D[:, :, ::-1, ::-1]], axis=1)

    for k, h, w in zip(*nnz):
        beta[k, h, w] -= z_i[k, h, w] * norm_atoms[k]

    dz_opt = soft_thresholding(-beta, lmbd, positive=z_positive) / norm_atoms \
        - z_i

    return beta, dz_opt


def _select_coordinate(dz_opt, seg_bounds, active_seg, strategy='greedy',
                       random_state=None):
    """Pick a coordinate to update

    Parameters
    ----------
    dz_opt : ndarray, shape (n_atoms, height_valid, width_valid)
        Difference between the current value and the optimal value for each
        coordinate.
    seg_bounds : ((int, int), (int, int))
        Boundaries of the current segment
    active_seg : boolean
        Encode the fact that the segment has a chance to contain a value over
        the tolerance of the algorithm or not.
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

    elif strategy == 'greedy':
        start_height_seg, end_height_seg = seg_bounds[0]
        start_width_seg, end_width_seg = seg_bounds[1]
        if active_seg:
            dz_opt_seg = dz_opt[:, start_height_seg:end_height_seg,
                                start_width_seg:end_width_seg]
            i0 = abs(dz_opt_seg).argmax()
            k0, h0, w0 = np.unravel_index(i0, dz_opt_seg.shape)
            h0 += start_height_seg
            w0 += start_width_seg
            dz = dz_opt[k0, h0, w0]
        else:
            k0, h0, w0, dz = None, None, None, 0
    else:
        raise ValueError("'The coordinate selection strategy should be in "
                         "{'greedy' | 'random' | 'cyclic'}. Got '{}'."
                         .format(strategy))
    return k0, h0, w0, dz


def _update_beta(dz, k0, h0, w0, beta, dz_opt, z_hat, D, lmbd, constants,
                 z_positive, coordinate_exist=True):
    """Update the optimal value for the coordinate updates.

    Parameters
    ----------
    dz : float
        Value of the update.
    k0, h0, w0 : int
        Indices of the coordinate updated.
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


def _update_active_segment(h0, w0, i_seg, active_segments, accumulator,
                           seg_bounds, grid_shape, atom_shape):
    """Update the active segment if the current update interfered with it.

    Parameters
    ----------
    h0, w0 : int
        Indices of the current update.
    i_seg : int
        Indice of the segment being updated.
    active_segments : list of boolean
        array encoding whether a segment is active or not.
    accumulator : int
        Total number of active segments
    seg_bounds : ((int, int), (int, int))
        Boundaries of the current segment
    grid_shape : (int, int)
        Size of the grid of segments.
    atom_shape : (int, int)
        Shape of the atoms in the dictionary.
    """
    interfering_neighbors = get_interfering_neighbors(
        h0, w0, i_seg, seg_bounds, grid_shape, atom_shape)
    for i_neighbor in interfering_neighbors:
        if i_neighbor is not None and i_neighbor > 0:
            accumulator += not active_segments[i_neighbor]
            active_segments[i_neighbor] = True


def _check_convergence(accumulator, tol, iteration, n_coordinates, strategy):
    """Check convergence for the coordinate descent algorithm

    Parameters
    ----------
    accumulator : int
        Number of active segment at this iteration.
    tol : float
        Tolerance for the minimal update size in this algorithm.
    iteration : int
        Current iteration number
    n_coordinates : int
        Number of coordinate in the considered problem.
    strategy : str in { 'greedy' | 'random' }
        Coordinate selection scheme for the coordinate descent. If set to
        'greedy', the coordinate with the largest value for dz_opt is selected.
        If set to 'random, the coordinate is chosen uniformly on the segment.
    """
    # check stopping criterion
    if strategy == 'greedy':
        if accumulator == 0:
            return True
    else:
        # only check at the last coordinate
        if (iteration + 1) % n_coordinates == 0 and accumulator <= tol:
            return True


def _next_seg(i_seg, seg_bounds, grid_seg, seg_shape):
    """Increment the current segment and update the segment bounds

    Parameters
    ----------
    i_seg : int
        Current segment indice
    seg_bounds : ((int, int), (int, int))
        Boundaries of the current segment
    grid_seg : (int, int)
        number of segments on each dimension
    seg_shape : (int, int)
        Size of each segment

    Return
    ------
    i_seg : int, update segment indice
    seg_bounds : ((int, int), (int, int)), updated segment bounds
    """

    height_n_seg, width_n_seg = grid_seg
    height_seg, width_seg = seg_shape

    # increment to next segment
    i_seg += 1
    seg_bounds[1][0] += width_seg
    seg_bounds[1][1] += width_seg

    if i_seg % width_n_seg == 0:
        # Got to the begining of the next line
        seg_bounds[1] = [0, width_seg]
        seg_bounds[0][0] += height_seg
        seg_bounds[0][1] += height_seg

        if i_seg == width_n_seg * height_n_seg:
            # reset to first segment
            i_seg = 0
            seg_bounds = [[0, height_seg], [0, width_seg]]
    return i_seg, seg_bounds
