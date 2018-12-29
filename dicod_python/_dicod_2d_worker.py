# Authors: Thomas Moreau <thomas.moreau@inria.fr>

import time
import numpy as np
from mpi4py import MPI

from dicod_python.utils_mpi import recv_broadcasted_array
from dicod_python.utils import compute_DtD, compute_norm_atoms
from dicod_python.utils import NEIGHBOR_POS

from dicod_python.coordinate_descent_2d import _update_beta
from dicod_python.coordinate_descent_2d import _select_coordinate
from dicod_python.coordinate_descent_2d import _check_convergence
from dicod_python.coordinate_descent_2d import _is_interfering_update
from dicod_python.coordinate_descent_2d import _init_beta, _get_seg_info
from dicod_python.coordinate_descent_2d import get_interfering_neighbors

ALGO_GS = 0
ALGO_RANDOM = 1


TAG_ROOT = 4242


class CoordinateSystems:
    """Utility to move from on coordinate system to another easily
    """

    def __init__(self, offset, bounds):
        self.offset = offset
        self.bounds = bounds

    def get_full_coordinate(self, pt):
        res = []
        for v, offset, (padding, _) in zip(pt, self.offset, self.bounds):
            res += [v - padding + offset]
        return res

    def get_local_coordinate(self, pt):
        res = []
        for v, offset, (padding, _) in zip(pt, self.offset, self.bounds):
            res += [v + padding - offset]
        return res

    def is_valid_local_coordinate(self, pt):
        is_valid = True
        for v, (v_start, v_end) in zip(pt, self.bounds):
            is_valid &= (v_start <= v < v_end)
        return is_valid


def dicod_worker(comm):
    # Receive the task and the parameters for the algorithm
    X_worker, D, worker_bounds, coordinate_system, params = _receive_task(comm)
    rank = comm.Get_rank()
    tol = params['tol']
    lmbd = params['lmbd']
    n_seg = params['n_seg']
    verbose = params['verbose']
    strategy = params['strategy']
    z_positive = params['z_positive']

    random_state = params['random_state']
    if isinstance(random_state, int):
        random_state += rank

    n_channels, height_worker, width_worker = X_worker.shape
    n_atoms, n_channels, height_atom, width_atom = D.shape
    atom_shape = (height_atom, width_atom)
    height_valid = height_worker - height_atom + 1
    width_valid = width_worker - width_atom + 1
    worker_shape = [v[0] for v in np.diff(worker_bounds, axis=1)]
    n_coordinates = n_atoms * height_valid * width_valid

    # compute sizes for the segments for LGCD
    seg_shape, grid_seg, n_seg = _get_seg_info(
        n_seg, *worker_shape, atom_shape)

    # Pre-compute some quantities
    constants = {}
    constants['norm_atoms'] = compute_norm_atoms(D)
    constants['DtD'] = compute_DtD(D)

    # Initialization of the algorithm variables
    i_seg = 0
    accumulator = n_seg
    k0, h0, w0 = 0, -1, -1
    active_segments = np.array([True] * n_seg)
    (h_start, _), (w_start, _) = worker_bounds
    seg_bounds = [[h_start, h_start + seg_shape[0]],
                  [w_start, w_start + seg_shape[1]]]
    z_hat = np.zeros((n_atoms, height_valid, width_valid))

    beta, dz_opt = _init_beta(X_worker, z_hat, D, lmbd, constants,
                              z_positive)

    t_start = time.time()
    for ii in range(params['max_iter']):
        if rank == 0 and ii % 1000 == 0 and verbose > 0:
            print("\rCD {:7.2%}".format(ii / params['max_iter']), end='',
                  flush=True)

        beta, dz_opt, active_segments, accumulator = _process_neighbors_update(
            comm, beta, dz_opt, z_hat, D, lmbd, constants, z_positive,
            active_segments, accumulator, coordinate_system)
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
        # greater than the convergence tolerance and is contained in the worker
        # If the update is not in the worker, this will effectively work has a
        # soft lock to prevent interferences.
        if abs(dz) > tol:
            if coordinate_system.is_valid_local_coordinate((h0, w0)):
                # update the selected coordinate
                z_hat[k0, h0, w0] += dz

                # update beta
                beta, dz_opt = _update_beta(dz, k0, h0, w0, beta, dz_opt,
                                            z_hat, D, lmbd, constants,
                                            z_positive)

                interfering_neighbors = get_interfering_neighbors(
                    h0, w0, i_seg, seg_bounds, grid_seg, atom_shape)
                interfering_workers = get_interfering_soft_lock(
                    h0, w0, rank, worker_bounds, params['grid_workers'],
                    atom_shape)

                update_msg = np.array([k0, h0, w0, dz], 'd')
                _update_active_segment(update_msg, interfering_neighbors,
                                       interfering_workers,
                                       active_segments, accumulator,
                                       coordinate_system)
            else:
                raise RuntimeError(rank, (k0, h0, w0), seg_bounds)

        elif active_segments[i_seg] and strategy == "greedy":
            accumulator -= 1
            active_segments[i_seg] = False

        if params['timing']:
            # TODO: logging stuff
            pass

        # check stopping criterion
        if _check_convergence(accumulator, tol, ii, n_coordinates,
                              strategy):
            if verbose > 1:
                print("[Worker-{:<3}] LGCD converged after {} iterations"
                      .format(rank, ii + 1))
            break

        i_seg, seg_bounds = _next_seg(i_seg, seg_bounds, grid_seg, seg_shape,
                                      worker_bounds)

    comm.Barrier()
    if rank == 0 and verbose > 0:
        print("\r[DICOD] LGCD finished in {}s".format(time.time() - t_start))

    _send_result(comm, z_hat, worker_bounds, atom_shape)
    return verbose


def _is_valid_coordinate_update(h0, w0, dz, tol, worker_bounds):
    (h_start, h_end), (w_start, w_end) = worker_bounds

    is_large = abs(dz) >= tol
    is_valid = True
    if is_large:
        is_valid = (h_start <= h0 < h_end)
        is_valid &= (w_start <= w0 < w_end)
    return is_large, is_valid


def get_interfering_soft_lock(h0, w0, i_seg, worker_bounds, grid_shape,
                              atom_shape):
    """Get the list of neighbor worker affected by the given update in their
    soft-lock area.

    Parameters
    ---------
    h0, w0 : int
        Position of the update
    i_seg : int
        Indice of the current segment, updated with this update.
    worker_bounds : ((int, int), (int, int))
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

    interfering_neighbors = [None] * len(NEIGHBOR_POS)
    h_interf = _is_interfering_update(h0, 2 * atom_shape[0], *worker_bounds[0])
    w_interf = _is_interfering_update(w0, 2 * atom_shape[1], *worker_bounds[1])

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


def _receive_task(comm):
    # Retrieve the rank of the worker
    rank = comm.Get_rank()
    n_jobs = comm.Get_size()

    # receive constants
    params = comm.bcast(None, root=0)
    D = recv_broadcasted_array(comm)

    strategy = params['strategy']
    height_valid, width_valid = params['valid_shape']
    n_atoms, n_channels, height_atom, width_atom = D.shape
    h_world, w_world = params['grid_workers']

    if rank == 0 and params['verbose'] > 0:
        print("DEBUG - Start dicod with {} workers and strategy '{}'"
              .format(n_jobs, strategy))

    # Infer some topological information
    h_rank = rank // w_world
    w_rank = rank % w_world

    # Compute the size of the signal for this worker and the worker_bounds
    height_worker = height_valid // h_world
    width_worker = width_valid // w_world
    height_offset = h_rank * height_worker
    width_offset = w_rank * width_worker
    h_start = max(0, h_rank * height_worker - height_atom + 1)
    h_end = min((h_rank + 1) * height_worker + 2 * height_atom - 2,
                height_valid + height_atom - 1)
    w_start = max(0, w_rank * width_worker - width_atom + 1)
    w_end = min((w_rank + 1) * width_worker + 2 * width_atom - 2,
                width_valid + width_atom - 1)
    if h_rank == h_world - 1:
        height_worker = height_valid - height_offset
    if w_rank == w_world - 1:
        width_worker = width_valid - width_offset
    worker_bounds = [
        [height_offset - h_start, height_offset - h_start + height_worker],
        [width_offset - w_start, width_offset - w_start + width_worker]
    ]
    worker_offset = [height_offset, width_offset]

    X_shape = (n_channels, h_end - h_start, w_end - w_start)
    X_worker = np.empty(X_shape, dtype='d')
    comm.Recv([X_worker.ravel(), MPI.DOUBLE], source=0, tag=TAG_ROOT + rank)

    if params['debug']:
        X_alpha = np.concatenate([X_worker,
                                  0.25 * np.ones((1,) + X_shape[1:])])

        comm.Send([X_alpha.ravel(), MPI.DOUBLE], dest=0,
                  tag=TAG_ROOT + rank)

    coordinate_system = CoordinateSystems(worker_offset, worker_bounds)
    comm.Barrier()
    return X_worker, D, worker_bounds, coordinate_system, params


def _update_active_segment(update, interfering_neighbors, interfering_workers,
                           active_segments, accumulator, coordinate_system):
    """Update the active segment if the current update interfered with it.

    Parameters
    ----------
    update : ndarray, shape (4,)
        Message to send to neighbor worker to update their info.
    interfering_neighbors : list
        Indices of the neighbors affected by the given update if they exist, or
        -1 if they do not exist. This returns None when the segment is not
    interfering_workers : list
        Indices of the neighbors affected by the given update if they exist, or
        -1 if they do not exist. This returns None when the segment is not
        touched by the given update.
    active_segments : list of boolean
        array encoding whether a segment is active or not.
    accumulator : int
        Total number of active segments
    """

    for i, (i_neighbor, i_worker) in enumerate(zip(interfering_neighbors,
                                                   interfering_workers)):
        if i_neighbor is not None and i_neighbor >= 0:
            accumulator += not active_segments[i_neighbor]
            active_segments[i_neighbor] = True
        if i_worker is not None and i_worker >= 0:
            # XXX: notify the neighbor
            k0, h0, w0, dz = update
            h0, w0 = coordinate_system.get_full_coordinate((h0, w0))
            msg = np.array([k0, h0, w0, dz], 'd')
            MPI.COMM_WORLD.Isend([msg, MPI.DOUBLE], dest=i_worker)


def _process_neighbors_update(comm, beta, dz_opt, z_hat, D, lmbd, constants,
                              z_positive, active_segments, accumulator,
                              coordinate_system):

    status = MPI.Status()
    msg = np.empty(4, 'd')
    while MPI.COMM_WORLD.Iprobe(status=status):
        src = status.source
        MPI.COMM_WORLD.Recv([msg, MPI.DOUBLE], source=src)
        k0, h0, w0, dz = msg

        k0, h0, w0 = int(k0), int(h0), int(w0)
        h0, w0 = coordinate_system.get_local_coordinate((h0, w0))
        height, width = beta.shape[1:]
        coordinate_exist = (0 <= h0 < height) and (0 <= w0 < width)
        beta, dz_opt = _update_beta(dz, k0, h0, w0, beta, dz_opt, z_hat, D,
                                    lmbd, constants, z_positive,
                                    coordinate_exist)
        for k in range(len(active_segments)):
            active_segments[k] = True
        accumulator = len(active_segments)
    return beta, dz_opt, active_segments, accumulator


def _next_seg(i_seg, seg_bounds, grid_seg, seg_shape, worker_bounds):
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
    (h_start, h_end), (w_start, w_end) = worker_bounds

    # increment to next segment
    i_seg += 1
    seg_bounds[1][0] += width_seg
    seg_bounds[1][1] = min(seg_bounds[1][1] + width_seg, w_end)

    if i_seg % width_n_seg == 0:
        # Got to the begining of the next line
        seg_bounds[1] = [w_start, w_start + width_seg]
        seg_bounds[0][0] += height_seg
        seg_bounds[0][1] = min(seg_bounds[0][1] + height_seg, h_end)

        if i_seg == width_n_seg * height_n_seg:
            # reset to first segment
            i_seg = 0
            seg_bounds = [[h_start, h_start + height_seg],
                          [w_start, w_start + width_seg]]
    return i_seg, seg_bounds


def _send_result(comm, z_hat, worker_bounds, atom_shape):
    rank = comm.Get_rank()
    height_atom, width_atom = atom_shape
    (h_start, h_end), (w_start, w_end) = worker_bounds

    z_worker = z_hat[:, h_start:h_end, w_start:w_end].ravel()
    comm.Send([z_worker, MPI.DOUBLE], dest=0,
              tag=TAG_ROOT + rank)
    comm.Barrier()


def shutdown(comm, verbose=0):
    if verbose > 5:
        print("[Worker-{:03}] clean shutdown".format(comm.Get_rank()))
    comm.Barrier()
    comm.Disconnect()


if __name__ == "__main__":
    comm = MPI.Comm.Get_parent()
    verbose = dicod_worker(comm)
    shutdown(comm, verbose)
