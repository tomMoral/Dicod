#!/usr/bin/env python
import os
import sys
import logging
import numpy as np
from time import time
from mpi4py import MPI

from dicod_python.utils.mpi import broadcast_array
from dicod_python.utils import debug_flags as flags
from dicod_python.utils import constants as cst
from dicod_python.utils.segmentation import Segmentation


log = logging.getLogger('dicod')

# debug flags

interactive_exec = "xterm"
interactive_args = ["-fa", "Monospace", "-fs", "12", "-e", "ipython", "-i"]


def dicod_2d(X_i, D, lmbd, n_seg='auto', tol=1e-5, strategy='greedy',
             n_jobs=1, w_world='auto', max_iter=100000, timeout=None,
             z_positive=False, use_soft_lock=True, timing=False,
             hostfile=None, random_state=None, verbose=0, debug=False):
    """DICOD for 2D convolutional sparse coding.

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
    n_jobs : int
        Number of workers used to compute the convolutional sparse coding
        solution.
    max_iter : int
        Maximal number of iteration run by this algorithm.
    timeout : int
        Timeout for the algorithm in seconds
    z_positive : boolean
        If set to true, the activations are constrained to be positive.
    use_soft_lock : boolean
        If set to true, use the soft-lock in LGCD.
    timing : boolean
        If set to True, log the cost and timing information.
    hostfile : str
        File containing the cluster information. See MPI documentation to have
        the format of this file.
    random_state : None or int or RandomState
        current random state to seed the random number generator.
    verbose : int
        Verbosity level of the algorithm.

    Return
    ------
    z_hat : ndarray, shape (n_atoms, height_valid, width_valid)
        Activation associated to X_i for the given dictionary D
    """
    params = dict(
        strategy=strategy, tol=tol, max_iter=max_iter, timeout=timeout,
        n_seg=n_seg, z_positive=z_positive, verbose=verbose, timing=timing,
        debug=debug, random_state=random_state, lmbd=lmbd,
        use_soft_lock=use_soft_lock
    )
    n_channels, height, width = X_i.shape
    n_atoms, n_channels, height_atom, width_atom = D.shape
    height_valid = height - height_atom + 1
    width_valid = width - width_atom + 1
    params['valid_shape'] = valid_shape = (height_valid, width_valid)

    if w_world == 'auto':
        params["workers_topology"] = _find_grid_size(n_jobs, width, height)
    else:
        assert n_jobs % w_world == 0
        params["workers_topology"] = w_world, n_jobs // w_world

    # compute a segmentation for the image,
    overlap = (height_atom - 1, width_atom - 1)
    workers_segments = Segmentation(n_seg=params['workers_topology'],
                                    signal_shape=valid_shape,
                                    overlap=overlap)

    comm = _spawn_workers(n_jobs, hostfile)
    _send_task(comm, X_i, D, lmbd, workers_segments, params)
    comm.Barrier()
    z_hat = _recv_result(comm, n_atoms, valid_shape, workers_segments,
                         verbose=verbose)
    comm.Barrier()
    return z_hat


def _find_grid_size(n_jobs, width, height):
    w_world, h_world = 1, n_jobs
    w_ratio = width * n_jobs / height
    for i in range(2, n_jobs + 1):
        if n_jobs % i != 0:
            continue
        j = n_jobs // i
        ratio = width * j / (height * i)
        if abs(ratio - 1) < abs(w_ratio - 1):
            w_ratio = ratio
            w_world, h_world = i, j
    return w_world, h_world


def _spawn_workers(n_jobs, hostfile):
    info = MPI.Info.Create()
    # info.Set("map_bynode", '1')
    if hostfile and os.path.exists(hostfile):
        info.Set("hostfile", hostfile)
    script_name = os.path.join(os.path.dirname(__file__),
                               "_dicod_2d_worker.py")
    if flags.INTERACTIVE_PROCESSES:
        comm = MPI.COMM_SELF.Spawn(
            interactive_exec, args=interactive_args + [script_name],
            maxprocs=n_jobs, info=info)

    else:
        comm = MPI.COMM_SELF.Spawn(sys.executable, args=[script_name],
                                   maxprocs=n_jobs, info=info)
    return comm


def _send_task(comm, X, D, lmbd, workers_segments, params):
    t_start = time()
    n_atoms, n_channels, *atom_shape = D.shape
    height_atom, width_atom = atom_shape

    comm.bcast(params, root=MPI.ROOT)
    broadcast_array(comm, D)

    X = np.array(X, dtype='d')
    if params['debug']:
        X_alpha = np.zeros(X.shape, 'd')

    for i_seg in range(workers_segments.effective_n_seg):
        seg_bounds = workers_segments.get_seg_bounds(i_seg)
        X_worker_slice = (Ellipsis,) + tuple([
            slice(start, end + size_atom_ax - 1)
            for (start, end), size_atom_ax in zip(seg_bounds, atom_shape)
        ])

        comm.Send([X[X_worker_slice].ravel(), MPI.DOUBLE],
                  dest=i_seg, tag=cst.TAG_ROOT + i_seg)
        if params['debug']:
            X_worker = np.empty(X_alpha[X_worker_slice].shape, 'd')
            comm.Recv([X_worker.ravel(), MPI.DOUBLE],
                      source=i_seg, tag=cst.TAG_ROOT + i_seg)
            X_alpha[X_worker_slice] += X_worker

    if params['debug']:
        import matplotlib.pyplot as plt
        plt.imshow(np.clip(X_alpha.swapaxes(0, 2), 0, 1))
        plt.show()
        assert (np.sum(X_alpha[0, 0] == 0.5) ==
                3 * (width_atom - 1) * (workers_segments.n_seg_per_axis[0] - 1)
                )

    comm.Barrier()

    t_init = time() - t_start
    if params['verbose'] > 0:
        print('End initialisation - {:.4}s'.format(t_init))
    return


def _recv_result(comm, n_atoms, shape_valid, workers_segments, verbose=0):

    t_start = time()

    z_hat = np.empty((n_atoms,) + shape_valid, dtype='d')
    for i_seg in range(workers_segments.effective_n_seg):
        worker_shape = workers_segments.get_seg_shape(
            i_seg, only_inner=True)
        worker_slice = workers_segments.get_seg_slice(
            i_seg, only_inner=True)
        z_worker = np.empty((n_atoms,) + worker_shape, 'd')
        comm.Recv([z_worker.ravel(), MPI.DOUBLE], source=i_seg,
                  tag=cst.TAG_ROOT + i_seg)
        z_hat[worker_slice] = z_worker

    stats = comm.gather(None, root=MPI.ROOT)
    iterations = np.sum(stats, axis=0)[0]
    runtime = np.max(stats, axis=0)[1]
    print("[DICOD-{}] converged in {}s with {} iteration."
          .format(workers_segments.effective_n_seg, runtime, iterations))

    t_reduce = time() - t_start
    if verbose > 0:
        print('[DICOD-{}:DEBUG] End finalization - {:.4}s'
              .format(workers_segments.effective_n_seg, t_reduce))
    return z_hat
