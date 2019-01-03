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


log = logging.getLogger('dicod')

# debug flags

interactive_exec = "xterm"
interactive_args = ["-fa", "Monospace", "-fs", "12", "-e", "ipython", "-i"]


def dicod_2d(X_i, D, lmbd, n_seg='auto', tol=1e-5, strategy='greedy',
             n_jobs=1, w_world='auto', max_iter=100000, timeout=None,
             z_positive=False, timing=False, hostfile=None, random_state=None,
             verbose=0, debug=False):
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
        debug=debug, random_state=random_state
    )
    n_channels, height, width = X_i.shape
    n_atoms, n_channels, height_atom, width_atom = D.shape
    height_valid = height - height_atom + 1
    width_valid = width - width_atom + 1
    shape_valid = (height_valid, width_valid)

    if w_world == 'auto':
        worker_topology = _find_grid_size(n_jobs, width, height)
    else:
        assert n_jobs % w_world == 0
        worker_topology = w_world, n_jobs // w_world

    comm = _spawn_workers(n_jobs, hostfile)
    _send_task(comm, X_i, D, lmbd, worker_topology, params)
    comm.Barrier()
    z_hat = _recv_result(comm, n_atoms, shape_valid, worker_topology,
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


def _send_task(comm, X, D, lmbd, worker_topology, params):
    t_start = time()
    n_atoms, n_channels, height_atom, width_atom = D.shape
    n_channels, height, width = X.shape
    h_world, w_world = worker_topology

    # Compute shape for the valid image
    height_valid = height - height_atom + 1
    width_valid = width - width_atom + 1

    params.update(lmbd=lmbd, valid_shape=(height_valid, width_valid),
                  worker_topology=worker_topology)
    comm.bcast(params, root=MPI.ROOT)
    broadcast_array(comm, D)

    X = np.array(X, dtype='d')
    if params['debug']:
        X_alpha = np.zeros((4, height, width), 'd')
    height_worker = height_valid // h_world
    width_worker = width_valid // w_world
    for i in range(h_world):
        h_start = max(0, i * height_worker - height_atom + 1)
        h_end = min((i+1)*height_worker + 2 * height_atom - 2, height)
        for j in range(w_world):
            dest = i * w_world + j
            w_start = max(0, j * width_worker - width_atom + 1)
            w_end = min((j+1) * width_worker + 2 * width_atom - 2, width)
            X_worker_slice = (slice(None), slice(h_start, h_end),
                              slice(w_start, w_end))
            comm.Send([X[X_worker_slice].ravel(), MPI.DOUBLE],
                      dest, tag=cst.TAG_ROOT + dest)
            if params['debug']:
                X_worker = np.empty(X_alpha[X_worker_slice].shape, 'd')
                comm.Recv([X_worker.ravel(), MPI.DOUBLE],
                          source=dest, tag=cst.TAG_ROOT + dest)
                X_alpha[:, h_start:h_end, w_start:w_end] += X_worker
    if params['debug']:
        import matplotlib.pyplot as plt
        X_alpha[:3] = X_alpha[3:]
        assert np.sum(X_alpha[3, 0] == 0.5) == 3*(width_atom - 1)
        plt.imshow(np.clip(X_alpha.swapaxes(0, 2), 0, 1))
        plt.show()

    comm.Barrier()

    t_init = time() - t_start
    if params['verbose'] > 0:
        print('End initialisation - {:.4}s'.format(t_init))
    return


def _recv_result(comm, n_atoms, shape_valid, worker_topology, verbose=0):

    t_start = time()
    height_valid, width_valid = shape_valid
    h_world, w_world = worker_topology
    n_jobs = h_world * w_world

    z_hat = np.empty((n_atoms,) + shape_valid, dtype='d')
    height_worker = height_valid // h_world
    width_worker = width_valid // w_world
    for i in range(h_world):
        h_end = (i+1) * height_worker
        if i == h_world - 1:
            h_end = height_valid
        for j in range(w_world):
            w_end = (j+1) * width_worker
            if j == w_world - 1:
                w_end = width_valid
            worker_slice = (slice(None), slice(i * height_worker, h_end),
                            slice(j * width_worker, w_end))
            z_worker = np.empty(z_hat[worker_slice].shape, 'd')
            dest = i * w_world + j
            comm.Recv([z_worker.ravel(), MPI.DOUBLE], source=dest,
                      tag=cst.TAG_ROOT + dest)
            z_hat[worker_slice] = z_worker

    stats = comm.gather(None, root=MPI.ROOT)
    iterations = np.sum(stats, axis=0)[0]
    runtime = np.max(stats, axis=0)[1]
    print("[DICOD-{}] converged in {}s with {} iteration."
          .format(n_jobs, runtime, iterations))

    t_reduce = time() - t_start
    if verbose > 5:
        print('[DICOD-{}:DEBUG] End finalization - {:.4}s'
              .format(n_jobs, t_reduce))
    return z_hat
