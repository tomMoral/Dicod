#!/usr/bin/env python
import os
import sys
import logging
import numpy as np
from time import time, sleep
from mpi4py import MPI
from scipy.signal import fftconvolve

from dicod_python.utils_mpi import broadcast_array


log = logging.getLogger('dicod')

TAG_ROOT = 4242

ALGO_GS = 0
ALGO_RANDOM = 1


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
        grid_workers = _find_grid_size(n_jobs, width, height)
    else:
        assert n_jobs % w_world == 0
        grid_workers = w_world, n_jobs // w_world

    comm = _spawn_workers(n_jobs, hostfile)
    _send_task(comm, X_i, D, lmbd, grid_workers, params)
    comm.Barrier()
    z_hat = _recv_result(comm, n_atoms, shape_valid, grid_workers,
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
    comm = MPI.COMM_SELF.Spawn(sys.executable, args=[script_name],
                               maxprocs=n_jobs, info=info)
    return comm


def _send_task(comm, X, D, lmbd, grid_workers, params):
    t_start = time()
    n_atoms, n_channels, height_atom, width_atom = D.shape
    n_channels, height, width = X.shape
    h_world, w_world = grid_workers

    # Compute shape for the valid image
    height_valid = height - height_atom + 1
    width_valid = width - width_atom + 1

    params.update(lmbd=lmbd, valid_shape=(height_valid, width_valid),
                  grid_workers=grid_workers)
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
                      dest, tag=TAG_ROOT + dest)
            if params['debug']:
                X_worker = np.empty(X_alpha[X_worker_slice].shape, 'd')
                comm.Recv([X_worker.ravel(), MPI.DOUBLE],
                          source=dest, tag=TAG_ROOT + dest)
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


def _recv_result(comm, n_atoms, shape_valid, grid_workers, verbose=0):

    t_start = time()
    height_valid, width_valid = shape_valid
    h_world, w_world = grid_workers

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
                      tag=TAG_ROOT + dest)
            z_hat[worker_slice] = z_worker

    t_reduce = time() - t_start
    if verbose > 0:
        print('End finalization - {:.4}s'.format(t_reduce))
    return z_hat


class DICOD:
    """MPI implementation of the distributed convolutional pursuit

    Parameters
    ----------
    n_jobs: int (default: 1)
        Maximal number of process to solve this problem
    n_seg: int (default: 1)
        If >1, further segment the updates and update
        the best coordinate over each segment cyclically
    hostfile: str (default: None)
        Specify a hostfile for mpi launch, permit to use
        more than one computer
    logging: bool (default: False)
        Enable the logging of the updates to allow printing a
        cost curve
    verbose: int (default: 0)
        verbosity level
    positive: bool (default: False)
        If set to true, enforces a positivity constraint for the solution.
    algorithm : { ALGO_GS, ALGO_RANDOM }
        Algorithm to use in local updates for the distributed algorithm. Either
        use Gauss-southwell greedy updates (ALGO_GS: 0) or uniform selection
        (ALGO_RANDOM: 1)
    tol: float, (default: 1e-10)
        minimal step size, stop once no coordinate can move more than this tol.
    max_iter: int (default: 1000)
        Maximal number of iteration of the optimization using LGCD
    timeout: int (default: 40)
        Maximal time in seconds to run this algorithm
    """

    def __init__(self, n_jobs=1, n_seg=0, hostfile=None,
                 logging=False, verbose=0, positive=False, w_world='auto',
                 algorithm=ALGO_GS, patience=1000, tol=1e-10, max_iter=100,
                 timeout=40):
        self.tol = tol
        self.max_iter = max_iter
        self.timeout = timeout
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.w_world = w_world
        self.hostfile = hostfile
        self.logging = 1 if logging else 0
        self.n_seg = n_seg
        self.positive = 1 if positive else 0
        self.algorithm = algorithm
        self.patience = 1000
        self.name = 'MPI_DCP' + str(self.n_jobs)

    def fit(self, X, D, lmbd):
        self._spawn_workers()
        self._send_task(X, D, lmbd)

        self.end()

    def _spawn_workers(self):
        info = MPI.Info.Create()
        info.Set("map_bynode", '1')
        if self.hostfile and os.path.exists(self.hostfile):
            info.Set("hostfile", self.hostfile)
        script_name = os.path.join(os.path.dirname(__file__),
                                   "_dicod_2d_worker.py")
        self._comm = MPI.COMM_SELF.Spawn(sys.executable, args=[script_name],
                                         maxprocs=self.n_jobs, info=info)
        self._comm.Barrier()

    def _find_grid_size(self, width, height):
        w_world, h_world = 1, self.n_jobs
        w_ratio = width * self.n_jobs / height
        for i in range(2, self.n_jobs + 1):
            if self.n_jobs % i != 0:
                continue
            j = self.n_jobs // i
            ratio = width * j / (height * i)
            if abs(ratio - 1) < abs(w_ratio - 1):
                w_ratio = ratio
                w_world, h_world = i, j
        return w_world, h_world

    def _send_task(self, X, D, lmbd):
        n_atoms, n_channels, height_atom, width_atom = D.shape
        n_channels, height, width = X.shape

        # Compute shape for the valid image
        height_valid = height - height_atom + 1
        width_valid = width - width_atom + 1

        if self.w_world == 'auto':
            w_world, h_world = self._find_grid_size(width, height)
        else:
            w_world, h_world = self.w_world, self.n_jobs // self.w_world
            assert self.n_jobs % self.w_world == 0

        # Share constants
        alpha_k = np.sum(np.mean(D * D, axis=1),
                         axis=(1, 2), keepdims=True)
        alpha_k += (alpha_k == 0)

        DD = np.mean([[[fftconvolve(dk0, dk1, mode='full')
                        for dk0, dk1 in zip(d0, d1)]
                       for d1 in D]
                      for d0 in D[:, :, ::-1, ::-1]], axis=2)

        self._broadcast_array(alpha_k)
        self._broadcast_array(DD)
        self._broadcast_array(D)

        # Send the constants of the algorithm
        max_iter = max(1, self.max_iter // self.n_jobs)
        N = np.array([float(width), float(height),
                      float(w_world), float(lmbd), float(self.tol),
                      float(self.timeout), float(max_iter),
                      float(self.verbose), float(self.logging),
                      float(self.n_seg), float(self.positive),
                      float(self.algorithm), float(self.patience)],
                     'd')
        self._broadcast_array(N)

        # Share the work between the processes
        X = np.array(X, dtype='d')
        height_worker = height_valid // h_world
        width_worker = width_valid // w_world
        expect = []
        for i in range(h_world):
            h_end = (i+1)*height_worker + height_atom - 1
            if i == h_world - 1:
                h_end = height
            for j in range(w_world):
                dest = i * w_world + j
                w_end = (j+1) * width_worker + width_atom - 1
                if j == w_world - 1:
                    w_end = width
                self._comm.Send([X[:, i*height_worker:h_end,
                                     j*width_worker:w_end].flatten(),
                                MPI.DOUBLE], dest, tag=TAG_ROOT + dest)
                expect += [X[0, i*height_worker, j*width_worker],
                           X[-1, h_end-1, w_end-1]]
        self.t_start = time()
        self._confirm_array(expect)
        self.height_valid, self.height_worker = height_valid, height_worker
        self.width_valid, self.width_worker = width_valid, width_worker

        # Wait end of initialization
        self._comm.Barrier()
        self.t_init = time() - self.t_start
        if self.verbose > 1:
            print('End initialisation - {:.4}s'.format(self.t_init))

    def end(self):
        # reduce_pt
        self._comm.Barrier()
        log.debug("End computation, gather result")

        raise RuntimeError()
        self._gather()

        log.debug("DICOD - Clean end")

    def _gather(self):
        K, L, L_proc = self.K, self.L, self.L_proc
        pt = np.empty((K, L), 'd')

        for i in range(self.n_jobs):
            off = i*self.L_proc
            L_proc_i = min(off+L_proc, L)-off
            gpt = np.empty(K*L_proc_i, 'd')
            self._comm.Recv([gpt, MPI.DOUBLE], i, tag=200+i)
            pt[:, i*L_proc:(i+1)*L_proc] = gpt.reshape((K, -1))

        cost = np.empty(self.n_jobs, 'd')
        iterations = np.empty(self.n_jobs, 'i')
        times = np.empty(self.n_jobs, 'd')
        init_times = np.empty(self.n_jobs, 'd')
        self._comm.Gather(None, [cost, MPI.DOUBLE],
                          root=MPI.ROOT)
        self._comm.Gather(None, [iterations, MPI.INT],
                          root=MPI.ROOT)
        self._comm.Gather(None, [times, MPI.DOUBLE],
                          root=MPI.ROOT)
        self._comm.Gather(None, [init_times, MPI.DOUBLE],
                          root=MPI.ROOT)
        self.cost = np.sum(cost)
        self.iteration = np.sum(iterations)
        self.time = times.max()
        log.debug("Iterations {}".format(iterations))
        log.debug("Times {}".format(times))
        log.debug("Cost {}".format(cost))
        self.pb.pt = pt
        self.pt_dbg = np.copy(pt)
        log.info('End for {} : iteration {}, time {:.4}s'
                 .format(self, self.iteration, self.time))

        if self.logging:
            self._log(iterations)

        self._comm.Barrier()
        self.runtime = time()-self.t_start
        log.debug('Total time: {:.4}s'.format(self.runtime))

    def _log(self, iterations):
        self._comm.Barrier()
        pb, L = self.pb, self.L
        updates, updates_t, updates_skip = [], [], []
        for id_worker, n_iter in enumerate(iterations):
            _log = np.empty(4 * n_iter)
            self._comm.Recv([_log, MPI.DOUBLE], id_worker, tag=300 + id_worker)
            updates += [(int(_log[4 * i]), _log[4 * i + 2])
                        for i in range(n_iter)]
            updates_t += [_log[4 * i + 1] for i in range(n_iter)]
            updates_skip += [_log[4 * i + 3] for i in range(n_iter)]

        i0 = np.argsort(updates_t)
        self.next_log = 1
        pb.reset()
        log.debug('Start logging cost')
        t = self.t_init
        it = 0
        for i in i0:
            if it + 1 >= self.next_log:
                self.record(it, t, pb.cost(pb.pt))
            j, du = updates[i]
            t = updates_t[i] + self.t_init
            pb.pt[j // L, j % L] += du
            it += 1 + updates_skip[i]
        self.log_update = (updates_t, updates)
        log.debug('End logging cost')

    def gather_AB(self):
        K, S, d = self.K, self.S, self.d
        A = np.empty(K*K*S, 'd')
        B = np.empty(d*K*S, 'd')
        self._comm.Barrier()
        log.debug("End computation, gather result")

        self._comm.Reduce(None, [A, MPI.DOUBLE], op=MPI.SUM,
                          root=MPI.ROOT)
        self._comm.Reduce(None, [B, MPI.DOUBLE], op=MPI.SUM,
                          root=MPI.ROOT)

        iterations = np.empty(self.n_jobs, 'i')
        self._comm.Gather(None, [iterations, MPI.INT],
                          root=MPI.ROOT)
        self.iteration = np.sum(iterations)
        log.debug("Iterations {}".format(iterations))

        self._comm.Barrier()
        self.gather()
        return A, B

    def _broadcast_array(self, arr):
        arr_shape = np.array(arr.shape, dtype='i')
        arr = np.array(arr.flatten(), dtype='d')
        N = np.array([arr.shape[0], len(arr_shape)], dtype='i')

        # Send the data and shape of the numpy array
        self._comm.Bcast([N, MPI.INT], root=MPI.ROOT)
        self._comm.Bcast([arr_shape, MPI.INT], root=MPI.ROOT)
        self._comm.Bcast([arr, MPI.DOUBLE], root=MPI.ROOT)

    def _confirm_array(self, expect):
        '''Aux function to confirm that we passed the correct array
        '''
        expect = np.array(expect)
        gathering = np.empty(expect.shape, 'd')
        self._comm.Gather(None, [gathering, MPI.DOUBLE],
                          root=MPI.ROOT)
        assert (np.allclose(expect, gathering)), (
            expect, gathering, 'Fail to transmit array')

    def p_update(self):
        return 0

    def _stop(self, dz):
        return True


if __name__ == "__main__":
    print("Hello world! :", os.getpid())
    parent_comm = MPI.Comm.Get_parent()
    parent_comm.Barrier()
    print("ciao")
    parent_comm.Disconnect()
