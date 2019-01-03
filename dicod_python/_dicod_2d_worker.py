# Authors: Thomas Moreau <thomas.moreau@inria.fr>

import time
import numpy as np
from mpi4py import MPI

from dicod_python.utils import constants as csts
from dicod_python.utils import check_random_state
from dicod_python.utils import debug_flags as flags
from dicod_python.utils.segmentation import Segmentation
from dicod_python.utils.mpi import recv_broadcasted_array
from dicod_python.utils.csc import compute_DtD, compute_norm_atoms

from dicod_python.coordinate_descent_2d import _select_coordinate
from dicod_python.coordinate_descent_2d import _check_convergence
from dicod_python.coordinate_descent_2d import _init_beta, _update_beta


class DICODWorker:
    """Utility to move from on coordinate system to another easily
    """

    def __init__(self, backend):
        # this function receive the parameters and the task from the master
        # node.
        self._backend = backend

        # Retrieve different constants from the base communicator
        self.rank, self.n_jobs, params, self.D = self.get_params()

        self.tol = params['tol']
        self.lmbd = params['lmbd']
        self.n_seg = params['n_seg']
        self.timing = params['timing']
        self.verbose = params['verbose']
        self.strategy = params['strategy']
        self.max_iter = params['max_iter']
        self.z_positive = params['z_positive']
        self.worker_topology = params['worker_topology']

        self.random_state = params['random_state']
        if isinstance(self.random_state, int):
            self.random_state += self.rank

        height_valid, width_valid = params['valid_shape']
        n_atoms, n_channels, *atom_shape = self.D.shape
        height_atom, width_atom = atom_shape
        h_world, w_world = self.worker_topology

        self.info("Start DICOD with {} workers and strategy '{}'", self.n_jobs,
                  self.strategy, global_msg=True)

        self.workers_segments = Segmentation(
            self.worker_topology, signal_shape=params['valid_shape'])

        # Infer some topological information
        h_rank = self.rank // w_world
        w_rank = self.rank % w_world

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

        if self.n_seg == 'auto':
            self.n_seg = []
            for axis_size, atom_size in zip(
                    self.workers_segments.signal_shape, atom_shape):
                self.n_seg.append(max(axis_size // (2 * atom_size - 1), 1))
        self.local_segments = Segmentation(
            self.n_seg, outer_bounds=worker_bounds)

        X_shape = (n_channels, h_end - h_start, w_end - w_start)
        self.X_worker = self.get_signal(X_shape)

        self.offset = worker_offset
        self.bounds = worker_bounds
        self.synchronize_workers()

    def get_params(self):
        if self._backend == "mpi":
            return self._get_params_mpi()
        else:
            raise NotImplementedError("Backend {} is not implemented"
                                      .format(self._backend))

    def _get_params_mpi(self):
        comm = MPI.Comm.Get_parent()

        rank = comm.Get_rank()
        n_jobs = comm.Get_size()
        params = comm.bcast(None, root=0)
        D = recv_broadcasted_array(comm)
        return rank, n_jobs, params, D

    def get_signal(self, X_shape):
        if self._backend == "mpi":
            return self._get_signal_mpi(X_shape)
        else:
            raise NotImplementedError("Backend {} is not implemented"
                                      .format(self._backend))

    def _get_signal_mpi(self, X_shape):
        comm = MPI.Comm.Get_parent()
        rank = comm.Get_rank()

        X_worker = np.empty(X_shape, dtype='d')
        comm.Recv([X_worker.ravel(), MPI.DOUBLE], source=0,
                  tag=csts.TAG_ROOT + rank)

        return X_worker

    def synchronize_workers(self):
        if self._backend == "mpi":
            comm = MPI.Comm.Get_parent()
            comm.Barrier()
        else:
            raise NotImplementedError("Backend {} is not implemented"
                                      .format(self._backend))

    def run(self):

        n_channels, height_worker, width_worker = self.X_worker.shape
        n_atoms, n_channels, height_atom, width_atom = self.D.shape
        self.atom_shape = atom_shape = (height_atom, width_atom)
        height_valid_worker = height_worker - height_atom + 1
        width_valid_worker = width_worker - width_atom + 1
        n_coordinates = n_atoms * height_valid_worker * width_valid_worker

        self._init_cd_variables()

        # Initialization of the algorithm variables
        random_state = check_random_state(self.random_state)
        i_seg = -1
        k0, h0, w0 = 0, -1, -1
        self.n_paused_worker = 0
        self.z_hat = z_hat = np.zeros((n_atoms, height_valid_worker,
                                       width_valid_worker))

        t_start = time.time()
        if flags.INTERACTIVE_PROCESSES and self.n_jobs == 1:
            import ipdb; ipdb.set_trace()  # noqa: E702
        for ii in range(self.max_iter):
            self.progress(ii)

            self._process_messages()

            i_seg = self.local_segments.increment_seg(i_seg)
            seg_bounds = self.local_segments.get_seg_bounds(i_seg)
            if self.active_segments[i_seg]:
                k0, pt0, dz = _select_coordinate(self.dz_opt, seg_bounds,
                                                 strategy=self.strategy,
                                                 random_state=random_state)
            else:
                k0, pt0, dz = None, None, 0

            # Update the selected coordinate and beta, only if the update is
            # greater than the convergence tolerance and is contained in the
            # worker. If the update is not in the worker, this will
            # effectively work has a soft lock to prevent interferences.
            if abs(dz) > self.tol:
                if self.is_valid_local_coordinate(pt0):
                    # update the selected coordinate
                    z_hat[(k0,) + pt0] += dz

                    # update beta
                    self._update_beta(k0, pt0, dz)

                    touched_segs = self.local_segments.get_touched_segments(
                        pt=pt0, radius=atom_shape)
                    n_changed_status = self.local_segments.set_active_segments(
                        touched_segs)

                    pt_global = self.get_global_coordinate(pt0)
                    workers = self.workers_segments.get_touched_segments(
                        pt=pt_global, radius=[2 * s - 1 for s in atom_shape]
                    )

                    msg = np.array([k0, *pt_global, dz], 'd')
                    self.notify_neighbors(msg, workers)

                    # Debug utility to check that the inactive segments have no
                    # coefficients to update over the tolerance.
                    if flags.CHECK_ACTIVE_SEGMENTS and n_changed_status:
                        self.local_segments.test_active_segments(
                            self.dz_opt, self.tol)
                else:
                    raise RuntimeError(self.rank, (k0, h0, w0), seg_bounds)

                if self.strategy == 'random':
                    # accumulate on all coordinates from the stopping criterion
                    if ii % n_coordinates == 0:
                        accumulator = 0
                    accumulator = max(accumulator, abs(dz))

            elif self.strategy == "greedy":
                self.local_segments.set_inactive_segments(i_seg)

            if self.timing:
                # TODO: logging stuff
                pass

            # check stopping criterion
            if _check_convergence(self.local_segments, self.tol, ii,
                                  self.dz_opt, n_coordinates, self.strategy):
                if self.check_no_transitting_message():
                    wake_up, stop = self.wait_for_wakeup_or_stop()
                    if stop:
                        self.info("LGCD converged after {} iterations", ii + 1)
                        break
                else:
                    time.sleep(.001)

        else:
            self.notify_worker_status(csts.TAG_PAUSED_WORKER)
            self.info("Reach maximal iteration. Max of dz_opt is {}.",
                      abs(self.dz_opt).max())

        self.synchronize_workers()
        assert not MPI.COMM_WORLD.Iprobe()
        runtime = time.time() - t_start
        self.send_result(ii + 1, runtime)
        self.shutdown()

    def check_no_transitting_message(self):
        if MPI.COMM_WORLD.Iprobe():
            return False
        for req in self.messages:
            if not req.Test():
                return False
            if MPI.COMM_WORLD.Iprobe():
                return False
        return True

    def progress(self, ii):
        self._log("progress : {:7.2%}", ii / self.max_iter, level=1,
                  level_name="PROGRESS", global_msg=True, endline=False)

    def _log(self, msg, *fmt_args, level=0, level_name="None",
             global_msg=False, endline=True):
        if self.verbose >= level:
            if global_msg:
                if self.rank != 0:
                    return
                msg_fmt = csts.GLOBAL_OUTPUT_TAG + msg
                identity = self.n_jobs
            else:
                msg_fmt = csts.WORKER_OUTPUT_TAG + msg
                identity = self.rank
            if endline:
                kwargs = {}
            else:
                kwargs = {'end': '', 'flush': True}
            print(msg_fmt.format(level_name, identity, *fmt_args), **kwargs)

    def info(self, msg, *args, global_msg=False):
        self._log(msg, *args, level=1, level_name="INFO",
                  global_msg=global_msg)

    def debug(self, msg, *args, global_msg=False):
        self._log(msg, *args, level=5, level_name="DEBUG",
                  global_msg=global_msg)

    def _init_cd_variables(self):

        # compute sizes for the segments for LGCD
        n_atoms, n_channels, *atom_shape = self.D.shape
        effective_n_seg = self.local_segments.effective_n_seg

        # Pre-compute some quantities
        constants = {}
        constants['norm_atoms'] = compute_norm_atoms(self.D)
        constants['DtD'] = compute_DtD(self.D)
        self.constants = constants

        # List of all pending messages
        self.messages = []

        # Initialization of the algorithm variables
        self.accumulator = effective_n_seg
        self.active_segments = np.array([True] * effective_n_seg)

        self.beta, self.dz_opt = _init_beta(
            self.X_worker, self.D, self.lmbd, z_i=None, constants=constants,
            z_positive=self.z_positive)

    def _update_beta(self, k0, pt0, dz, coordinate_exist=True):
        self.beta, self.dz_opt = _update_beta(
            k0, pt0, dz, self.beta, self.dz_opt, self.z_hat, self.D,
            self.lmbd, self.constants, self.z_positive,
            coordinate_exist=coordinate_exist)

    def notify_neighbors(self, msg, neighbors):
        if self._backend == "mpi":
            for i_neighbor in neighbors:
                if i_neighbor != self.rank:
                    req = MPI.COMM_WORLD.Issend([msg, MPI.DOUBLE],
                                                dest=i_neighbor,
                                                tag=csts.TAG_UPDATE_BETA)
                    self.messages.append(req)
        else:
            raise NotImplementedError("Backend {} is not implemented"
                                      .format(self._backend))

    def _process_messages(self, paused_worker=False):

        status = MPI.Status()
        wake_up, stop = False, False
        while MPI.COMM_WORLD.Iprobe(status=status):
            src = status.source
            tag = status.tag
            if tag == csts.TAG_UPDATE_BETA:
                if paused_worker:
                    if self.rank != 0:
                        self.notify_worker_status(csts.TAG_WAKE_UP_WORKER,
                                                  wait=True)
                    else:
                        self.n_paused_worker -= 1
                    paused_worker = False
                wake_up = True
            elif tag == csts.TAG_STOP:
                stop = True
            elif tag == csts.TAG_PAUSED_WORKER:
                self.n_paused_worker += 1
                assert self.n_paused_worker <= self.n_jobs
            elif tag == csts.TAG_WAKE_UP_WORKER:
                self.n_paused_worker -= 1
                assert self.n_paused_worker >= 0

            msg = np.empty(csts.SIZE_MSG, 'd')
            MPI.COMM_WORLD.Recv([msg, MPI.DOUBLE], source=src,
                                tag=tag)

            if tag == csts.TAG_UPDATE_BETA:
                self._message_update_beta(msg)
            status = MPI.Status()

        return wake_up, (self.n_paused_worker == self.n_jobs or stop)

    def _message_update_beta(self, msg):
        k0, h0, w0, dz = msg

        k0, h0, w0 = int(k0), int(h0), int(w0)
        h0, w0 = self.get_local_coordinate((h0, w0))
        height, width = self.beta.shape[1:]
        coordinate_exist = (0 <= h0 < height) and (0 <= w0 < width)
        if coordinate_exist:
            self.z_hat[k0, h0, w0] += dz
        self._update_beta(k0, (h0, w0), dz, coordinate_exist=coordinate_exist)
        touched_segment = self.local_segments.get_touched_segments(
            pt=(h0, w0), radius=self.atom_shape)
        n_changed_status = self.local_segments.set_active_segments(
            touched_segment)

        if flags.CHECK_ACTIVE_SEGMENTS and n_changed_status:
            self.local_segments.test_active_segments(self.dz_opt, self.tol)

    def get_global_coordinate(self, pt):
        """Convert a point from local coordinate to global coordinate

        Parameters
        ----------
        pt: (int, int)
            Coordinate to convert, from the local coordinate system.

        Return
        ------
        pt : (int, int)
            Coordinate converted in the global coordinate system.
        """
        res = []
        for v, offset, (padding, _) in zip(pt, self.offset, self.bounds):
            res += [v - padding + offset]
        return tuple(res)

    def get_local_coordinate(self, pt):
        """Convert a point from global coordinate to local coordinate

        Parameters
        ----------
        pt: (int, int)
            Coordinate to convert, from the global coordinate system.

        Return
        ------
        pt : (int, int)
            Coordinate converted in the local coordinate system.
        """
        res = []
        for v, offset, (padding, _) in zip(pt, self.offset, self.bounds):
            res += [v + padding - offset]
        return tuple(res)

    def is_valid_local_coordinate(self, pt):
        """Ensure that a given point is in the bounds to be a local coordinate.
        """
        is_valid = True
        for v, (v_start, v_end) in zip(pt, self.bounds):
            is_valid &= (v_start <= v < v_end)
        return is_valid

    def send_result(self, iterations, runtime):
        if self._backend == "mpi":
            self._send_result_mpi(iterations, runtime)
        else:
            raise NotImplementedError("Backend {} is not implemented"
                                      .format(self._backend))

    def _send_result_mpi(self, iterations, runtime):
        comm = MPI.Comm.Get_parent()

        (h_start, h_end), (w_start, w_end) = self.bounds

        z_worker = self.z_hat[:, h_start:h_end, w_start:w_end].ravel()
        comm.Send([z_worker, MPI.DOUBLE], dest=0,
                  tag=csts.TAG_ROOT + self.rank)
        comm.gather([iterations, runtime], root=0)
        comm.Barrier()

    def shutdown(self):
        comm = MPI.Comm.Get_parent()
        self.debug("clean shutdown")
        comm.Barrier()
        comm.Disconnect()

    def notify_worker_status(self, tag, i_worker=0, wait=False):
        msg = np.empty(csts.SIZE_MSG, 'd')
        if wait:
            MPI.COMM_WORLD.Ssend([msg, MPI.DOUBLE], i_worker, tag=tag)
        else:
            MPI.COMM_WORLD.Isend([msg, MPI.DOUBLE], i_worker, tag=tag)

    def wait_for_wakeup_or_stop(self):
        if self.rank != 0:
            self.notify_worker_status(csts.TAG_PAUSED_WORKER)
        else:
            self.n_paused_worker += 1
        self.debug("paused_worker")

        # Wait for all sent message to be processed
        wake_up, stop = False, False
        while not (stop or wake_up):
            time.sleep(.001)
            wake_up, stop = self._process_messages(paused_worker=True)

        if self.rank == 0 and stop:
            for i_worker in range(1, self.n_jobs):
                self.notify_worker_status(csts.TAG_STOP, i_worker, wait=True)
        elif not stop:
            self.debug("wake up")
        return wake_up, stop


if __name__ == "__main__":
    dicod = DICODWorker(backend='mpi')
    dicod.run()
    # verbose = dicod_worker(comm)
    # shutdown(comm, verbose)
