# Authors: Thomas Moreau <thomas.moreau@inria.fr>

import time
import numpy as np
from mpi4py import MPI

from dicod_python.utils import check_random_state
from dicod_python.utils import debug_flags as flags
from dicod_python.utils import constants as constants
from dicod_python.utils.segmentation import Segmentation
from dicod_python.utils.mpi import recv_broadcasted_array
from dicod_python.utils.csc import compute_DtD, compute_norm_atoms

from dicod_python.coordinate_descent_2d import _select_coordinate
from dicod_python.coordinate_descent_2d import _check_convergence
from dicod_python.coordinate_descent_2d import _init_beta, _update_beta


class DICODWorker:
    """Worker for DICOD, running LGCD locally and using MPI for communications

    Parameters
    ----------
    backend: str
        Backend used to communicate between workers. Available backends are
        { 'mpi' }.
    """

    def __init__(self, backend):
        self._backend = backend

        # Retrieve different constants from the base communicator and store
        # then in the class.
        self.rank, self.n_jobs, params, self.D = self.get_params()

        self.tol = params['tol']
        self.lmbd = params['lmbd']
        self.n_seg = params['n_seg']
        self.timing = params['timing']
        self.verbose = params['verbose']
        self.strategy = params['strategy']
        self.max_iter = params['max_iter']
        self.z_positive = params['z_positive']
        self.use_soft_lock = params['use_soft_lock']

        self.random_state = params['random_state']
        if isinstance(self.random_state, int):
            self.random_state += self.rank

        self.info("Start DICOD with {} workers and strategy '{}'", self.n_jobs,
                  self.strategy, global_msg=True)

        n_atoms, n_channels, *atom_shape = self.D.shape
        self.overlap = [size_atom_ax - 1 for size_atom_ax in atom_shape]
        self.workers_segments = Segmentation(
            n_seg=params['workers_topology'],
            signal_shape=params['valid_shape'],
            overlap=self.overlap)

        worker_shape = self.workers_segments.get_seg_shape(self.rank)
        X_shape = (n_channels,) + tuple([
            size_worker_ax + size_atom_ax - 1
            for size_worker_ax, size_atom_ax in zip(worker_shape, atom_shape)
        ])
        self.X_worker = self.get_signal(X_shape, params['debug'])

        n_seg = self.n_seg
        seg_shape = None
        if self.n_seg == 'auto':
            n_seg, seg_shape = None, []
            for size_atom_ax in atom_shape:
                seg_shape.append(2 * size_atom_ax - 1)

        # Get local inner bounds. First, compute the seg_bound without overlap
        # in gocal coordinates and then convert the bounds in the local
        # coordinate system.
        inner_bounds = self.workers_segments.get_seg_bounds(
            self.rank, only_inner=True)
        inner_bounds = np.transpose([
            self.workers_segments.get_local_coordinate(self.rank, bound)
            for bound in np.transpose(inner_bounds)])

        # get a segmentation for the local LGCD
        self.local_segments = Segmentation(n_seg=n_seg, seg_shape=seg_shape,
                                           inner_bounds=inner_bounds,
                                           full_shape=worker_shape)

        self.synchronize_workers()

    def run(self):

        self._init_cd_variables()

        # Initialization of the algorithm variables
        random_state = check_random_state(self.random_state)
        i_seg = -1
        accumulator = 0
        k0, pt0 = 0, None
        self.n_paused_worker = 0

        # Initialize the solution
        n_atoms = self.D.shape[0]
        seg_shape = self.workers_segments.get_seg_shape(self.rank)
        self.z_hat = np.zeros((n_atoms,) + seg_shape)
        n_coordinates = self.z_hat.size

        t_start = time.time()
        diverging = False
        if flags.INTERACTIVE_PROCESSES and self.n_jobs == 1:
            import ipdb; ipdb.set_trace()  # noqa: E702
        for ii in range(self.max_iter):
            # Display the progress of the algorithm
            self.progress(ii, max_ii=self.max_iter, unit="iterations")

            # Process incoming messages
            self._process_messages()

            # Increment the segment and select the coordinate to update
            i_seg = self.local_segments.increment_seg(i_seg)
            if self.local_segments.is_active_segment(i_seg):
                k0, pt0, dz = _select_coordinate(
                    self.dz_opt, self.local_segments, i_seg,
                    strategy=self.strategy, random_state=random_state)
            else:
                k0, pt0, dz = None, None, 0

            # update the accumulator for 'random' strategy
            accumulator = max(abs(dz), accumulator)

            # If requested, check that the update chosen only have an impact on
            # the segment and its overlap area.
            if flags.CHECK_UPDATE_CONTAINED and pt0 is not None:
                self.workers_segments.check_area_contained(self.rank,
                                                           pt0, self.overlap)

            # Check if gthe coordinate is soft-locked or not.
            soft_locked = False
            if pt0 is not None and self.use_soft_lock:
                lock_slices = self.workers_segments.get_touched_overlap_slices(
                    self.rank, pt0, self.overlap)
                max_on_lock = 0
                for u_slice in lock_slices:
                    # print(self.rank, inner_bounds, u_slice, max_on_lock)
                    max_on_lock = max(abs(self.dz_opt[u_slice]).max(),
                                      max_on_lock)
                soft_locked = max_on_lock > abs(dz)

            # Update the selected coordinate and beta, only if the update is
            # greater than the convergence tolerance and is contained in the
            # worker. If the update is not in the worker, this will
            # effectively work has a soft lock to prevent interferences.
            if abs(dz) > self.tol and not soft_locked:
                # update the selected coordinate and beta
                self.z_hat[(k0,) + pt0] += dz
                self._update_beta(k0, pt0, dz)

                # Re-activate the segments where beta have been updated to
                # ensure convergence.
                touched_segments = self.local_segments.get_touched_segments(
                    pt=pt0, radius=self.overlap)
                n_changed_status = self.local_segments.set_active_segments(
                    touched_segments)

                # Notify neighboring workers of the update if needed.
                pt_global = self.workers_segments.get_global_coordinate(
                    self.rank, pt0)
                workers = self.workers_segments.get_touched_segments(
                    pt=pt_global, radius=self.overlap
                )
                msg = np.array([k0, *pt_global, dz], 'd')
                self.notify_neighbors(msg, workers)

                # If requested, check that all inactive segments have no
                # coefficients to update over the tolerance.
                if flags.CHECK_ACTIVE_SEGMENTS and n_changed_status:
                    self.local_segments.test_active_segments(
                        self.dz_opt, self.tol)

            # Inactivate the current segment if the magnitude of the update is
            # too small. This only work when using LGCD.
            if abs(dz) <= self.tol and self.strategy == "greedy":
                self.local_segments.set_inactive_segments(i_seg)

            # When workers are diverging, finish the worker to avoid having to
            # wait until max_iter for stopping the algorithm.
            if abs(dz) >= 1e2:
                self.info("diverging worker")
                self.wait_status_changed(status=constants.STATUS_FINISHED)
                diverging = True
                break

            if self.timing:
                # TODO: logging stuff
                pass

            # Check the stopping criterion and if we have locally converged,
            # wait either for an incoming message or for full convergence.
            if _check_convergence(self.local_segments, self.tol, ii,
                                  self.dz_opt, n_coordinates, self.strategy,
                                  accumulator=accumulator):
                if self.check_no_transitting_message():
                    status = self.wait_status_changed()
                    if status == constants.STATUS_STOP:
                        self.debug("LGCD converged after {} iterations",
                                   ii + 1)
                        break

        else:
            self.info("Reach maximal iteration. Max of dz_opt is {}.",
                      abs(self.dz_opt).max())
            self.wait_status_changed(status=constants.STATUS_FINISHED)

        self.synchronize_workers()
        assert diverging or self.check_no_transitting_message()
        runtime = time.time() - t_start
        self.send_result(ii + 1, runtime)

    def progress(self, ii, max_ii, unit):
        if max_ii > 10000 and (ii % 100) != 0:
            return
        self._log("progress : {:7.2%} {}", ii / max_ii, unit, level=1,
                  level_name="PROGRESS", global_msg=True, endline=False)

    def _log(self, msg, *fmt_args, level=0, level_name="None",
             global_msg=False, endline=True):
        if self.verbose >= level:
            if global_msg:
                if self.rank != 0:
                    return
                msg_fmt = constants.GLOBAL_OUTPUT_TAG + msg
                identity = self.n_jobs
            else:
                msg_fmt = constants.WORKER_OUTPUT_TAG + msg
                identity = self.rank
            if endline:
                kwargs = {}
            else:
                kwargs = {'end': '', 'flush': True}
            msg_fmt = msg_fmt.ljust(80)
            print(msg_fmt.format(level_name, identity, *fmt_args), **kwargs)

    def info(self, msg, *args, global_msg=False):
        self._log(msg, *args, level=1, level_name="INFO",
                  global_msg=global_msg)

    def debug(self, msg, *args, global_msg=False):
        self._log(msg, *args, level=5, level_name="DEBUG",
                  global_msg=global_msg)

    def _init_cd_variables(self):

        # Pre-compute some quantities
        constants = {}
        constants['norm_atoms'] = compute_norm_atoms(self.D)
        constants['DtD'] = compute_DtD(self.D)
        self.constants = constants

        # List of all pending messages
        self.messages = []

        # Initialization of the auxillary variable for LGCD
        self.beta, self.dz_opt = _init_beta(
            self.X_worker, self.D, self.lmbd, z_i=None, constants=constants,
            z_positive=self.z_positive)

    def _update_beta(self, k0, pt0, dz, coordinate_exist=True):
        self.beta, self.dz_opt = _update_beta(
            k0, pt0, dz, self.beta, self.dz_opt, self.z_hat, self.D,
            self.lmbd, self.constants, self.z_positive,
            coordinate_exist=coordinate_exist)

    def _process_messages(self, worker_status=constants.STATUS_RUNNING):

        mpi_status = MPI.Status()
        while MPI.COMM_WORLD.Iprobe(status=mpi_status):
            src = mpi_status.source
            tag = mpi_status.tag
            if tag == constants.TAG_UPDATE_BETA:
                if worker_status == constants.STATUS_PAUSED:
                    self.notify_worker_status(constants.TAG_RUNNING_WORKER,
                                              wait=True)
                    worker_status = constants.STATUS_RUNNING
            elif tag == constants.TAG_STOP:
                worker_status = constants.STATUS_STOP
            elif tag == constants.TAG_PAUSED_WORKER:
                self.n_paused_worker += 1
                assert self.n_paused_worker <= self.n_jobs
            elif tag == constants.TAG_RUNNING_WORKER:
                self.n_paused_worker -= 1
                assert self.n_paused_worker >= 0

            msg = np.empty(constants.SIZE_MSG, 'd')
            MPI.COMM_WORLD.Recv([msg, MPI.DOUBLE], source=src,
                                tag=tag)

            if tag == constants.TAG_UPDATE_BETA:
                self._message_update_beta(msg)

        if self.n_paused_worker == self.n_jobs:
            worker_status = constants.STATUS_STOP
        return worker_status

    def _message_update_beta(self, msg):
        k0, h0, w0, dz = msg

        k0, h0, w0 = int(k0), int(h0), int(w0)
        h0, w0 = self.workers_segments.get_local_coordinate(self.rank,
                                                            (h0, w0))
        height, width = self.beta.shape[1:]
        coordinate_exist = (0 <= h0 < height) and (0 <= w0 < width)
        if coordinate_exist:
            self.z_hat[k0, h0, w0] += dz
        self._update_beta(k0, (h0, w0), dz, coordinate_exist=coordinate_exist)
        touched_segment = self.local_segments.get_touched_segments(
            pt=(h0, w0), radius=self.overlap)
        n_changed_status = self.local_segments.set_active_segments(
            touched_segment)

        if flags.CHECK_ACTIVE_SEGMENTS and n_changed_status:
            self.local_segments.test_active_segments(self.dz_opt, self.tol)

    def notify_neighbors(self, msg, neighbors):
        assert self.rank in neighbors
        for i_neighbor in neighbors:
            if i_neighbor != self.rank:
                req = self.send_message(msg, constants.TAG_UPDATE_BETA,
                                        i_neighbor, wait=False)
                self.messages.append(req)

    def notify_worker_status(self, tag, i_worker=0, wait=False):
        # handle the messages from Worker0 to himself.
        if self.rank == 0 and i_worker == 0:
            if tag == constants.TAG_PAUSED_WORKER:
                self.n_paused_worker += 1
                assert self.n_paused_worker <= self.n_jobs
            elif tag == constants.TAG_RUNNING_WORKER:
                self.n_paused_worker -= 1
                assert self.n_paused_worker >= 0
            else:
                raise ValueError("Got tag {}".format(tag))
            return

        # Else send the message to the required destination
        msg = np.empty(constants.SIZE_MSG, 'd')
        self.send_message(msg, tag, i_worker, wait=wait)

    def wait_status_changed(self, status=constants.STATUS_PAUSED):
        self.notify_worker_status(constants.TAG_PAUSED_WORKER)
        self.debug("paused worker")

        # Wait for all sent message to be processed
        count = 0
        while status not in [constants.STATUS_RUNNING, constants.STATUS_STOP]:
            time.sleep(.001)
            status = self._process_messages(worker_status=status)
            if (count % 500) == 0:
                self.progress(self.n_paused_worker, max_ii=self.n_jobs,
                              unit="done workers")

        if self.rank == 0 and status == constants.STATUS_STOP:
            for i_worker in range(1, self.n_jobs):
                self.notify_worker_status(constants.TAG_STOP, i_worker,
                                          wait=True)
        elif status == constants.STATUS_RUNNING:
            self.debug("wake up")
        return status

    ###########################################################################
    #     Communication primitives
    ###########################################################################

    def synchronize_workers(self):
        if self._backend == "mpi":
            self._synchronize_workers_mpi()
        else:
            raise NotImplementedError("Backend {} is not implemented"
                                      .format(self._backend))

    def get_params(self):
        """Receive the parameter of the algorithm from the master node."""
        if self._backend == "mpi":
            return self._get_params_mpi()
        else:
            raise NotImplementedError("Backend {} is not implemented"
                                      .format(self._backend))

    def get_signal(self, X_shape, debug=False):
        """Receive the part of the signal to encode from the master node."""
        if self._backend == "mpi":
            return self._get_signal_mpi(X_shape, debug=debug)
        else:
            raise NotImplementedError("Backend {} is not implemented"
                                      .format(self._backend))

    def send_message(self, msg, tag, i_worker, wait=False):
        """Send a message to a specified worker."""
        assert self.rank != i_worker
        if self._backend == "mpi":
            return self._send_message_mpi(msg, tag, i_worker, wait=wait)
        else:
            raise NotImplementedError("Backend {} is not implemented"
                                      .format(self._backend))

    def send_result(self, iterations, runtime):
        if self._backend == "mpi":
            self._send_result_mpi(iterations, runtime)
        else:
            raise NotImplementedError("Backend {} is not implemented"
                                      .format(self._backend))

    def shutdown(self):
        if self._backend == "mpi":
            self._shutdown_mpi()
        else:
            raise NotImplementedError("Backend {} is not implemented"
                                      .format(self._backend))

    ###########################################################################
    #     mpi4py implementation
    ###########################################################################

    def _synchronize_workers_mpi(self):
        comm = MPI.Comm.Get_parent()
        comm.Barrier()

    def check_no_transitting_message(self):
        """Check no message is in waiting to complete to or from this worker"""
        if MPI.COMM_WORLD.Iprobe():
            return False
        while self.messages:
            if not self.messages[0].Test() or MPI.COMM_WORLD.Iprobe():
                return False
            self.messages.pop(0)
        return True

    def _get_params_mpi(self):
        comm = MPI.Comm.Get_parent()

        rank = comm.Get_rank()
        n_jobs = comm.Get_size()
        params = comm.bcast(None, root=0)
        D = recv_broadcasted_array(comm)
        return rank, n_jobs, params, D

    def _get_signal_mpi(self, X_shape, debug):
        comm = MPI.Comm.Get_parent()
        rank = comm.Get_rank()

        X_worker = np.empty(X_shape, dtype='d')

        comm.Recv([X_worker.ravel(), MPI.DOUBLE], source=0,
                  tag=constants.TAG_ROOT + rank)

        if debug:

            X_alpha = 0.25 * np.ones(X_shape)
            comm.Send([X_alpha.ravel(), MPI.DOUBLE], dest=0,
                      tag=constants.TAG_ROOT + rank)

        return X_worker

    def _send_message_mpi(self, msg, tag, i_worker, wait=False):
        if wait:
            return MPI.COMM_WORLD.Ssend([msg, MPI.DOUBLE], i_worker, tag=tag)
        else:
            return MPI.COMM_WORLD.Isend([msg, MPI.DOUBLE], i_worker, tag=tag)

    def _send_result_mpi(self, iterations, runtime):
        comm = MPI.Comm.Get_parent()

        res_slice = (Ellipsis,) + tuple([
            slice(start, end)
            for start, end in self.local_segments.inner_bounds
        ])

        z_worker = self.z_hat[res_slice].ravel()
        comm.Send([z_worker, MPI.DOUBLE], dest=0,
                  tag=constants.TAG_ROOT + self.rank)
        comm.gather([iterations, runtime], root=0)
        comm.Barrier()

    def _shutdown_mpi(self):
        comm = MPI.Comm.Get_parent()
        self.debug("clean shutdown")
        comm.Barrier()
        comm.Disconnect()


if __name__ == "__main__":
    dicod = DICODWorker(backend='mpi')
    dicod.run()
    dicod.shutdown()
