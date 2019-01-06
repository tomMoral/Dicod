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
        overlap = [size_atom_ax - 1 for size_atom_ax in atom_shape]
        self.workers_segments = Segmentation(
            n_seg=params['workers_topology'],
            signal_shape=params['valid_shape'],
            overlap=overlap)

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
        k0, pt0 = 0, None
        self.n_paused_worker = 0
        self.z_hat = z_hat = np.zeros((n_atoms, height_valid_worker,
                                       width_valid_worker))

        t_start = time.time()
        diverging = False
        if flags.INTERACTIVE_PROCESSES and self.n_jobs == 1:
            import ipdb; ipdb.set_trace()  # noqa: E702
        for ii in range(self.max_iter):
            self.progress(ii, max_ii=self.max_iter, unit="iterations")

            self._process_messages()

            i_seg = self.local_segments.increment_seg(i_seg)
            seg_bounds = self.local_segments.get_seg_bounds(i_seg)
            if self.local_segments.is_active_segment(i_seg):
                k0, pt0, dz = _select_coordinate(self.dz_opt, seg_bounds,
                                                 strategy=self.strategy,
                                                 random_state=random_state)
            else:
                k0, pt0, dz = None, None, 0

            if flags.CHECK_UPDATE_CONTAINED and pt0 is not None:
                update_bounds = [
                    [v - size_atom_ax + 1, v + size_atom_ax - 1]
                    for v, size_atom_ax in zip(pt0, atom_shape)
                ]
                assert self.workers_segments.is_inner_coordinate(
                    self.rank, pt0), (self.rank, pt0, seg_bounds)
                seg_bounds = self.workers_segments.get_seg_bounds(self.rank)
                seg_bounds_inner = self.workers_segments.get_seg_bounds(
                    self.rank, only_inner=True)
                for i in range(2):
                    assert (update_bounds[i][0] >= 0 or
                            seg_bounds[i][0] == seg_bounds_inner[i][0]
                            ), (self.rank, update_bounds, self.beta.shape,
                                seg_bounds, seg_bounds_inner)
                    assert (update_bounds[i][1] <= self.beta.shape[i + 1]
                            or seg_bounds[i][1] == seg_bounds_inner[i][1]
                            ), (self.rank, update_bounds, self.beta.shape,
                                seg_bounds, seg_bounds_inner)

            should_update = True
            if pt0 is not None and self.use_soft_lock:
                update_bounds = [
                    [min(max(0, v - size_atom_ax + 1), size_valid_ax),
                     max(min(v + size_atom_ax - 1, size_valid_ax), 0)]
                    for v, size_atom_ax, size_valid_ax in zip(
                        pt0, atom_shape, self.z_hat.shape[1:])
                ]
                inner_bounds = self.local_segments.inner_bounds
                updated_slice = []
                pre_slice = (Ellipsis,)
                post_slice = tuple([slice(start, end)
                                    for start, end in update_bounds[1:]])
                for (start, end), (start_inner, end_inner) in zip(
                        update_bounds, inner_bounds):
                    if start < start_inner:
                        assert start_inner < end < end_inner
                        updated_slice.append(
                            pre_slice + (slice(start, start_inner),) +
                            post_slice)
                    if end > end_inner:
                        assert start_inner < start < end_inner
                        updated_slice.append(
                            pre_slice + (slice(end_inner, end),) +
                            post_slice)
                    pre_slice = pre_slice + (slice(start, end),)
                    post_slice = post_slice[1:]
                max_on_slice = 0
                for u_slice in updated_slice:
                    # print(self.rank, inner_bounds, u_slice, max_on_slice)
                    max_on_slice = max(abs(self.dz_opt[u_slice]).max(),
                                       max_on_slice)
                should_update = max_on_slice < abs(dz)

            # Update the selected coordinate and beta, only if the update is
            # greater than the convergence tolerance and is contained in the
            # worker. If the update is not in the worker, this will
            # effectively work has a soft lock to prevent interferences.
            if abs(dz) > self.tol:
                if should_update:

                    # update the selected coordinate
                    z_hat[(k0,) + pt0] += dz

                    # update beta
                    self._update_beta(k0, pt0, dz)

                    touched_segments = self.local_segments.get_touched_segments(
                        pt=pt0, radius=atom_shape)
                    n_changed_status = self.local_segments.set_active_segments(
                        touched_segments)

                    pt_global = self.workers_segments.get_global_coordinate(
                        self.rank, pt0)
                    workers = self.workers_segments.get_touched_segments(
                        pt=pt_global, radius=[s - 1 for s in atom_shape]
                    )
                    msg = np.array([k0, *pt_global, dz], 'd')
                    self.notify_neighbors(msg, workers)

                    # Debug utility to check that the inactive segments have no
                    # coefficients to update over the tolerance.
                    if flags.CHECK_ACTIVE_SEGMENTS and n_changed_status:
                        self.local_segments.test_active_segments(
                            self.dz_opt, self.tol)

                    if self.strategy == 'random':
                        # accumulate on all coordinates from the stopping criterion
                        if ii % n_coordinates == 0:
                            accumulator = 0
                        accumulator = max(accumulator, abs(dz))
                else:
                    self.debug("Should not update")

            elif self.strategy == "greedy":
                self.local_segments.set_inactive_segments(i_seg)

            if abs(dz) >= 1e2:
                self.info("diverging worker")
                self.wait_status_changed(status=csts.STATUS_FINISHED)
                diverging = True
                break

            if self.timing:
                # TODO: logging stuff
                pass

            # check stopping criterion
            if _check_convergence(self.local_segments, self.tol, ii,
                                  self.dz_opt, n_coordinates, self.strategy):
                if self.check_no_transitting_message():
                    status = self.wait_status_changed()
                    if status == csts.STATUS_STOP:
                        self.info("LGCD converged after {} iterations", ii + 1)
                        break

        else:
            self.info("Reach maximal iteration. Max of dz_opt is {}.",
                      abs(self.dz_opt).max())
            self.wait_status_changed(status=csts.STATUS_FINISHED)

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
                msg_fmt = csts.GLOBAL_OUTPUT_TAG + msg
                identity = self.n_jobs
            else:
                msg_fmt = csts.WORKER_OUTPUT_TAG + msg
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

        # compute sizes for the segments for LGCD
        n_atoms, n_channels, *atom_shape = self.D.shape

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

    def _process_messages(self, worker_status=csts.STATUS_RUNNING):

        mpi_status = MPI.Status()
        while MPI.COMM_WORLD.Iprobe(status=mpi_status):
            src = mpi_status.source
            tag = mpi_status.tag
            if tag == csts.TAG_UPDATE_BETA:
                if worker_status == csts.STATUS_PAUSED:
                    self.notify_worker_status(csts.TAG_RUNNING_WORKER,
                                              wait=True)
                    worker_status = csts.STATUS_RUNNING
            elif tag == csts.TAG_STOP:
                worker_status = csts.STATUS_STOP
            elif tag == csts.TAG_PAUSED_WORKER:
                self.n_paused_worker += 1
                assert self.n_paused_worker <= self.n_jobs
            elif tag == csts.TAG_RUNNING_WORKER:
                self.n_paused_worker -= 1
                assert self.n_paused_worker >= 0

            msg = np.empty(csts.SIZE_MSG, 'd')
            MPI.COMM_WORLD.Recv([msg, MPI.DOUBLE], source=src,
                                tag=tag)

            if tag == csts.TAG_UPDATE_BETA:
                self._message_update_beta(msg)

        if self.n_paused_worker == self.n_jobs:
            worker_status = csts.STATUS_STOP
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
            pt=(h0, w0), radius=self.atom_shape)
        n_changed_status = self.local_segments.set_active_segments(
            touched_segment)

        if flags.CHECK_ACTIVE_SEGMENTS and n_changed_status:
            self.local_segments.test_active_segments(self.dz_opt, self.tol)

    def notify_neighbors(self, msg, neighbors):
        assert self.rank in neighbors
        for i_neighbor in neighbors:
            if i_neighbor != self.rank:
                req = self.send_message(msg, csts.TAG_UPDATE_BETA, i_neighbor,
                                        wait=False)
                self.messages.append(req)

    def notify_worker_status(self, tag, i_worker=0, wait=False):
        # handle the messages from Worker0 to himself.
        if self.rank == 0 and i_worker == 0:
            if tag == csts.TAG_PAUSED_WORKER:
                self.n_paused_worker += 1
                assert self.n_paused_worker <= self.n_jobs
            elif tag == csts.TAG_RUNNING_WORKER:
                self.n_paused_worker -= 1
                assert self.n_paused_worker >= 0
            else:
                raise ValueError("Got tag {}".format(tag))
            return

        # Else send the message to the required destination
        msg = np.empty(csts.SIZE_MSG, 'd')
        self.send_message(msg, tag, i_worker, wait=wait)

    def wait_status_changed(self, status=csts.STATUS_PAUSED):
        self.notify_worker_status(csts.TAG_PAUSED_WORKER)
        self.debug("paused worker")

        # Wait for all sent message to be processed
        count = 0
        while status not in [csts.STATUS_RUNNING, csts.STATUS_STOP]:
            time.sleep(.001)
            status = self._process_messages(worker_status=status)
            if (count % 500) == 0:
                self.progress(self.n_paused_worker, max_ii=self.n_jobs,
                              unit="done workers")

        if self.rank == 0 and status == csts.STATUS_STOP:
            for i_worker in range(1, self.n_jobs):
                self.notify_worker_status(csts.TAG_STOP, i_worker, wait=True)
        elif status == csts.STATUS_RUNNING:
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
                  tag=csts.TAG_ROOT + rank)

        if debug:

            X_alpha = 0.25 * np.ones(X_shape)
            comm.Send([X_alpha.ravel(), MPI.DOUBLE], dest=0,
                      tag=csts.TAG_ROOT + rank)

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
                  tag=csts.TAG_ROOT + self.rank)
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
