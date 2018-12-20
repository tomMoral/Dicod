# Authors: Thomas Moreau <thomas.moreau@inria.fr>

import numpy as np
from mpi4py import MPI

from scipy.signal import fftconvolve

from faulthandler import dump_traceback_later

ALGO_GS = 0
ALGO_RANDOM = 1


TAG_ROOT = 4242


class DICODWorker():
    def __init__(self):
        self._comm = MPI.Comm.Get_parent()

        self.rank = self._comm.Get_rank()
        self.n_jobs = self._comm.Get_size()

        print("{} - {}/{} started".format(
            MPI.Get_processor_name(), self.rank, self.n_jobs))
        self._comm.Barrier()

    def shutdown(self):
        print("Worker{} finished".format(self.rank))
        print("ciao!")
        self._comm.Barrier()
        self._comm.Disconnect()

    def _receive_array(self):
        N = np.empty(2, dtype='i')
        self._comm.Bcast([N, MPI.INT], root=0)

        arr_shape = np.empty(N[1], dtype='i')
        self._comm.Bcast([arr_shape, MPI.INT], root=0)
        
        arr = np.empty(N[0], dtype='d')
        self._comm.Bcast([arr, MPI.DOUBLE], root=0)
        return arr.reshape(arr_shape)

    def _receive_task(self):
        # receive constants
        self.alpha_k = self._receive_array()
        self.DD = self._receive_array()
        self.D = self._receive_array()

        constants = self._receive_array()

        (width, height, self.w_world, self.lmbd, self.tol,
         self.timeout, self.max_iter, self.verbose, self.logging, self.n_seg,
         self.positive, self.algorithm, self.patience) = constants

        # convert back some quantities to integers
        height, width = int(height), int(width)
        self.w_world = int(self.w_world)

        # Infer some topological information
        self.h_world = self.n_jobs // self.w_world
        self.w_rank = self.rank % self.w_world
        self.h_rank = self.rank // self.w_world

        if self.rank == 0 and self.verbose > 5:
            print("DEBUG:jobs - Start with algorithm : {}"
                  .format("Random" if self.algorithm else "Gauss-Southwell"))

        # Compute the size of the signal for this worker
        _, n_channels, height_atom, width_atom = self.D.shape
        height_valid = height - height_atom + 1
        width_valid = width - width_atom + 1
        height_worker = height_valid // self.h_world
        width_worker = width_valid // self.w_world
        height_offset = self.h_rank * height_worker
        width_offset = self.w_rank * width_worker
        if self.h_rank == self.h_world - 1:
            height_worker = height_valid - height_offset

        if self.w_rank == self.w_world - 1:
            width_worker = width_valid - width_offset

        self.height_worker, self.width_worker = height_worker, width_worker

        X_shape = (n_channels, height_worker + height_atom - 1,
                   width_worker + width_atom - 1)
        X_shape = [int(v) for v in X_shape]
        self.X_worker = np.empty(X_shape, dtype='d')
        self._comm.Recv([self.X_worker.ravel(), MPI.DOUBLE], source=0,
                        tag=TAG_ROOT + self.rank)

        import matplotlib.pyplot as plt
        plt.imshow(self.X_worker.swapaxes(0, 2))
        plt.savefig("X_worker{}.png".format(self.rank))

        # Confirm the received arrays
        self._comm.Gather([self.X_worker.ravel()[[0, -1]], MPI.DOUBLE], None,
                          root=0)
        self._comm.Barrier()

    def _init_beta(self, X):
        # Init beta with the gradient in the current point 0
        beta = np.sum(
            [[fftconvolve(dkp, res_p, mode='valid')
              for dkp, res_p in zip(dk, -X)]
             for dk in self.D[:, :, ::-1, ::-1]], axis=1)

        dz_opt = np.maximum(-beta - self.lmbd, 0) / self.alpha_k
        return beta, dz_opt

    def _lgcd(self):
        n_atoms, n_channels, height_atom, width_atom = self.D.shape
        height_worker, width_worker = self.height_worker, self.width_worker
        n_coordinates = n_atoms * height_worker * width_worker
        z_hat = np.zeros((n_atoms, height_worker, width_worker), dtype='d')

        beta, dz_opt = self._init_beta(self.X_worker)

        assert beta.shape == z_hat.shape, (beta.shape, z_hat.shape)
        assert dz_opt.shape == z_hat.shape, (dz_opt.shape, z_hat.shape)

        if self.n_seg == 0:
            # use the default value for n_seg, ie 2x the size of D
            height_n_seg = max(height_worker // (2 * height_atom), 1)
            width_n_seg = max(width_worker // (2 * width_atom), 1)
        else:
            height_n_seg = width_n_seg = self.n_seg
        n_seg = width_n_seg * height_n_seg
        height_seg = (height_worker // height_n_seg) + 1
        width_seg = (width_worker // width_n_seg) + 1

        accumulator = n_seg
        active_segs = np.array([True] * n_seg)
        i_seg = 0
        seg_bounds = [[0, height_seg], [0, width_seg]]
        k0, h0, w0 = 0, -1, -1
        for ii in range(int(self.max_iter)):

            k0, h0, w0, dz = self._select_coordinate(
                dz_opt, active_segs[i_seg], seg_bounds)
            if self.algorithm == ALGO_RANDOM:
                # accumulate on all coordinates from the stopping criterion
                if ii % n_coordinates == 0:
                    accumulator = 0
                accumulator += abs(dz)

            # Update the selected coordinate and beta, only if the update is
            # greater than the convergence tolerance.
            if abs(dz) > self.tol:
                # update the selected coordinate
                z_hat[k0, h0, w0] += dz

                # update beta
                beta, dz_opt, accumulator, active_segs = self._update_beta(
                    beta, dz_opt, accumulator, active_segs, z_hat,
                    dz, k0, h0, w0, seg_bounds, i_seg, width_n_seg)

            elif active_segs[i_seg]:
                accumulator -= 1
                active_segs[i_seg] = False

            if self.logging:
                # TODO: logging stuff
                pass

            # check stopping criterion
            if self.algorithm == ALGO_GS:
                if accumulator == 0:
                    if self.verbose > 10:
                        print('[{}] {} iterations'.format(self.name, ii + 1))
                    break
            else:
                # only check at the last coordinate
                if (ii + 1) % n_coordinates == 0 and accumulator <= self.tol:
                    if self.verbose > 10:
                        print('[{}] {} iterations'.format(self.name, ii + 1))
                    break

            # increment to next segment
            i_seg += 1
            seg_bounds[1][0] += width_seg
            seg_bounds[1][1] += width_seg

            if seg_bounds[1][0] >= width_worker:
                # Got to the begining of the next line
                seg_bounds[1] = [0, width_seg]
                seg_bounds[0][0] += height_seg
                seg_bounds[0][1] += height_seg

                if seg_bounds[0][0] >= height_worker:
                    # reset to first segment
                    i_seg = 0
                    seg_bounds = [[0, height_seg], [0, width_seg]]

        else:
            if self.verbose > 10:
                print('[{}] did not converge'.format(self.name))

        # Now you can gather z_hat

    def _select_coordinate(self, dz_opt, active_seg, seg_bounds):
        # Pick a coordinate to update
        if self.algorithm == ALGO_RANDOM:
            n_atoms, height_valid, width_valid = dz_opt.shape
            k0 = np.random.randint(n_atoms)
            h0 = np.random.randint(height_valid)
            w0 = np.random.randint(width_valid)
            dz = dz_opt[k0, h0, w0]

        elif self.algorithm == ALGO_GS:
            # if dZs[i_seg] > tol:
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
            raise ValueError("'The coordinate selection method should be in "
                             "{'greedy' | 'random' | 'cyclic'}. Got '%s'."
                             % (self.algorithm, ))
        return k0, h0, w0, dz

    def _update_beta(self, beta, dz_opt, accumulator, active_segs, z_hat,
                     dz, k0, h0, w0, seg_bounds, i_seg, width_n_seg):
        n_atoms, height_valid, width_valid = beta.shape
        n_atoms, n_channels, height_atom, width_atom = self.D.shape

        # define the bounds for the beta update
        start_height_up = max(0, h0 - height_atom + 1)
        end_height_up = min(h0 + height_atom, self.height_worker)
        start_width_up = max(0, w0 - width_atom + 1)
        end_width_up = min(w0 + width_atom, self.width_worker)
        update_slice = (slice(None), slice(start_height_up, end_height_up),
                        slice(start_width_up, end_width_up))

        # update beta
        beta_i0 = beta[k0, h0, w0]
        update_height = end_height_up - start_height_up
        offset_height = max(0, height_atom - h0 - 1)
        update_width = end_width_up - start_width_up
        offset_width = max(0, width_atom - w0 - 1)
        beta[update_slice] += (
            self.DD[:, k0, offset_height:offset_height + update_height,
                    offset_width:offset_width + update_width] * dz
        )
        beta[k0, h0, w0] = beta_i0

        # update dz_opt
        tmp = np.maximum(-beta[update_slice] - self.lmbd, 0) / self.alpha_k
        dz_opt[update_slice] = tmp - z_hat[update_slice]
        dz_opt[k0, h0, w0] = 0

        # wake up greedy updates in neighboring segments if beta was updated
        # outside the segment
        start_height_seg, end_height_seg = seg_bounds[0]
        start_width_seg, end_width_seg = seg_bounds[1]

        if start_height_up < start_height_seg:
            # There are segments above the current one and some are impacted
            # by the current update
            i_seg_above = i_seg - width_n_seg
            if start_height_seg > 0:
                # Above segment
                accumulator += not active_segs[i_seg_above]
                active_segs[i_seg_above] = True
            elif self.h_rank > 0:
                # Send message to above neighbor (rank - w_world)
                pass

            if start_width_up < start_width_seg:
                # Impact the Top-left corner
                if start_width_seg > 0 and start_height_seg > 0:
                    accumulator += not active_segs[i_seg_above - 1]
                    active_segs[i_seg_above - 1] = True
                elif (start_height_seg == 0 and self.h_rank > 0
                        and self.w_rank > 0):
                    # Send message to above-left neighbor (rank - w_world - 1)
                    pass
            if end_width_up > end_width_seg:
                # Impact the top-right corner
                if end_width_seg < self.width_worker and start_height_seg > 0:
                    accumulator += not active_segs[i_seg_above + 1]
                    active_segs[i_seg_above + 1] = True
                elif (start_height_seg == 0 and self.h_rank > 0
                        and self.w_rank < self.w_world):
                    # Send message to above-right neighbor (rank - w_world + 1)
                    pass

        if start_width_up < start_width_seg:
            # Impact the left neighbor
                if start_width_seg > 0:
                    accumulator += not active_segs[i_seg - 1]
                    active_segs[i_seg - 1] = True
                elif self.w_rank > 0:
                    # Send message to left neighbor (rank - 1)
                    pass
        if end_width_up > end_width_seg:
            # Impact the right neighbor
                if end_width_seg < self.width_worker:
                    accumulator += not active_segs[i_seg + 1]
                    active_segs[i_seg + 1] = True
                elif self.w_rank < self.w_world:
                    # Send message to right neighbor (rank + 1)
                    pass

        if end_height_up > end_height_seg:
            # There are segments bellow the current one and some are impacted
            # by the current update
            i_seg_bellow = i_seg + width_n_seg
            if end_height_seg < self.height_worker:
                # Above segment
                accumulator += not active_segs[i_seg_bellow]
                active_segs[i_seg_bellow] = True
            elif self.h_rank < self.h_world:
                # Send message to bellow neighbor (rank + w_world)
                pass

            if start_width_up < start_width_seg:
                # Impact the Top-left corner
                if start_width_seg > 0 and end_height_seg < self.height_worker:
                    accumulator += not active_segs[i_seg_bellow - 1]
                    active_segs[i_seg_bellow - 1] = True
                elif (end_width_seg >= self.height_worker
                        and self.h_rank < self.h_world
                        and self.w_rank > 0):
                    # Send message to above-left neighbor (rank - w_world - 1)
                    pass
            if end_width_up > end_width_seg:
                # Impact the top-right corner
                if (end_width_seg < self.width_worker
                        and end_height_seg < self.height_worker):
                    accumulator += not active_segs[i_seg_bellow + 1]
                    active_segs[i_seg_bellow + 1] = True
                elif (end_width_seg >= self.height_worker
                        and self.h_rank < self.h_world
                        and self.w_rank < self.w_world):
                    # Send message to above-right neighbor (rank - w_world + 1)
                    pass

        return beta, dz_opt, accumulator, active_segs


if __name__ == "__main__":
    worker = DICODWorker()
    worker._receive_task()
    print("youhou")
    worker._lgcd()
    worker.shutdown()
