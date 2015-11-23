#!/usr/bin/env python
from mpi4py import MPI
import numpy as np
from time import time
from toolbox.optim import _GradientDescent
from os import path

from toolbox.logger import Logger
log = Logger('MPI_DCP')


class DICOD(_GradientDescent):
    """MPI implementation of the distributed convolutional pursuit

    Parameters
    ----------
    pb: toolbox.optim._Problem
        convolutional coding problem
    n_jobs: int, optional (default: 1)
        Maximal number of process to solve this problem
    use_seg: int, optional (default: 1)
        If >1, further segment the updates and update
        the best coordinate over each semgnet cyclically
    hostfile: str, optional (default: None)
        Specify a hostfile for mpi launch, permit to use
        more than one computer
    logging: bool, optional (default: False)
        Enable the logging of the updates to allow printing a
        cost curve
    debug: int, optional (default: 0)
        verbosity level

    kwargs
    ------
    tol: float, default: 1e-10
    i_max: int, default: 1000
    t_max: int default: 40

    """

    def __init__(self, pb, n_jobs=1, use_seg=1, hostfile=None,
                 logging=False, debug=0, positive=False, **kwargs):
        log.set_level(max(3-debug, 1)*10)
        debug = max(debug-1, 0)
        super(DICOD, self).__init__(pb, debug=debug, **kwargs)
        self.debug = debug
        self.n_jobs = n_jobs
        self.hostfile = hostfile
        self.logging = 1 if logging else 0
        self.use_seg = use_seg
        self.positive = 1 if positive else 0
        if self.name == '_GD'+str(self.id):
            self.name = 'MPI_DCP' + str(self.n_jobs) + '_' + str(self.id)
        print('Logger', log.level)

    def fit(self, pb):
        self.pb = pb
        self._init_pool()
        self.end()

    def _init_pool(self):
        '''Launch n_jobs process to compute the convolutional
        coding solution with MPI process
        '''
        # Rename to call local variables
        self.K, self.d, self.S = self.pb.D.shape

        # Create a pool of worker
        t = time()
        mpi_info = MPI.Info.Create()
        if self.hostfile is not None:
            mpi_info.Set("add-hostfile", self.hostfile)
            mpi_info.Set("map_bynode", '1')
        c_prog = path.dirname(path.abspath(__file__))
        c_prog = path.join(c_prog, 'c_dicod', 'c_dicod')
        self.comm = MPI.COMM_SELF.Spawn(c_prog, maxprocs=self.n_jobs,
                                        info=mpi_info)
        log.debug('Created pool of worker in {:.4}s'.format(time()-t))

        # Send the job to process
        self.send_task()

    def send_task(self):
        self.K, self.d, self.S = self.pb.D.shape
        self.t_start = time()
        pb = self.pb
        K, d, S = self.K, self.d, self.S
        T = pb.x.shape[1]
        L = T-S+1

        # Share constants
        pb.compute_DD()
        alpha_k = np.sum(np.mean(pb.D*pb.D, axis=1), axis=1)
        alpha_k += (alpha_k == 0)

        self._broadcast_array(alpha_k)
        self._broadcast_array(pb.DD)
        self._broadcast_array(pb.D)

        # Send the constants of the algorithm
        N = np.array([float(d), float(K), float(S), float(T),
                      self.pb.lmbd, self.tol, float(self.t_max),
                      self.i_max/self.n_jobs, float(self.debug),
                      float(self.logging), float(self.use_seg),
                      float(self.positive)], 'd')
        self._broadcast_array(N)

        # Share the work between the processes
        sig = np.array(pb.x, dtype='d')
        L_proc = L//self.n_jobs + 1
        expect = []
        for i in range(self.n_jobs):
            end = min(T, (i+1)*L_proc+S-1)
            self.comm.Send([sig[:, i*L_proc:end].flatten(),
                            MPI.DOUBLE], i, tag=100+i)
            expect += [sig[0, i*L_proc], sig[-1, end-1]]
        self._confirm_array(expect)
        self.L, self.L_proc = L, L_proc

        # Wait end of initialisation
        self.comm.Barrier()
        self.t_init = time() - self.t_start
        self.t1 = time()
        log.info('End initialisation - {:.4}s'.format(self.t_init))

    def end(self):

        #reduce_pt
        self._gather()

        log.debug("DICOD - Clean end")
        self.comm.Barrier()
        self.comm.Disconnect()

        if type(self.t) == int:
            self.t = time()-self.t_start

    def _gather(self):
        K, L, L_proc = self.K, self.L, self.L_proc
        pt = np.empty((K, L), 'd')
        self.comm.Barrier()
        log.debug("End computation, gather result")
        self.t = time()-self.t_start

        for i in range(self.n_jobs):
            off = i*self.L_proc
            L_proc_i = min(off+L_proc, L)-off
            gpt = np.empty(K*L_proc_i, 'd')
            self.comm.Recv([gpt, MPI.DOUBLE], i, tag=200+i)
            pt[:, i*L_proc:(i+1)*L_proc] = gpt.reshape((K, -1))

        cost = np.empty(self.n_jobs, 'd')
        iterations = np.empty(self.n_jobs, 'i')
        times = np.empty(self.n_jobs, 'd')
        self.comm.Gather(None, [cost, MPI.DOUBLE],
                         root=MPI.ROOT)
        self.comm.Gather(None, [iterations, MPI.INT],
                         root=MPI.ROOT)
        self.comm.Gather(None, [times, MPI.DOUBLE],
                         root=MPI.ROOT)
        self.comm.Gather(None, [init_times, MPI.DOUBLE],
                         root=MPI.ROOT)
        self.t_init += max(init_times)
        self.cost = np.sum(cost)
        self.iteration = np.sum(iterations)
        self.runtime = times.max()
        log.debug("Iterations", iterations)
        log.debug("Times", times)
        t_end = time()
        self.pb.pt = pt
        log.info('End for {}'.format(self),
                 'iteration {}, time {:.4}s'.format(self.iteration, self.t))
        log.info('Total time: {:.4}s'.format(time()-self.t_start))
        log.info('Total time: {:.4}s'.format(self.runtime))
        self.runtime2 = time() - self.t_start

        if self.logging:
            self._log(iterations, t_end)

        self.comm.Barrier()

    def _log(self, iterations, t_end):
        self.comm.Barrier()
        pb, L = self.pb, self.L
        updates = []
        updates_t = []
        for i, it in enumerate(iterations):
            _log = np.empty(3*it)
            self.comm.Recv([_log, MPI.DOUBLE], i, tag=300+i)
            updates += [(round(_log[3*i]), _log[3*i+2]) for i in range(it)]
            updates_t += [_log[3*i+1] for i in range(it)]

        i0 = np.argsort(updates_t)
        next_log = 1
        pb.reset()
        t = np.min(updates_t)
        for it, i in enumerate(i0):
            if it+1 >= next_log:
                log.log_obj(name='cost'+str(self.id), obj=np.copy(pb.pt),
                            iteration=it+1, fun=pb.cost,
                            graph_cost=self.graph_cost, time=t)
                next_log = self.log_rate(it+1)
            j, du = updates[i]
            t = updates_t[i]+self.t_init
            pb.pt[j//L, j%L] += du
        log.log_obj(name='cost'+str(self.id), obj=np.copy(pb.pt),
                    iteration=it, fun=pb.cost,
                    graph_cost=self.graph_cost, time=t_end-self.t_start)
        self.log_update = (updates_t, updates)

    def gather_AB(self):
        K, S, d = self.K, self.S, self.d
        A = np.empty(K*K*S, 'd')
        B = np.empty(d*K*S, 'd')
        self.comm.Barrier()
        log.debug("End computation, gather result")

        self.comm.Reduce(None, [A, MPI.DOUBLE], op=MPI.SUM,
                         root=MPI.ROOT)
        self.comm.Reduce(None, [B, MPI.DOUBLE], op=MPI.SUM,
                         root=MPI.ROOT)

        iterations = np.empty(self.n_jobs, 'i')
        self.comm.Gather(None, [iterations, MPI.INT],
                         root=MPI.ROOT)
        self.iteration = np.sum(iterations)
        log.debug("Iterations", iterations)

        self.comm.Barrier()
        self.gather()
        return A, B

    def _broadcast_array(self, arr):
        arr = np.array(arr).astype('d')
        T = np.prod(arr.shape)
        N = np.array(T, 'i')
        self.comm.Bcast([N, MPI.INT], root=MPI.ROOT)
        self.comm.Bcast([arr.flatten(), MPI.DOUBLE], root=MPI.ROOT)

    def _confirm_array(self, expect):
        '''Aux function to confirm that we passed the correct array
        '''
        expect = np.array(expect)
        gathering = np.empty(expect.shape, 'd')
        self.comm.Gather(None, [gathering, MPI.DOUBLE],
                         root=MPI.ROOT)
        assert (np.allclose(expect, gathering)), (expect, gathering,
                                                  'Fail to transmit array')

    def p_update(self):
        return 0

    def _stop(self, dz):
        return True
