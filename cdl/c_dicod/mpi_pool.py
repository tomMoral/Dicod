from mpi4py import MPI
from os import path
from time import time, sleep
import numpy as np
import threading

DEBUG = True

MNG_STOP = 0
MNG_RESIZE_SERVER = 1
MNG_RESIZE_CLIENT = 2

# // CONTROL MSG TAG
TAG_MNG_MSG = 0
TAG_PORT_MSG = 1

RUN = 0
BROKEN = 1
TERMINATE = 2


_local = threading.local()


def get_reusable_pool(n_jobs=None, hostfile=None):
    _pool = getattr(_local, '_pool', None)
    t = time()
    if _pool is None:
        _local._pool = _pool = MPI_Pool(n_jobs=n_jobs, hostfile=hostfile)
    elif _pool._state != RUN or n_jobs != _pool.n_jobs:
        if DEBUG:
            print("DEBUG   - Create a new pool as the previous one"
                  " was in state {}".format(_pool._state))
        _pool.terminate()
        _local._pool = None
        return get_reusable_pool(n_jobs=n_jobs, hostfile=hostfile)
    elif n_jobs == _pool.n_jobs:
        return _pool
    else:
        _pool.resize(n_jobs)
        print('Created pool of worker in {:.4}s'.format(time()-t))
        return _pool
    print('Created pool of worker in {:.4}s'.format(time()-t))
    print(_pool.n_jobs, '/', n_jobs)
    return _pool


class MPI_Pool(object):
    """docstring for MPI_Pool"""
    def __init__(self, n_jobs=1, hostfile=None):
        super(MPI_Pool, self).__init__()
        self.n_jobs = n_jobs
        self.hostfile = hostfile
        c_prog = path.dirname(path.abspath(__file__))
        self.c_prog = path.join(c_prog, 'start_worker')
        self._init_pool()
        self._state = RUN

    def _init_pool(self):
        '''Launch n_jobs process to compute the convolutional
        coding solution with MPI process
        '''

        # Create a pool of worker
        mpi_info = MPI.Info.Create()
        if self.hostfile is not None:
            mpi_info.Set("add-hostfile", self.hostfile)
            # mpi_info.Set("map_bynode", '1')
        self.comm = MPI.COMM_SELF.Spawn(
            self.c_prog, maxprocs=self.n_jobs,
            info=mpi_info)
        self.comm.Barrier()
        print("Pool initialized")

    def resize(self, n_jobs):
        '''TODO: Robustify
        '''
        t = time()
        mpi_info = MPI.Info.Create()
        if self.hostfile is not None:
            mpi_info.Set("add-hostfile", self.hostfile)
            mpi_info.Set("map_bynode", '1')
        comm2 = MPI.COMM_SELF.Spawn(
            self.c_prog, maxprocs=n_jobs-self.n_jobs,
            info=mpi_info)
        i0 = np.random.randint(1024)
        msg = np.array([MNG_RESIZE_SERVER, i0, 0, 0]).astype('i')
        self.mng_bcast(msg)
        portname = np.empty(MPI.MAX_PORT_NAME).astype('c')
        self.comm.Recv([portname, MPI.CHAR], 0, TAG_PORT_MSG)
        print(portname)
        msg[0] = MNG_RESIZE_CLIENT
        comm2.Barrier()
        self.mng_bcast(msg, comm2)
        for i in range(comm2.remote_size):
            comm2.Send([portname, MPI.CHAR], 0, TAG_PORT_MSG)
        print('Resized pool of worker in {:.4}s'.format(time()-t))

        msg[0] = MNG_STOP
        self.mng_bcast(msg, comm2)
        comm2.Disconnect()

    def mng_bcast(self, msg, comm=None):
        print("mng_bcast: ", msg)
        if comm is None:
            comm = self.comm
        for i in range(comm.remote_size, 0, -1):
            comm.Send([msg, MPI.INT], i - 1, TAG_MNG_MSG)

    def terminate(self):
        msg = np.array([MNG_STOP] * 4).astype('i')
        self.mng_bcast(msg)
        self.comm.Disconnect()


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('Test MPI Rpool')
    parser.add_argument('--njobs', type=int, default=1,
                        help='# of spawned processes')
    parser.add_argument('--hostfile', type=str, default=None,
                        help='# of spawned processes')
    args = parser.parse_args()
    p = get_reusable_pool(n_jobs=args.njobs, hostfile=args.hostfile)
    sleep(1)
    p = get_reusable_pool(n_jobs=args.njobs+1)
    p.terminate()
