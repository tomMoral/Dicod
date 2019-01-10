"""Start and shutdown MPI workers

Author : tommoral <thomas.moreau@inria.fr>
"""
import os
import sys
import numpy as np
from mpi4py import MPI
import multiprocessing as mp


from .utils import constants
from .utils import debug_flags as flags

# global worker communicator
_n_workers = None
_worker_comm = None


# Constants to start interactive workers
INTERACTIVE_EXEC = "xterm"
INTERACTIVE_ARGS = ["-fa", "Monospace", "-fs", "12", "-e", "ipython", "-i"]


def get_reusable_workers(n_jobs=4, hostfile=None):

    global _worker_comm, _n_workers
    if _worker_comm is None:
        _n_workers = n_jobs
        _worker_comm = _spawn_workers(n_jobs, hostfile)
        mp.util.Finalize(None, shutdown_reusable_workers, exitpriority=20)
    else:
        assert _n_workers == n_jobs, "You should not require different size"

    return _worker_comm


def send_command_to_reusable_workers(tag):
    global _worker_comm, _n_workers
    msg = np.empty(1, dtype='i')
    msg[0] = tag
    for i_worker in range(_n_workers):
        _worker_comm.Send([msg, MPI.INT], dest=i_worker, tag=tag)


def shutdown_reusable_workers():
    global _worker_comm
    send_command_to_reusable_workers(constants.TAG_WORKER_STOP)
    _worker_comm.Barrier()
    print("Clean shutdown")


def _spawn_workers(n_jobs, hostfile):
    info = MPI.Info.Create()
    # info.Set("map_bynode", '1')
    if hostfile and os.path.exists(hostfile):
        info.Set("hostfile", hostfile)
    script_name = os.path.join(os.path.dirname(__file__),
                               "main_worker.py")
    if flags.INTERACTIVE_PROCESSES:
        comm = MPI.COMM_SELF.Spawn(
            INTERACTIVE_EXEC, args=INTERACTIVE_ARGS + [script_name],
            maxprocs=n_jobs, info=info)

    else:
        comm = MPI.COMM_SELF.Spawn(sys.executable, args=[script_name],
                                   maxprocs=n_jobs, info=info)
    return comm