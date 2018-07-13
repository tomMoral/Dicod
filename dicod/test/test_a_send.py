import pytest
import numpy as np
from os import path
from mpi4py import MPI

from faulthandler import dump_traceback_later
from faulthandler import cancel_dump_traceback_later

from dicod import c_dicod


@pytest.yield_fixture
def exit_on_deadlock():
    dump_traceback_later(timeout=30, exit=True)
    yield
    cancel_dump_traceback_later()


def test_mpi_send(exit_on_deadlock):

    c_prog = path.dirname(path.abspath(c_dicod.__file__))
    c_prog = path.join(c_prog, "_test_send")

    msg = np.array([42] * 1).astype('i')
    print("Spawning")
    mpi_info = MPI.Info.Create()
    mpi_info.Set("add-hostfile", "hostfile")
    comm = MPI.COMM_SELF.Spawn(c_prog, maxprocs=1, info=mpi_info)
    print("Barriere")
    comm.Barrier()
    comm.Barrier()
    comm.Barrier()
    print("Sending", comm)
    comm.Send([msg, MPI.INT], dest=0, tag=100)
    print("sent")
    comm.Barrier()
