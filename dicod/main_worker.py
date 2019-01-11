"""Main script for MPI workers

Author : tommoral <thomas.moreau@inria.fr>
"""
from mpi4py import MPI
import numpy as np
import time

from dicod.utils import constants
from dicod._dicod_worker import DICODWorker


def _wait_message():
    comm = MPI.Comm.Get_parent()
    mpi_status = MPI.Status()
    while not comm.Iprobe(status=mpi_status):
        time.sleep(.001)

    # Receive a message
    msg = np.empty(1, dtype='i')
    src = mpi_status.source
    tag = mpi_status.tag
    comm.Recv([msg, MPI.INT], source=src, tag=tag)

    assert tag == msg[0], "tag and msg should be equal"

    return tag


def _shutdown_mpi():
    comm = MPI.Comm.Get_parent()
    comm.Barrier()
    comm.Disconnect()


def sync_workers():
    comm = MPI.Comm.Get_parent()
    comm.Barrier()


def main():
    sync_workers()
    tag = _wait_message()
    while tag != constants.TAG_WORKER_STOP:
        if tag == constants.TAG_WORKER_RUN_DICOD:
            dicod = DICODWorker(backend='mpi')
            dicod.run()
        tag = _wait_message()

    _shutdown_mpi()


if __name__ == "__main__":
    main()
