"""Helper functions for MPI communication

Author : tommoral <thomas.moreau@inria.fr>
"""
import time
import numpy as np
from mpi4py import MPI


def broadcast_array(comm, arr):
    arr_shape = np.array(arr.shape, dtype='i')
    arr = np.array(arr.flatten(), dtype='d')
    N = np.array([arr.shape[0], len(arr_shape)], dtype='i')

    # Send the data and shape of the numpy array
    comm.Bcast([N, MPI.INT], root=MPI.ROOT)
    comm.Bcast([arr_shape, MPI.INT], root=MPI.ROOT)
    comm.Bcast([arr, MPI.DOUBLE], root=MPI.ROOT)


def recv_broadcasted_array(comm):
    N = np.empty(2, dtype='i')
    comm.Bcast([N, MPI.INT], root=0)

    arr_shape = np.empty(N[1], dtype='i')
    comm.Bcast([arr_shape, MPI.INT], root=0)

    arr = np.empty(N[0], dtype='d')
    comm.Bcast([arr, MPI.DOUBLE], root=0)
    return arr.reshape(arr_shape)


def wait_message():
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


def shutdown_mpi():
    comm = MPI.Comm.Get_parent()
    comm.Barrier()
    comm.Disconnect()


def sync_workers():
    comm = MPI.Comm.Get_parent()
    comm.Barrier()
