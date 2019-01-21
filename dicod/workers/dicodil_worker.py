from dicod.utils import constants
from dicod.workers.dicod_worker import DICODWorker
from dicod.utils.mpi import wait_message


def dicodil_worker():
    dicod = DICODWorker(backend='mpi')
    dicod.recv_task()

    tag = wait_message()
    while tag != constants.TAG_DICODIL_STOP:
        if tag == constants.TAG_DICODIL_UPDATE_Z:
            n_coord_updated, runtime = dicod.compute_z_hat()
        if tag == constants.TAG_DICODIL_UPDATE_D:
            dicod.get_D()
        if tag == constants.TAG_DICODIL_GET_COST:
            dicod.return_cost()
        if tag == constants.TAG_DICODIL_GET_Z_HAT:
            dicod.return_z_hat()
        if tag == constants.TAG_DICODIL_GET_Z_NNZ:
            dicod.return_z_nnz()
        if tag == constants.TAG_DICODIL_GET_SUFFICIENT_STAT:
            dicod.return_sufficient_statistics()
        tag = wait_message()