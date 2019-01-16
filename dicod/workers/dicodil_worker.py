from dicod.utils import constants
from dicod.workers.dicod_worker import DICODWorker
from dicod.utils.mpi import wait_message


def dicodil_worker():
    dicod = DICODWorker(backend='mpi')
    dicod.receive_task()

    tag = wait_message()
    while tag != constants.TAG_DICODIL_STOP_DICODIL:
        if tag == constants.TAG_DICODIL_UPDATE_Z:
            n_coord_updated, runtime = dicod.compute_z_hat()
        if tag == constants.TAG_DICODIL_UPDATE_D:
            dicod.D = dicod.get_D()
        if tag == constants.TAG_DICODIL_GET_COST:
            dicod._return_cost_mpi()
        if tag == constants.TAG_DICODIL_GET_Z_HAT:
            dicod._return_signal_mpi()
        if tag == constants.TAG_DICODIL_GET_SUFFICIENT_STAT:
            dicod._return_sufficient_statistics_mpi()
        tag = wait_message()
