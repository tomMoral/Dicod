from dicod.utils import constants
from dicod.workers.dicod_worker import DICODWorker
from dicod.utils.mpi import wait_message


def dicodil_worker():
    dicod = DICODWorker(backend='mpi')
    dicod.receive_task()

    tag = wait_message()
    while tag != constants.TAG_WORKER_STOP:
        if tag == constants.TAG_WORKER_RUN_DICOD:
            dicod = DICODWorker(backend='mpi')
            dicod.run()
        tag = wait_message()
