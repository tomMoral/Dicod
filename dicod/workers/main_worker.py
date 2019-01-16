"""Main script for MPI workers

Author : tommoral <thomas.moreau@inria.fr>
"""
from dicod.utils import constants
from dicod.workers.dicod_worker import DICODWorker
from dicod.utils.mpi import wait_message, sync_workers, shutdown_mpi


def main():
    sync_workers()
    tag = wait_message()
    while tag != constants.TAG_WORKER_STOP:
        if tag == constants.TAG_WORKER_RUN_DICOD:
            dicod = DICODWorker(backend='mpi')
            dicod.run()
        tag = wait_message()

    shutdown_mpi()


if __name__ == "__main__":
    main()
