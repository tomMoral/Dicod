"""Constants for interprocess communication

Author : tommoral <thomas.moreau@inria.fr>
"""

# Inter-process communication constants
TAG_ROOT = 4242

# Worker control flow
TAG_WORKER_STOP = 0
TAG_WORKER_RUN_DICOD = 1
TAG_WORKER_RUN_DICODIL = 2

# DICOD worker control messages
TAG_DICOD_STOP = 8
TAG_DICOD_UPDATE_BETA = 9
TAG_DICOD_PAUSED_WORKER = 10
TAG_DICOD_RUNNING_WORKER = 11
TAG_DICOD_INIT_DONE = 12

# DICODIL worker control tags
TAG_DICODIL_STOP = 16
TAG_DICODIL_UPDATE_Z = 17
TAG_DICODIL_UPDATE_D = 18
TAG_DICODIL_GET_COST = 19
TAG_DICODIL_GET_Z_HAT = 20
TAG_DICODIL_GET_Z_NNZ = 21
TAG_DICODIL_GET_SUFFICIENT_STAT = 22


# inter-process message size
SIZE_MSG = 4


# Output control
GLOBAL_OUTPUT_TAG = "\r[DICOD-{}:{}] "
WORKER_OUTPUT_TAG = "\r[DICOD:Worker-{:<3}:{}] "


# Worker status
STATUS_STOP = 0
STATUS_PAUSED = 1
STATUS_RUNNING = 2
STATUS_FINISHED = 4
