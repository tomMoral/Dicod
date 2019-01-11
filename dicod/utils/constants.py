

# Inter-process communication constants
TAG_STOP = 0
TAG_UPDATE_BETA = 1
TAG_PAUSED_WORKER = 2
TAG_RUNNING_WORKER = 3
TAG_INIT_DONE = 4

TAG_ROOT = 4242


# Worker control flow
TAG_WORKER_STOP = 16
TAG_WORKER_RUN_DICOD = 17


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
