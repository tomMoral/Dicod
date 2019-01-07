

# Inter-process communication constants
TAG_STOP = 0
TAG_UPDATE_BETA = 1
TAG_PAUSED_WORKER = 2
TAG_RUNNING_WORKER = 4
TAG_ROOT = 4242


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
