REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 128
EMBEDDING_DIM = 128
NODE_PADDING_SIZE = 360  # the number of nodes will be padded to this value
K_SIZE = 20  # the number of neighboring nodes

USE_GPU = False  # do you want to collect training data using GPUs
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
NUM_GPU = 1
NUM_META_AGENT = 32
LR = 1e-5
GAMMA = 0.99
DECAY_STEP = 256  # not use
SUMMARY_WINDOW = 5
LOAD_MODEL = False # do you want to load the model trained before
SAVE_IMG_GAP = 100
N_ROBOTS = 2
USE_ROBOT_ATTENTION = 1
USE_SEQUENCE_POLICY = True
USE_K_FLAGS = False

# ICRA
USE_C = 1
ALLOW_STAY = True

# TCDS
INTERSECTFRONTIER = False
USE_COLLISION = False
USE_TRANS = True

# USELESS
USE_GUIDEPOST = True

if USE_ROBOT_ATTENTION:
    USE_K_FLAGS = True

INPUT_DIM = 3 + int(USE_C) + int(USE_GUIDEPOST)
IOU_LAMBDA = 0.05
train_mode = True

extra_info = ''
FOLDER_NAME = 'MAAEs{}l{}r{}K{:d}A{:d}S{:d}ST{:d}C{:d}CO{:d}trans{:d}{}'.format(
    70,
    6,
    N_ROBOTS,
    USE_K_FLAGS,
    USE_ROBOT_ATTENTION,
    USE_SEQUENCE_POLICY,
    ALLOW_STAY,
    USE_C,
    USE_COLLISION,
    USE_TRANS,
    extra_info
)
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
