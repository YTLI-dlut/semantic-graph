EMBEDDING_DIM = 128
K_SIZE = 20  # the number of neighbors

USE_GPU = True  # do you want to use GPUS?
NUM_GPU = 2  # the number of GPUs
NUM_META_AGENT = 16  # the number of processes


NUM_TEST = 200
NUM_RUN = 1
SAVE_GIFS = 1  # do you want to save GIFs
SAVE_TRAJECTORY = 0  # do you want to save per-step metrics
SAVE_LENGTH = 1  # do you want to save per-episode metrics
N_ROBOTS = 2
USE_ROBOT_ATTENTION = 1
USE_SEQUENCE_POLICY = 1
USE_K_FLAGS = 1
USE_C = False
USE_TRANS = True
ALLOW_STAY = False
USE_COLLISION = False
INTERSECTFRONTIER = True
USE_GUIDEPOST = True
INPUT_DIM = 3 + int(USE_C) + int(USE_GUIDEPOST)
if USE_ROBOT_ATTENTION:
    USE_K_FLAGS = True

extra_info = '-intersectR'
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

report = False
model_path = f'model/{FOLDER_NAME}'
gifs_path = f'results/{FOLDER_NAME}/gifs' if not report else f'report/results/{FOLDER_NAME}/gifs'
trajectory_path = f'results/trajectory'
length_path = f'results/length'