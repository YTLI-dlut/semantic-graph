import time
import os

# MAME Parameter Configuration (Single Agent Refactor)

REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 1000
BATCH_SIZE = 32
EMBEDDING_DIM = 128

NODE_PADDING_SIZE = 500  
# Action Space
K_SIZE = 15  
NUM_ANGLES_BIN = 36
NUM_HEADING_CANDIDATES = 3

USE_GPU = True  
USE_GPU_GLOBAL = True
NUM_GPU = 2
NUM_META_AGENT = 16 

LR = 3e-4
GAMMA = 1
DECAY_STEP = 256
SUMMARY_WINDOW = 5
LOAD_MODEL = False
LOAD_MODEL_PATH = '/home/iiau/createGraph_ws/model_save/semantic_20260202_130039/checkpoint.pth'
SAVE_IMG_GAP = 100

# Training Control
RANDOM_EXPLORE_EPOCHS = 30
USE_ASTAR_EXPLORATION = True # Use A* based frontier exploration in early epochs
MAX_EPISODE_STEPS = 350
STAY_STILL_PENALTY = -0.500

# Single Agent Settings
N_ROBOTS = 1 

# Features
ALLOW_STAY = True
USE_GUIDEPOST = True
USE_UNCONFIRMED_VECTOR = True

#node_coords, node_utility, guidepost, entropy_feats, vector_feats 2 1 1 1 2
INPUT_DIM = 2 + 1 + int(USE_GUIDEPOST) + 1 + 2 * int(USE_UNCONFIRMED_VECTOR)
IOU_LAMBDA = 0.05
train_mode = True

# Entropy Boost Parameters
ENTROPY_UNCONFIRMED_BOOST = 2.0
ENTROPY_BOOST_RADIUS = 30  # pixels

# Reward Weights
REWARD_CONFIRM = 10.0
REWARD_EXPLORE_CELL = 0.5
REWARD_NEW_SEEN = 5.0
REWARD_DONE = 100.0
REWARD_STEP_PENALTY = -0.1

# SAC Parameters
TAU = 0.005
ALPHA = 0.02
TARGET_ENTROPY_SCALE = 0.2

# GPU Load Balancing
ENABLE_GPU_BALANCING = True
GPU_MEMORY_THRESHOLD = 0.85
GPU_IMBALANCE_THRESHOLD = 0.3
GPU_MONITOR_INTERVAL = 100




# Paths
MODEL_SAVE_DIR = 'model_save'
LOG_DIR = 'log'
GIFS_DIR = 'gifs'

# Dataset Curriculum Configuration
DATASET_EASY_EPOCHS = 8000
DATASET_MEDIUM_START_EPOCH = 8000
DATASET_MEDIUM_END_EPOCH = 17000
DATASET_HARD_START_EPOCH = 17000

DATASET_EASY_PATH = 'generated_data_easy'
DATASET_MEDIUM_PATH = 'generated_data_medium'
DATASET_HARD_PATH = 'generated_data_hard'

FOLDER_NAME = f'semantic_{time.strftime("%Y%m%d_%H%M%S")}'
model_path = f'{MODEL_SAVE_DIR}/{FOLDER_NAME}'
train_path = f'{LOG_DIR}/{FOLDER_NAME}'
gifs_path = f'{GIFS_DIR}/{FOLDER_NAME}'

def init_dirs():
    # Create directories if they don't exist
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(gifs_path):
        os.makedirs(gifs_path)
