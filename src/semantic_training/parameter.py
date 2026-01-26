import time
import os

# MAME Parameter Configuration (Single Agent Refactor)

REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 64
EMBEDDING_DIM = 128

NODE_PADDING_SIZE = 500  
K_SIZE = 20  

USE_GPU = True  
USE_GPU_GLOBAL = True
NUM_GPU = 2
NUM_META_AGENT = 12 

LR = 1e-5
GAMMA = 0.99
DECAY_STEP = 256
SUMMARY_WINDOW = 5
LOAD_MODEL = False 
SAVE_IMG_GAP = 100

# Training Control
RANDOM_EXPLORE_EPOCHS = 50
USE_ASTAR_EXPLORATION = True # Use A* based frontier exploration in early epochs
MAX_EPISODE_STEPS = 256
STAY_STILL_PENALTY = -0.500

# Single Agent Settings
N_ROBOTS = 1 
USE_ROBOT_ATTENTION = False # Disabled for single agent
USE_SEQUENCE_POLICY = False # No need for sequential decision making among robots
USE_K_FLAGS = False

# Features
USE_C = 0
ALLOW_STAY = True
INTERSECTFRONTIER = False
USE_COLLISION = False
USE_TRANS = True
USE_GUIDEPOST = True
USE_ENTROPY = True
USE_UNCONFIRMED_VECTOR = True

INPUT_DIM = 3 + int(USE_C) + int(USE_GUIDEPOST) + int(USE_ENTROPY) + 2 * int(USE_UNCONFIRMED_VECTOR)
IOU_LAMBDA = 0.05
train_mode = True

# Entropy Boost Parameters
ENTROPY_UNCONFIRMED_BOOST = 2.0
ENTROPY_BOOST_RADIUS = 30  # pixels

# Reward Weights
REWARD_CONFIRM = 50.0
REWARD_EXPLORE_CELL = 0.1
REWARD_NEW_SEEN = 5.0
REWARD_DONE = 1000.0
REWARD_STEP_PENALTY = -0.1
REWARD_REPEAT_PENALTY = -0.5

# SAC Parameters
TAU = 0.005
ALPHA = 0.2
TARGET_ENTROPY_SCALE = 0.98

# GPU Load Balancing
ENABLE_GPU_BALANCING = True
GPU_MEMORY_THRESHOLD = 0.85
GPU_IMBALANCE_THRESHOLD = 0.3
GPU_MONITOR_INTERVAL = 100




# Paths
MODEL_SAVE_DIR = 'model_save'
LOG_DIR = 'log'
GIFS_DIR = 'gifs'

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
