
# MAME Parameter Configuration (Single Agent Refactor)

REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 128
EMBEDDING_DIM = 128

# Habitat maps are large (1000x1000), graph might be large
NODE_PADDING_SIZE = 1000  
K_SIZE = 20  

USE_GPU = True  
USE_GPU_GLOBAL = True
NUM_GPU = 1
NUM_META_AGENT = 16 

LR = 1e-5
GAMMA = 0.99
DECAY_STEP = 256
SUMMARY_WINDOW = 5
LOAD_MODEL = False 
SAVE_IMG_GAP = 50

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

INPUT_DIM = 3 + int(USE_C) + int(USE_GUIDEPOST)
IOU_LAMBDA = 0.05
train_mode = True

# Paths
FOLDER_NAME = 'Semantic_SingleAgent_v1'
model_path = f'model_save/{FOLDER_NAME}'
train_path = f'log/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
