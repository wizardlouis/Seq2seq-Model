import math

# hyper parameters for dataset
T_ON = 15
T_OFF = 5
T_REST = 30
T_RON = 10
T_ROFF = 5
T_CUE = 5
T_DELAY = 50
T_ADD = 50
T_RETRIEVE = 150
G_D = 0
DELAY_FIXED = True

RANK_SIZE = 2
N_ITEM = 6
ITEM_LIST = list(range(1, N_ITEM + 1))
TARGET_DIM = 2
TRAIN_RATIO = 0.8

# hyper parameters for training
N_EPOCHS = 5
LEARNING_RATE = 0.001
BURN_IN = 2
BATCH_SIZE = 6 * (RANK_SIZE - 1)

# hyper parameters for model
N_NEURONS = 128
DECAY = 0.1


# default vector is used to represent sequence in three dimension:(x,y,cue)
DEFAULT_VECTOR = [
    [0., 0., 1.],
    [math.cos(0), math.sin(0), 0.],
    [math.cos(math.pi / 3), math.sin(math.pi / 3), 0.],
    [math.cos(math.pi * 2 / 3), math.sin(math.pi * 2 / 3), 0.],
    [math.cos(math.pi), math.sin(math.pi), 0],
    [math.cos(math.pi * 4 / 3), math.sin(math.pi * 4 / 3), 0.],
    [math.cos(math.pi * 5 / 3), math.sin(math.pi * 5 / 3), 0.]]
DIRECTION = [
    [math.cos(0), math.sin(0)],
    [math.cos(math.pi / 3), math.sin(math.pi / 3)],
    [math.cos(math.pi * 2 / 3), math.sin(math.pi * 2 / 3)],
    [math.cos(math.pi), math.sin(math.pi)],
    [math.cos(math.pi * 4 / 3), math.sin(math.pi * 4 / 3)],
    [math.cos(math.pi * 5 / 3), math.sin(math.pi * 5 / 3)]
]
