import torch

from enum import Enum


class TaskType(Enum):
    SMS = 'sms'
    TWEET = 'tweet'


class NetworkType(Enum):
    LSTM = 'lstm'


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_FOLDER = 'results'
CHECKPOINTS_FOLDER = 'checkpoints'
CHECKPOINT_NAME = 'checkpoint.pth'

TASK_TYPE = TaskType.TWEET
NETWORK_TYPE = NetworkType.LSTM
TRAIN_DATA_RATIO = 0.8
BATCH_SIZE = 32
HIDDEN_STATE_SIZE = 256
EMBEDDING_VECTOR_SIZE = 100
NUM_RECURRENT_LAYERS = 2
OUTPUT_CLASSES = 1

EPOCHS = 3
LEARNING_RATE = 0.005
LOAD_MODEL = False
SAVE_CHECKPOINTS = False
