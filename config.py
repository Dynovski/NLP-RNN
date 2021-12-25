import torch

from enum import Enum


class TaskType(Enum):
    SMS = 'sms'
    TWEET = 'tweet'
    NEWS = 'news'


class NetworkType(Enum):
    LSTM = 'lstm'
    LSTM_UNIVERSAL = 'lstm_universal'


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_FOLDER = 'results'
CHECKPOINTS_FOLDER = 'checkpoints'
CHECKPOINT_NAME = 'final.pth'

TASK_TYPE = TaskType.SMS
NETWORK_TYPE = NetworkType.LSTM_UNIVERSAL
IS_MULTICLASS = False
TRAIN_DATA_RATIO = 0.8
BATCH_SIZE = 1
HIDDEN_STATE_SIZE = 256
EMBEDDING_VECTOR_SIZE = 100
NUM_RECURRENT_LAYERS = 2
OUTPUT_CLASSES = 1
VALIDATE_EVERY = 300
CLIP_VALUE = 5

EPOCHS = 1
LEARNING_RATE = 0.005
LOAD_MODEL = False
SAVE_CHECKPOINTS = True
