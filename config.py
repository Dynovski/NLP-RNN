import torch

from enum import Enum


class TaskType(Enum):
    SMS = 'sms'
    TWEET = 'tweet'
    NEWS = 'news'


class NetworkType(Enum):
    LSTM = 'lstm'
    LSTM_UNIVERSAL = 'lstm_universal'
    LSTM_PENN = 'lstm_penn'


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINTS_FOLDER = 'checkpoints'
CHECKPOINT_NAME = 'checkpoint_tweet_lstm.pth'
LOSS_PLOT_NAME = 'tweet_lstm'
CM_NAME = 'tweet_lstm_cm'
CM_LABELS = ['Real', 'Disaster']

TASK_TYPE = TaskType.TWEET
NETWORK_TYPE = NetworkType.LSTM
IS_MULTICLASS = False
TRAIN_DATA_RATIO = 0.8
BATCH_SIZE = 32
HIDDEN_STATE_SIZE = 256
EMBEDDING_VECTOR_SIZE = 100
NUM_RECURRENT_LAYERS = 2
OUTPUT_CLASSES = 1
VALIDATE_EVERY = 20
CLIP_VALUE = 0.1

EPOCHS = 5
LEARNING_RATE = 0.0001
LOAD_MODEL = False
SAVE_CHECKPOINTS = True
