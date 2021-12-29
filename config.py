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
CHECKPOINT_NAME = 'checkpoint_news_lstm.pth'
LOSS_PLOT_NAME = 'news_lstm'
CM_NAME = 'news_lstm_cm'
CM_LABELS = ['World', 'Sports', 'Business', 'Sci/Tech']

TASK_TYPE = TaskType.NEWS
NETWORK_TYPE = NetworkType.LSTM
IS_MULTICLASS = True
TRAIN_DATA_RATIO = 0.9
BATCH_SIZE = 32
HIDDEN_STATE_SIZE = 256
EMBEDDING_VECTOR_SIZE = 100
NUM_RECURRENT_LAYERS = 2
OUTPUT_CLASSES = 4
VALIDATE_EVERY = 500
CLIP_VALUE = 0.1

EPOCHS = 5
LEARNING_RATE = 0.0001
LOAD_MODEL = False
SAVE_CHECKPOINTS = True
