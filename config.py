import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_FOLDER = 'results'
CHECKPOINTS_FOLDER = 'checkpoints'
CHECKPOINT_NAME = 'checkpoint.pth'

TRAIN_DATA_RATIO = 0.8
BATCH_SIZE = 32
HIDDEN_SIZE = 1024
EMBEDDING_DIM = 100
NUM_RECURRENT_LAYERS = 10

EPOCHS = 3
LEARNING_RATE = 1e-3
LOAD_MODEL = False
SAVE_CHECKPOINTS = False
