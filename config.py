import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_FOLDER = 'results'
CHECKPOINTS_FOLDER = 'checkpoints'
CHECKPOINT_NAME = 'checkpoint.pth'

TRAIN_DATA_RATIO = 0.7
BATCH_SIZE = 32
HIDDEN_SIZE = 512
EMBEDDING_DIM = 100
NUM_RECURRENT_LAYERS = 6

EPOCHS = 10
LEARNING_RATE = 1e-3
LOAD_MODEL = False
SAVE_CHECKPOINTS = False
