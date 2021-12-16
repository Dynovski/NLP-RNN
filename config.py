import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_FOLDER = 'results'
CHECKPOINTS_FOLDER = 'checkpoints'
TRAIN_DATA_RATIO = 0.7
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
HIDDEN_DIM = 256
NUM_LAYERS = 2
EPOCHS = 10
LOAD_MODEL = False
SAVE_CHECKPOINTS = True
