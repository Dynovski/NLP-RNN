import torch.nn as nn

from torch import tensor, float32
from numpy import ndarray

import config


class LSTMModel(nn.Module):
    def __init__(
            self,
            n_unique_words: int,
            n_output_classes: int,
            embedding_vector_size: int,
            embedding_weights_matrix: ndarray,
            hidden_state_size: int,
            n_lstm_layers: int,
            is_lstm_bidirectional: bool = False,
            dropout_prob: float = 0.5
    ):
        super(LSTMModel, self).__init__()
        self.n_output_classes: int = n_output_classes
        self.n_lstm_layers: int = n_lstm_layers
        self.hidden_state_size: int = hidden_state_size
        self.multiplier: int = int(is_lstm_bidirectional) + 1

        self.embedding = nn.Embedding(n_unique_words, embedding_vector_size)
        self.embedding.weight = nn.Parameter(tensor(embedding_weights_matrix, dtype=float32))
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            embedding_vector_size,
            hidden_state_size,
            n_lstm_layers,
            bidirectional=is_lstm_bidirectional,
            dropout=dropout_prob,
            batch_first=True
        )

        self.dense = nn.Sequential(
            nn.BatchNorm1d(hidden_state_size if not is_lstm_bidirectional else 2 * hidden_state_size),
            nn.Linear(hidden_state_size if not is_lstm_bidirectional else 2 * hidden_state_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, n_output_classes)
        )

        if not config.IS_MULTICLASS:
            self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        batch = batch.long()
        out = self.embedding(batch)
        out, _ = self.lstm(out)
        out = out[:, -1, :]
        out = self.dense(out)

        if not config.IS_MULTICLASS:
            out = self.sigmoid(out)

        return out.squeeze()
