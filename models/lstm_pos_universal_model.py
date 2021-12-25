import torch.nn as nn

from torch import tensor, float32, zeros
from numpy import ndarray
from typing import Dict

import config

from preprocessing.tagging import UniversalTagger
from index_mapper import IndexMapper


# For this to work batch size must be set to 1
class LSTMPosUniversalModel(nn.Module):
    def __init__(
            self,
            n_unique_words: int,
            n_output_classes: int,
            embedding_vector_size: int,
            embedding_weights_matrix: ndarray,
            hidden_state_size: int,
            index_mapper: IndexMapper,
            longest_sequence: int,
            dropout_prob: float = 0.5
    ):
        super(LSTMPosUniversalModel, self).__init__()
        self.n_output_classes: int = n_output_classes
        self.hidden_state_size: int = hidden_state_size
        self.embedding_vector_size: int = embedding_vector_size
        self.tagger: UniversalTagger = UniversalTagger()
        self.indexMapper: IndexMapper = index_mapper
        self.longest_sequence: int = longest_sequence

        self.embedding = nn.Embedding(n_unique_words, embedding_vector_size)
        self.embedding.weight = nn.Parameter(tensor(embedding_weights_matrix, dtype=float32))
        self.embedding.weight.requires_grad = True

        self.cells_dict: Dict[str, nn.LSTMCell] = {}
        for key in self.tagger.all_tags:
            self.cells_dict[key] = nn.LSTMCell(embedding_vector_size, hidden_state_size)

        self.dense = nn.Sequential(
            nn.Linear(hidden_state_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, n_output_classes)
        )

        if not config.IS_MULTICLASS:
            self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        hidden_state = (
            zeros(config.BATCH_SIZE, self.hidden_state_size).to(config.DEVICE),
            zeros(config.BATCH_SIZE, self.hidden_state_size).to(config.DEVICE)
        )
        batch = batch.long()

        indices_list = batch.tolist()[0]
        word_list = self.indexMapper.indices_to_words(indices_list)
        tag_list = self.tagger.tag_tokens(word_list)

        out = self.embedding(batch)
        for i in range(self.longest_sequence):
            tag = tag_list[i][1]
            cell_input = out[:, i, :]
            hidden_state = self.cells_dict.get(tag)(
                cell_input,
                hidden_state
            )

        out = hidden_state[0]
        out = self.dense(out)

        if not config.IS_MULTICLASS:
            out = self.sigmoid(out)

        # import ipdb; ipdb.set_trace()

        return out.squeeze(1)
