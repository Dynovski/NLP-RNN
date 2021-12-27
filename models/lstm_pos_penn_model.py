import torch.nn as nn

from torch import tensor, float32, zeros
from numpy import ndarray

import config

from preprocessing.tagging import FullTagger
from dataprocessing.index_mapper import IndexMapper


# For this to work batch size must be set to 1
class LSTMPosPennModel(nn.Module):
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
        super(LSTMPosPennModel, self).__init__()
        self.n_output_classes: int = n_output_classes
        self.hidden_state_size: int = hidden_state_size
        self.embedding_vector_size: int = embedding_vector_size
        self.tagger: FullTagger = FullTagger()
        self.indexMapper: IndexMapper = index_mapper
        self.longest_sequence: int = longest_sequence

        self.embedding = nn.Embedding(n_unique_words, embedding_vector_size)
        self.embedding.weight = nn.Parameter(tensor(embedding_weights_matrix, dtype=float32))
        self.embedding.weight.requires_grad = False

        self.lstm_cell_vbn = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_vbz = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_vbg = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_vbp = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_vbd = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_md = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_nn = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_nnps = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_nnp = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_nns = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_jjs = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_jjr = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_jj = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_rb = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_rbr = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_rbs = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_EMPTY = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_cd = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_in = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_pdt = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_cc = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_ex = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_pos = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_rp = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_fw = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_dt = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_uh = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_to = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_prp = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_prp_dollar = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_dollar = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_wp = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_wp_dollar = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_wdt = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_wrb = nn.LSTMCell(embedding_vector_size, hidden_state_size)
        self.lstm_cell_other = nn.LSTMCell(embedding_vector_size, hidden_state_size)

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
            if tag == 'VBN':
                hidden_state = self.lstm_cell_vbn(cell_input, hidden_state)
            elif tag == 'VBZ':
                hidden_state = self.lstm_cell_vbz(cell_input, hidden_state)
            elif tag == 'VBG':
                hidden_state = self.lstm_cell_vbg(cell_input, hidden_state)
            elif tag == 'VBP':
                hidden_state = self.lstm_cell_vbp(cell_input, hidden_state)
            elif tag == 'VBD':
                hidden_state = self.lstm_cell_vbd(cell_input, hidden_state)
            elif tag == 'MD':
                hidden_state = self.lstm_cell_md(cell_input, hidden_state)
            elif tag == 'NN':
                hidden_state = self.lstm_cell_nn(cell_input, hidden_state)
            elif tag == 'NNPS':
                hidden_state = self.lstm_cell_nnps(cell_input, hidden_state)
            elif tag == 'NNP':
                hidden_state = self.lstm_cell_nnp(cell_input, hidden_state)
            elif tag == 'NNS':
                hidden_state = self.lstm_cell_nns(cell_input, hidden_state)
            elif tag == 'JJS':
                hidden_state = self.lstm_cell_jjs(cell_input, hidden_state)
            elif tag == 'JJR':
                hidden_state = self.lstm_cell_jjr(cell_input, hidden_state)
            elif tag == 'JJ':
                hidden_state = self.lstm_cell_jj(cell_input, hidden_state)
            elif tag == 'RB':
                hidden_state = self.lstm_cell_rb(cell_input, hidden_state)
            elif tag == 'RBR':
                hidden_state = self.lstm_cell_rbr(cell_input, hidden_state)
            elif tag == 'RBS':
                hidden_state = self.lstm_cell_rbs(cell_input, hidden_state)
            elif tag == 'EMPTY':
                hidden_state = self.lstm_cell_EMPTY(cell_input, hidden_state)
            elif tag == 'CD':
                hidden_state = self.lstm_cell_cd(cell_input, hidden_state)
            elif tag == 'IN':
                hidden_state = self.lstm_cell_in(cell_input, hidden_state)
            elif tag == 'PDT':
                hidden_state = self.lstm_cell_pdt(cell_input, hidden_state)
            elif tag == 'CC':
                hidden_state = self.lstm_cell_cc(cell_input, hidden_state)
            elif tag == 'EX':
                hidden_state = self.lstm_cell_ex(cell_input, hidden_state)
            elif tag == 'POS':
                hidden_state = self.lstm_cell_pos(cell_input, hidden_state)
            elif tag == 'RP':
                hidden_state = self.lstm_cell_rp(cell_input, hidden_state)
            elif tag == 'FW':
                hidden_state = self.lstm_cell_fw(cell_input, hidden_state)
            elif tag == 'DT':
                hidden_state = self.lstm_cell_dt(cell_input, hidden_state)
            elif tag == 'UH':
                hidden_state = self.lstm_cell_uh(cell_input, hidden_state)
            elif tag == 'TO':
                hidden_state = self.lstm_cell_to(cell_input, hidden_state)
            elif tag == 'PRP':
                hidden_state = self.lstm_cell_prp(cell_input, hidden_state)
            elif tag == 'PRP$':
                hidden_state = self.lstm_cell_prp_dollar(cell_input, hidden_state)
            elif tag == '$':
                hidden_state = self.lstm_cell_dollar(cell_input, hidden_state)
            elif tag == 'WP':
                hidden_state = self.lstm_cell_wp(cell_input, hidden_state)
            elif tag == 'WP$':
                hidden_state = self.lstm_cell_wp_dollar(cell_input, hidden_state)
            elif tag == 'WDT':
                hidden_state = self.lstm_cell_wdt(cell_input, hidden_state)
            elif tag == 'WRB':
                hidden_state = self.lstm_cell_wrb(cell_input, hidden_state)
            elif tag == 'OTHER':
                hidden_state = self.lstm_cell_other(cell_input, hidden_state)

        out = hidden_state[0]
        out = self.dense(out)

        if not config.IS_MULTICLASS:
            out = self.sigmoid(out)

        return out.squeeze(1)
