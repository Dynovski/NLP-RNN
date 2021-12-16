import config
import torch.nn as nn
from torch.optim import Adam
from data_preprocessing import DataPreProcessor
from torch.utils.data import DataLoader
from datasets.sms_dataset import create_sms_datasets
from networks.lstm_network import LSTMNetwork
from tqdm import tqdm
import numpy as np
import torch


def train():
    train_dataset, val_dataset, test_dataset = create_sms_datasets('data/SMSDataset.csv', config.TRAIN_DATA_RATIO)

    train_dl = DataLoader(train_dataset, config.BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_dataset, config.BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)
    test_dl = DataLoader(test_dataset, config.BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)

    preprocessor = DataPreProcessor('data/SMSDataset.csv', 'latin-1', ['class', 'message'])
    preprocessor.preprocess('message', 'class')
    preprocessor.tokenize()
    vocab_size = len(preprocessor.tokenizer.word_index) + 1
    embedding_matrix = preprocessor.create_glove_embedding_matrix('data/glove.42B.300d.txt', config.EMBEDDING_DIM, vocab_size)

    output_size = 1

    model = LSTMNetwork(vocab_size, output_size, config.EMBEDDING_DIM, embedding_matrix, config.HIDDEN_SIZE, config.NUM_RECURRENT_LAYERS)
    model.to(config.DEVICE)

    print(model)

    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)

    model.train()
    for i in range(config.EPOCHS):
        h = model.init_hidden(config.BATCH_SIZE)
        loop = tqdm(train_dl, leave=True)

        for index, (inputs, labels) in enumerate(loop):
            h = tuple([e.data for e in h])
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            model.zero_grad()
            output, h = model(inputs, h)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            if index % 19 == 0:
                val_h = model.init_hidden(config.BATCH_SIZE)
                val_losses = []
                model.eval()
                for inp, lab in val_dl:
                    val_h = tuple([each.data for each in val_h])
                    inp, lab = inp.to(config.DEVICE), lab.to(config.DEVICE)
                    out, val_h = model(inp, val_h)
                    val_loss = criterion(out.squeeze(), lab.float())
                    val_losses.append(val_loss.item())

                model.train()
                print("Epoch: {}/{}...".format(i + 1, config.EPOCHS),
                      "Step: {}...".format(index),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))

    test_losses = []
    num_correct = 0
    h = model.init_hidden(config.BATCH_SIZE)

    model.eval()
    for inputs, labels in test_dl:
        h = tuple([each.data for each in h])
        inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
        output, h = model(inputs, h)
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())
        pred = torch.round(output.squeeze())  # rounds the output to 0/1
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    test_acc = num_correct / len(test_dl.dataset)
    print("Test accuracy: {:.3f}%".format(test_acc * 100))


if __name__ == '__main__':
    train()
