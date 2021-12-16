import numpy as np
import torch.nn as nn

from torch import round, sigmoid
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import config

from models.lstm_model import LSTMModel
from data_preprocessing import DataPreProcessor
from dataprocessing.datasets import create_sms_datasets
from dataprocessing.dataloaders import create_data_loader
from models.utils import save_checkpoint, load_checkpoint


def train() -> None:
    datasets = create_sms_datasets('data/SMSDataset.csv', config.TRAIN_DATA_RATIO)

    train_dl, val_dl, test_dl = [create_data_loader(dataset) for dataset in datasets]

    preprocessor = DataPreProcessor('data/SMSDataset.csv', 'latin-1', ['class', 'message'])
    preprocessor.preprocess('message', 'class')
    preprocessor.tokenize()
    vocab_size = len(preprocessor.tokenizer.word_index) + 1
    embedding_matrix = preprocessor.create_glove_embedding_matrix('data/glove.6B.100d.txt', config.EMBEDDING_DIM, vocab_size)

    output_size = 1

    model = LSTMModel(
        vocab_size,
        output_size,
        config.EMBEDDING_DIM,
        embedding_matrix,
        config.HIDDEN_SIZE,
        config.NUM_RECURRENT_LAYERS
    )
    model.to(config.DEVICE)

    print(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(
        model.parameters(),
        lr=config.LEARNING_RATE
    )

    if config.LOAD_MODEL:
        load_checkpoint(
            model,
            optimizer,
            config.LEARNING_RATE,
            config.CHECKPOINT_NAME,
            config.CHECKPOINTS_FOLDER
        )

    scaler: GradScaler = GradScaler()

    model.train()
    train_acc = 0
    for epoch in range(config.EPOCHS):
        h = model.init_hidden(config.BATCH_SIZE)
        loop = tqdm(train_dl, leave=True)
        train_num_correct = 0

        for index, (inputs, labels) in enumerate(loop):
            h = tuple([e.data for e in h])
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)

            with autocast():
                output, h = model(inputs, h)
                loss = criterion(output.squeeze(), labels.float())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if index % 29 == 0:
                val_h = model.init_hidden(config.BATCH_SIZE)
                val_losses = []
                num_val_correct = 0
                model.eval()
                for val_inputs, val_labels in val_dl:
                    val_h = tuple([each.data for each in val_h])
                    val_inputs, val_labels = val_inputs.to(config.DEVICE), val_labels.to(config.DEVICE)
                    val_output, val_h = model(val_inputs, val_h)
                    val_loss = criterion(val_output.squeeze(), val_labels.float())
                    val_losses.append(val_loss.item())
                    predictions = round(sigmoid(val_output.squeeze()))
                    correct_tensor = predictions.eq(val_labels.float().view_as(predictions))
                    correct = np.squeeze(correct_tensor.cpu().numpy())
                    num_val_correct += np.sum(correct)

                val_acc = num_val_correct / len(val_dl.dataset)

                model.train()
                loop.set_postfix(
                    epoch=epoch + 1,
                    train_loss=loss.item(),
                    train_accuracy=train_acc,
                    val_loss=np.mean(val_losses),
                    val_accuracy=val_acc
                )
            predictions = round(sigmoid(output.squeeze()))
            correct_tensor = predictions.eq(labels.float().view_as(predictions))
            correct = np.squeeze(correct_tensor.cpu().numpy())
            train_num_correct += np.sum(correct)

        train_acc = train_num_correct / len(train_dl.dataset)

        if config.SAVE_CHECKPOINTS and epoch % 10 == 9:
            save_checkpoint(
                model,
                optimizer,
                config.CHECKPOINT_NAME,
                config.CHECKPOINTS_FOLDER
            )

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
        predictions = round(sigmoid(output.squeeze()))  # rounds the output to 0/1
        correct_tensor = predictions.eq(labels.float().view_as(predictions))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    test_acc = num_correct / len(test_dl.dataset)
    print("Test accuracy: {:.3f}%".format(test_acc * 100))


if __name__ == '__main__':
    train()
