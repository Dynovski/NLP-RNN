import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from sklearn import metrics
from typing import Optional
from torch.optim import Adam
from torch import round, sigmoid
from torch.cuda.amp import GradScaler, autocast
from typing import List

import config as cfg

from models.lstm_model import LSTMModel
from data_preprocessing import DataPreprocessor, SmsDataPreprocessor, TweetDataPreprocessor
from dataprocessing.datasets import create_double_split_dataset, create_dataset, create_single_split_dataset
from dataprocessing.dataloaders import create_data_loader
from models.utils import save_checkpoint, load_checkpoint


def train(
        embedding_matrix: np.ndarray,
        train_dl: 'torch.utils.data.Dataloader',
        val_dl: 'torch.utils.data.Dataloader'
) -> None:
    model: Optional[torch.Module] = None

    if cfg.NETWORK_TYPE == cfg.NetworkType.LSTM:
        model: LSTMModel = LSTMModel(
            embedding_matrix.shape[0],
            cfg.OUTPUT_CLASSES,
            cfg.EMBEDDING_VECTOR_SIZE,
            embedding_matrix,
            cfg.HIDDEN_STATE_SIZE,
            cfg.NUM_RECURRENT_LAYERS
        ).to(cfg.DEVICE)

    assert model is not None

    print(model)

    criterion = nn.BCEWithLogitsLoss().to(cfg.DEVICE)
    optimizer = Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    if cfg.LOAD_MODEL:
        load_checkpoint(
            model,
            optimizer,
            cfg.LEARNING_RATE,
            cfg.CHECKPOINT_NAME,
            cfg.CHECKPOINTS_FOLDER
        )

    scaler: GradScaler = GradScaler()

    model.train()
    train_acc: float = 0.0
    best_val_loss = np.Inf
    for epoch in range(cfg.EPOCHS):
        loop = tqdm(train_dl, leave=True)
        train_num_correct = 0
        val_l = []
        train_l = []
        steps_l = []

        for index, (inputs, labels) in enumerate(loop):
            inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)

            with autocast():
                output = model(inputs)
                loss = criterion(output.squeeze(), labels.float())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if index % 19 == 0:
                val_losses = []
                num_val_correct = 0
                model.eval()
                for val_inputs, val_labels in val_dl:
                    val_inputs, val_labels = val_inputs.to(cfg.DEVICE), val_labels.to(cfg.DEVICE)
                    val_output = model(val_inputs)
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
                val_l.append(np.mean(val_losses))
                train_l.append(loss.item())
                steps_l.append(index)

                if np.mean(val_losses) <= best_val_loss:
                    torch.save(model.state_dict(), './state/state_dict.pth')
                    print(
                        '\nValidation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(best_val_loss, np.mean(val_losses))
                    )
                    best_val_loss = np.mean(val_losses)
            predictions = round(sigmoid(output.squeeze()))
            correct_tensor = predictions.eq(labels.float().view_as(predictions))
            correct = np.squeeze(correct_tensor.cpu().numpy())
            train_num_correct += np.sum(correct)

        train_acc = train_num_correct / len(train_dl.dataset)

        if cfg.SAVE_CHECKPOINTS and epoch % 10 == 9:
            save_checkpoint(
                model,
                optimizer,
                cfg.CHECKPOINT_NAME,
                cfg.CHECKPOINTS_FOLDER
            )


def test(
        embedding_matrix: np.ndarray,
        data: torch.Tensor,
        labels: torch.Tensor
) -> None:
    model: Optional[torch.Module] = None

    if cfg.NETWORK_TYPE == cfg.NetworkType.LSTM:
        model: LSTMModel = LSTMModel(
            embedding_matrix.shape[0],
            cfg.OUTPUT_CLASSES,
            cfg.EMBEDDING_VECTOR_SIZE,
            embedding_matrix,
            cfg.HIDDEN_STATE_SIZE,
            cfg.NUM_RECURRENT_LAYERS
        ).to(cfg.DEVICE)

    model.load_state_dict(torch.load('./state/state_dict.pth'))

    model.eval()
    data = data.to(cfg.DEVICE)
    predictions = round(sigmoid(model(data))).detach().numpy()

    print('\nConfusion matrix:')
    print(metrics.confusion_matrix(labels, predictions))

    print("\nAccuracy: {:.3f}%".format(metrics.accuracy_score(labels, predictions) * 100))
    print("F1-score: {:.3f}%".format(metrics.f1_score(labels, predictions) * 100))
    print("Precision: {:.3f}%".format(metrics.precision_score(labels, predictions) * 100))
    print("Recall: {:.3f}%".format(metrics.recall_score(labels, predictions) * 100))
    print("-" * 53)
    print(metrics.classification_report(labels, predictions))


if __name__ == '__main__':
    preprocessor: Optional[DataPreprocessor] = None

    if cfg.TASK_TYPE == cfg.TaskType.SMS:
        preprocessor: SmsDataPreprocessor = SmsDataPreprocessor()
    elif cfg.TASK_TYPE == cfg.TaskType.TWEET:
        preprocessor: TweetDataPreprocessor = TweetDataPreprocessor('data/tweets_train.csv')

    assert preprocessor is not None

    preprocessor.run()

    training_data: np.ndarray = preprocessor.tokenize()

    datasets = create_double_split_dataset(training_data, preprocessor.target_labels, cfg.TRAIN_DATA_RATIO)

    train_dl, val_dl, test_dl = [create_data_loader(dataset) for dataset in datasets]

    embedding_matrix: np.ndarray = preprocessor.make_embedding_matrix(
        'data/glove.6B.100d.txt',
        cfg.EMBEDDING_VECTOR_SIZE
    )

    train(embedding_matrix, train_dl, val_dl)
    test(embedding_matrix, datasets[2][:][0], datasets[2][:][1])
