import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from sklearn import metrics
from typing import Optional, List
from torch.optim import Adam
from torch import round
from torch.nn.functional import binary_cross_entropy_with_logits

import config as cfg

from models.lstm_model import LSTMModel
from data_preprocessing import DataPreprocessor, SmsDataPreprocessor, TweetDataPreprocessor, NewsDataPreprocessor
from dataprocessing.datasets import create_double_split_dataset, create_dataset, create_single_split_dataset
from dataprocessing.dataloaders import create_data_loader
from models.utils import save_checkpoint, load_checkpoint, load_model_state_dict
from analyzing.analyzer import Analyzer


class Trainer:
    def __init__(
            self,
            embedding_weights: np.ndarray,
            train_dl: 'torch.utils.data.Dataloader',
            val_dl: 'torch.utils.data.Dataloader'
    ):
        self.embedding_weights: np.ndarray = embedding_weights
        self.train_dl: 'torch.utils.data.Dataloader' = train_dl
        self.val_dl: 'torch.utils.data.Dataloader' = val_dl
        self.is_multiclass: bool = cfg.IS_MULTICLASS

    def _choose_model(self):
        if cfg.NETWORK_TYPE == cfg.NetworkType.LSTM:
            self.model: LSTMModel = LSTMModel(
                self.embedding_weights.shape[0],
                cfg.OUTPUT_CLASSES,
                cfg.EMBEDDING_VECTOR_SIZE,
                self.embedding_weights,
                cfg.HIDDEN_STATE_SIZE,
                cfg.NUM_RECURRENT_LAYERS
            ).to(cfg.DEVICE)

    def _choose_criterion(self):
        if self.is_multiclass:
            self.criterion = nn.CrossEntropyLoss().to(cfg.DEVICE)
        else:
            self.criterion = nn.BCELoss().to(cfg.DEVICE)

    def _choose_optimizer(self):
        self.optimizer = Adam(self.model.parameters(), lr=cfg.LEARNING_RATE)

    def _load_checkpoint(self):
        load_checkpoint(
            self.model,
            self.optimizer,
            cfg.LEARNING_RATE,
            cfg.CHECKPOINT_NAME,
            cfg.CHECKPOINTS_FOLDER
        )

    def _save_checkpoint(self):
        save_checkpoint(
            self.model,
            self.optimizer,
            cfg.CHECKPOINT_NAME,
            cfg.CHECKPOINTS_FOLDER
        )

    def _validate(self):
        loss_values: List[torch.Tensor] = []
        self.model.eval()
        num_correct: int = 0
        for inputs, labels in self.val_dl:
            inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            output = self.model(inputs)
            loss = self.criterion(output.squeeze(), labels.long() if cfg.IS_MULTICLASS else labels.float())
            loss_values.append(loss.item())
            predictions: Optional[torch.Tensor] = None
            if self.is_multiclass:
                predictions: torch.Tensor = torch.nn.functional.softmax(output.squeeze(), dim=0).argmax(1)
            else:
                predictions: torch.Tensor = round(output.squeeze())
            num_correct += (predictions == labels).cpu().sum().item()

        self.model.train()
        return np.mean(loss_values), num_correct / len(val_dl.dataset)

    def run(self):
        self._choose_model()
        self._choose_criterion()
        self._choose_optimizer()

        if cfg.LOAD_MODEL:
            self._load_checkpoint()

        self.model.train()
        train_acc: float = 0.0
        lowest_val_loss = np.Inf
        for epoch in range(cfg.EPOCHS):
            loop = tqdm(self.train_dl, leave=True)
            train_num_correct = 0
            val_l = []
            train_l = []
            steps_l = []

            for index, (inputs, labels) in enumerate(loop):
                inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)

                output = self.model(inputs)
                loss = self.criterion(output.squeeze(), labels.long() if cfg.IS_MULTICLASS else labels.float())

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.CLIP_VALUE)
                self.optimizer.step()

                if index % cfg.ITERS_TO_VALIDATE == 0:
                    val_loss, val_acc = self._validate()
                    loop.set_postfix(
                        epoch=epoch + 1,
                        train_loss=loss.item(),
                        train_accuracy=train_acc,
                        val_loss=val_loss,
                        val_accuracy=val_acc
                    )

                    val_l.append(val_loss)
                    train_l.append(loss.item())
                    steps_l.append(index)

                    if cfg.SAVE_CHECKPOINTS and val_loss <= lowest_val_loss:
                        save_checkpoint(
                            self.model,
                            self.optimizer,
                            cfg.CHECKPOINT_NAME,
                            cfg.CHECKPOINTS_FOLDER
                        )
                        lowest_val_loss = val_loss

                predictions: Optional[torch.Tensor] = None
                if self.is_multiclass:
                    predictions: torch.Tensor = torch.nn.functional.softmax(output.squeeze(), dim=0).argmax(1)
                else:
                    predictions: torch.Tensor = round(output.squeeze())
                train_num_correct += (predictions == labels).cpu().sum().item()

            train_acc = train_num_correct / len(train_dl.dataset)


class Tester:
    def __init__(
            self,
            embedding_weights: np.ndarray,
            data: torch.Tensor,
            labels: torch.Tensor
    ):
        self.embedding_weights: np.ndarray = embedding_weights
        self.data: torch.Tensor = data.to(cfg.DEVICE)
        self.labels: torch.Tensor = labels.to(cfg.DEVICE)
        self.is_multiclass: bool = cfg.IS_MULTICLASS

    def _choose_model(self):
        if cfg.NETWORK_TYPE == cfg.NetworkType.LSTM:
            self.model: LSTMModel = LSTMModel(
                self.embedding_weights.shape[0],
                cfg.OUTPUT_CLASSES,
                cfg.EMBEDDING_VECTOR_SIZE,
                self.embedding_weights,
                cfg.HIDDEN_STATE_SIZE,
                cfg.NUM_RECURRENT_LAYERS
            ).to(cfg.DEVICE)

    def _load_checkpoint(self):
        load_model_state_dict(
            self.model,
            cfg.CHECKPOINT_NAME,
            cfg.CHECKPOINTS_FOLDER
        )

    def _predict(self):
        self.model.eval()
        output = self.model(self.data)
        predictions: Optional[torch.Tensor] = None
        if self.is_multiclass:
            predictions: torch.Tensor = torch.nn.functional.softmax(output.squeeze(), dim=0).argmax(1)
        else:
            predictions: torch.Tensor = round(output.squeeze())
        return predictions.cpu().detach().numpy()

    def _print_metrics(self, predictions):
        print('\nConfusion matrix:')
        cm = metrics.confusion_matrix(self.labels, predictions)
        print(cm)

        # Analyzer.plot_confusion_matrix(cm, ['Ham', 'Spam'], 'figures/confusionMatrix.png')

        print("\nAccuracy: {:.3f}%".format(metrics.accuracy_score(self.labels, predictions) * 100))
        print("F1-score: {:.3f}%".format(metrics.f1_score(self.labels, predictions) * 100))
        print("Precision: {:.3f}%".format(metrics.precision_score(self.labels, predictions) * 100))
        print("Recall: {:.3f}%".format(metrics.recall_score(self.labels, predictions) * 100))
        print("-" * 53)
        print(metrics.classification_report(self.labels, predictions))

    def run(self):
        self._choose_model()
        self._load_checkpoint()
        predictions = self._predict()
        self._print_metrics(predictions)


if __name__ == '__main__':
    preprocessors: Optional[List[DataPreprocessor]] = None

    if cfg.TASK_TYPE == cfg.TaskType.SMS:
        preprocessors: List[SmsDataPreprocessor] = [SmsDataPreprocessor()]
    elif cfg.TASK_TYPE == cfg.TaskType.TWEET:
        preprocessors: List[TweetDataPreprocessor] = [TweetDataPreprocessor()]
    elif cfg.TASK_TYPE == cfg.TaskType.NEWS:
        preprocessors: List[NewsDataPreprocessor] = [NewsDataPreprocessor('data/news_train.csv'),
                                                     NewsDataPreprocessor('data/news_test.csv')]

    assert preprocessors is not None

    [preprocessor.run() for preprocessor in preprocessors]

    training_data: np.ndarray = preprocessors[0].tokenize()
    test_data: Optional[np.ndarray] = None

    if len(preprocessors) > 1:
        test_data: np.ndarray = preprocessors[1].tokenize()

    datasets = None

    if len(preprocessors) == 1:
        datasets = create_double_split_dataset(training_data, preprocessors[0].target_labels, cfg.TRAIN_DATA_RATIO)
    elif len(preprocessors) == 2:
        train_and_val_datasets = create_single_split_dataset(
            training_data,
            preprocessors[0].target_labels,
            cfg.TRAIN_DATA_RATIO
        )
        test_dataset = create_dataset(test_data, preprocessors[1].target_labels)
        datasets = train_and_val_datasets + (test_dataset,)

    train_dl, val_dl, test_dl = [create_data_loader(dataset) for dataset in datasets]

    embedding_matrix: np.ndarray = preprocessors[0].make_embedding_matrix(
        'data/glove.6B.100d.txt',
        cfg.EMBEDDING_VECTOR_SIZE
    )

    Trainer(embedding_matrix, train_dl, val_dl).run()
    Tester(embedding_matrix, datasets[2][:][0], datasets[2][:][1]).run()
