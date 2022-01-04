import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from sklearn import metrics
from typing import Optional, List
from torch.optim import Adam
from torch import round
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import DataLoader
from keras.preprocessing.text import Tokenizer

import config as cfg

from models.lstm_model import LSTMModel
from models.lstm_pos_universal_model import LSTMPosUniversalModel
from models.lstm_pos_penn_model import LSTMPosPennModel
from preprocessing.data_preprocessing import DataPreprocessor, SmsDataPreprocessor, TweetDataPreprocessor, NewsDataPreprocessor
from dataprocessing.datasets import create_double_split_dataset
from dataprocessing.dataloaders import create_data_loader
from models.utils import save_checkpoint, load_checkpoint, load_model_state_dict
from analyzing.analyzer import Analyzer
from dataprocessing.index_mapper import IndexMapper


class Trainer:
    def __init__(
            self,
            embedding_weights: np.ndarray,
            train_dataloader: DataLoader,
            validation_dataloader: DataLoader,
            tokenizer: Tokenizer,
            longest_sequence: int,
            network_type: cfg.NetworkType = cfg.NETWORK_TYPE
    ):
        self.embedding_weights: np.ndarray = embedding_weights
        self.train_dl: DataLoader = train_dataloader
        self.val_dl: DataLoader = validation_dataloader
        self.tokenizer: Tokenizer = tokenizer
        self.longest_sequence: int = longest_sequence
        self.is_multiclass: bool = cfg.IS_MULTICLASS
        self.network_type: cfg.NetworkType = network_type

    def _choose_model(self):
        if self.network_type == cfg.NetworkType.LSTM:
            self.model: LSTMModel = LSTMModel(
                self.embedding_weights.shape[0],
                cfg.OUTPUT_SIZE,
                cfg.EMBEDDING_VECTOR_SIZE,
                self.embedding_weights,
                cfg.HIDDEN_STATE_SIZE,
                cfg.NUM_RECURRENT_LAYERS
            ).to(cfg.DEVICE)
        elif self.network_type == cfg.NETWORK_TYPE.LSTM_UNIVERSAL:
            self.model: LSTMPosUniversalModel = LSTMPosUniversalModel(
                self.embedding_weights.shape[0],
                cfg.OUTPUT_SIZE,
                cfg.EMBEDDING_VECTOR_SIZE,
                self.embedding_weights,
                cfg.HIDDEN_STATE_SIZE,
                IndexMapper(self.tokenizer),
                self.longest_sequence
            ).to(cfg.DEVICE)
        elif self.network_type == cfg.NETWORK_TYPE.LSTM_PENN:
            self.model: LSTMPosPennModel = LSTMPosPennModel(
                self.embedding_weights.shape[0],
                cfg.OUTPUT_SIZE,
                cfg.EMBEDDING_VECTOR_SIZE,
                self.embedding_weights,
                cfg.HIDDEN_STATE_SIZE,
                IndexMapper(self.tokenizer),
                self.longest_sequence
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
        instances: int = 0
        for inputs, labels in self.val_dl:
            inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            instances += inputs.shape[0]
            output = self.model(inputs)
            loss = self.criterion(output, labels.long() if cfg.IS_MULTICLASS else labels.float())
            loss_values.append(loss.item())
            predictions: Optional[torch.Tensor] = None
            if self.is_multiclass:
                dim: int = 1 if cfg.TASK_TYPE == cfg.TaskType.NEWS else 0
                predictions: torch.Tensor = torch.nn.functional.softmax(output, dim=dim).argmax(1)
            else:
                predictions: torch.Tensor = round(output)
            num_correct += (predictions == labels).cpu().sum().item()

        self.model.train()
        return np.mean(loss_values), num_correct / instances

    def run(self):
        self._choose_model()
        self._choose_criterion()
        self._choose_optimizer()

        if cfg.LOAD_MODEL:
            self._load_checkpoint()

        self.model.train()
        train_acc: float = 0.0
        lowest_val_loss = np.Inf
        best_val_acc: float = 0.0
        lowest_train_loss = np.Inf
        for epoch in range(cfg.EPOCHS):
            loop = tqdm(self.train_dl, leave=True)
            train_num_correct = 0
            instances: int = 0
            val_l = []
            train_l = []
            steps_l = []

            for index, (inputs, labels) in enumerate(loop):
                inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                instances += inputs.shape[0]
                output = self.model(inputs)
                loss = self.criterion(output, labels.long() if cfg.IS_MULTICLASS else labels.float())

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.CLIP_VALUE)
                self.optimizer.step()

                if index % cfg.VALIDATE_EVERY == 0:
                    val_loss, val_acc = self._validate()
                    loop.set_postfix(
                        epoch=epoch + 1,
                        train_loss=loss.item(),
                        train_accuracy=train_acc,
                        val_loss=val_loss,
                        val_accuracy=val_acc
                    )

                    val_l.append(val_loss.item())
                    train_l.append(loss.item())
                    steps_l.append(index)

                    if (
                            cfg.SAVE_CHECKPOINTS and
                            (
                                    val_loss <= lowest_val_loss or
                                    (val_acc >= best_val_acc and loss.item() <= lowest_train_loss)
                            )
                    ):
                        save_checkpoint(
                            self.model,
                            self.optimizer,
                            cfg.CHECKPOINT_NAME,
                            cfg.CHECKPOINTS_FOLDER
                        )
                        lowest_val_loss = val_loss
                        best_val_acc = val_acc
                        lowest_train_loss = loss.item()

                predictions: Optional[torch.Tensor] = None
                if self.is_multiclass:
                    dim: int = 1 if cfg.TASK_TYPE == cfg.TaskType.NEWS else 0
                    predictions: torch.Tensor = torch.nn.functional.softmax(output, dim=dim).argmax(1)
                else:
                    predictions: torch.Tensor = round(output)
                train_num_correct += (predictions == labels).cpu().sum().item()

            train_acc = train_num_correct / instances
            Analyzer.plot_losses(train_l, val_l, steps_l)


class Tester:
    def __init__(
            self,
            embedding_weights: np.ndarray,
            data: torch.Tensor,
            labels: torch.Tensor,
            tokenizer: Tokenizer,
            longest_sequence: int,
            network_type: cfg.NetworkType = cfg.NETWORK_TYPE
    ):
        self.embedding_weights: np.ndarray = embedding_weights
        self.data: torch.Tensor = data.to(cfg.DEVICE)
        self.labels: torch.Tensor = labels
        self.tokenizer: Tokenizer = tokenizer
        self.longest_sequence: int = longest_sequence
        self.is_multiclass: bool = cfg.IS_MULTICLASS
        self.network_type: cfg.NetworkType = network_type

    def _choose_model(self):
        if self.network_type == cfg.NetworkType.LSTM:
            self.model: LSTMModel = LSTMModel(
                self.embedding_weights.shape[0],
                cfg.OUTPUT_SIZE,
                cfg.EMBEDDING_VECTOR_SIZE,
                self.embedding_weights,
                cfg.HIDDEN_STATE_SIZE,
                cfg.NUM_RECURRENT_LAYERS
            ).to(cfg.DEVICE)
        elif self.network_type == cfg.NETWORK_TYPE.LSTM_UNIVERSAL:
            self.model: LSTMPosUniversalModel = LSTMPosUniversalModel(
                self.embedding_weights.shape[0],
                cfg.OUTPUT_SIZE,
                cfg.EMBEDDING_VECTOR_SIZE,
                self.embedding_weights,
                cfg.HIDDEN_STATE_SIZE,
                IndexMapper(self.tokenizer),
                self.longest_sequence
            ).to(cfg.DEVICE)
        elif self.network_type == cfg.NETWORK_TYPE.LSTM_PENN:
            self.model: LSTMPosPennModel = LSTMPosPennModel(
                self.embedding_weights.shape[0],
                cfg.OUTPUT_SIZE,
                cfg.EMBEDDING_VECTOR_SIZE,
                self.embedding_weights,
                cfg.HIDDEN_STATE_SIZE,
                IndexMapper(self.tokenizer),
                self.longest_sequence
            ).to(cfg.DEVICE)

    def _load_checkpoint(self):
        load_model_state_dict(
            self.model,
            cfg.CHECKPOINT_NAME,
            cfg.CHECKPOINTS_FOLDER
        )

    def _predict(self):
        self.model.eval()
        output = None
        if cfg.BATCH_SIZE == 1:
            outputs = tuple(self.model(data.unsqueeze(0)) for data in self.data)
            output = torch.stack(outputs, dim=0)
        else:
            output = self.model(self.data)
        predictions: Optional[torch.Tensor] = None
        if self.is_multiclass:
            dim: int = 2 if cfg.TASK_TYPE == cfg.TaskType.NEWS and cfg.NETWORK_TYPE != cfg.NetworkType.LSTM else 1
            predictions: torch.Tensor = torch.nn.functional.softmax(output, dim=0).argmax(dim)
        else:
            predictions: torch.Tensor = round(output)
        return predictions.cpu().detach().numpy()

    def _print_metrics(self, predictions: np.ndarray):
        print('\nConfusion matrix:')
        cm = metrics.confusion_matrix(self.labels, predictions)
        print(cm)

        Analyzer.plot_confusion_matrix(cm, cfg.CM_LABELS)

        print("\nAccuracy: {:.3f}%".format(metrics.accuracy_score(self.labels, predictions) * 100))
        print("F1-score: {:.3f}%".format(metrics.f1_score(
            self.labels,
            predictions,
            average='binary' if not cfg.IS_MULTICLASS else 'weighted'
        ) * 100))
        print("Precision: {:.3f}%".format(metrics.precision_score(
            self.labels,
            predictions,
            average='binary' if not cfg.IS_MULTICLASS else 'weighted'
        ) * 100))
        print("Recall: {:.3f}%".format(metrics.recall_score(
            self.labels,
            predictions,
            average='binary' if not cfg.IS_MULTICLASS else 'weighted'
        ) * 100))
        print("-" * 53)
        print(metrics.classification_report(self.labels, predictions))

    def get_metrics(self, predictions: np.ndarray):
        return (
            metrics.accuracy_score(self.labels, predictions),
            metrics.f1_score(self.labels, predictions, average='binary' if not cfg.IS_MULTICLASS else 'weighted'),
            metrics.precision_score(self.labels, predictions, average='binary' if not cfg.IS_MULTICLASS else 'weighted'),
            metrics.recall_score(self.labels, predictions, average='binary' if not cfg.IS_MULTICLASS else 'weighted')
        )

    def run(self):
        self._choose_model()
        self._load_checkpoint()
        predictions = self._predict()
        self._print_metrics(predictions)
        return self.get_metrics(predictions)


if __name__ == '__main__':
    preprocessor: Optional[DataPreprocessor] = None

    if cfg.TASK_TYPE == cfg.TaskType.SMS:
        preprocessor: SmsDataPreprocessor = SmsDataPreprocessor()
    elif cfg.TASK_TYPE == cfg.TaskType.TWEET:
        preprocessor: TweetDataPreprocessor = TweetDataPreprocessor()
    elif cfg.TASK_TYPE == cfg.TaskType.NEWS:
        preprocessor: NewsDataPreprocessor = NewsDataPreprocessor()

    preprocessor.run()

    training_data: np.ndarray = preprocessor.tokenize()

    datasets = create_double_split_dataset(training_data, preprocessor.target_labels, cfg.TRAIN_DATA_RATIO)

    train_dl, val_dl, test_dl = [create_data_loader(dataset) for dataset in datasets]

    embedding_matrix: np.ndarray = preprocessor.make_embedding_matrix(
        'data/glove.6B.100d.txt',
        cfg.EMBEDDING_VECTOR_SIZE
    )

    Trainer(embedding_matrix, train_dl, val_dl, preprocessor.tokenizer, training_data.shape[1]).run()
    Tester(
        embedding_matrix,
        datasets[2][:][0],
        datasets[2][:][1],
        preprocessor.tokenizer,
        training_data.shape[1]
    ).run()
