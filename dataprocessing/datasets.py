from typing import Tuple
from numpy import ndarray
from torch import from_numpy
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split


def create_sms_datasets(
        data: ndarray,
        labels: ndarray,
        train_data_ratio: float
) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Create train, validation and test data processing for SMS Collection data

    :param data: numpy.ndarray
        Array containing data for training
    :param labels: numpy.ndarray
        Array containing labels for training data
    :param train_data_ratio: float
        Ratio from 0 to 1 specifying how much of all data should be used for training, remaining data is split
        50/50 between validation and test
    :return: Tuple[TensorDataset, TensorDataset, TensorDataset]
        Train, validation and test dataset
    """
    assert 0 <= train_data_ratio <= 1

    train_data, test_data, train_labels, test_labels = train_test_split(
        data,
        labels,
        train_size=train_data_ratio,
        random_state=7
    )

    split_idx = int(0.5 * len(test_data))
    val_data, test_data = test_data[:split_idx], test_data[split_idx:]
    val_labels, test_labels = test_labels[:split_idx], test_labels[split_idx:]

    train_data = TensorDataset(from_numpy(train_data), from_numpy(train_labels))
    val_data = TensorDataset(from_numpy(val_data), from_numpy(val_labels))
    test_data = TensorDataset(from_numpy(test_data), from_numpy(test_labels))

    return train_data, val_data, test_data
