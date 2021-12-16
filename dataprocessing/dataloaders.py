from torch.utils.data import DataLoader, Dataset
from config import BATCH_SIZE


def create_data_loader(dataset: Dataset) -> DataLoader:
    """
    Create DataLoader for Dataset

    :param dataset: torch.utils.data.Dataset
        Input dataset
    :return: torch.utils.data.Dataloader
        Dataloader with batch size equal to one provided in config
    """
    return DataLoader(dataset, BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)
