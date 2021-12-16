import os
import torch
import config


def save_checkpoint(
        model,
        optimizer,
        filename: str = 'checkpoint.pth',
        directory_path: str = 'checkpoints'
) -> None:
    if not directory_exists(directory_path):
        print(f'Creating directory {directory_path}')
        os.mkdir(directory_path)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    print(f'==> Saving checkpoint { os.path.join(directory_path, filename) }')
    torch.save(checkpoint, os.path.join(directory_path, filename))


def load_checkpoint(
        model,
        optimizer,
        lr: float,
        filename: str = 'checkpoint.pth',
        directory_path: str = 'checkpoints'
) -> None:
    if not directory_exists(directory_path):
        return
    print(f'==> Loading checkpoint { os.path.join(directory_path, filename) }')
    checkpoint = torch.load(
        os.path.join(directory_path, filename),
        map_location=config.DEVICE
    )
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def load_model_state_dict(
        model,
        filename: str = 'checkpoint.pth',
        directory_path: str = 'checkpoints'
) -> None:
    if not directory_exists(directory_path):
        return
    print(f'==> Loading state_dict {os.path.join(directory_path, filename)}')
    checkpoint = torch.load(
        os.path.join(directory_path, filename),
        map_location=config.DEVICE
    )
    model.load_state_dict(checkpoint['model'])


def directory_exists(directory: str) -> bool:
    if not os.path.isdir(directory):
        print(f'{directory} is not a valid directory')
        return False
    return True
