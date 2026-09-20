"""GCSC differs from Vanilla only through the loader's RandAugment transform."""
from task4.models import resnet18_cifar
from .vanilla import seed_everything, train_closed_set


def train_gcsc(train_loader, val_loader, config: dict, checkpoint_path, device):
    if not config.get("randaugment", False):
        raise ValueError("GCSC requires RandAugment in its training loader")
    seed_everything(int(config.get("seed", 6304)))
    return train_closed_set(resnet18_cifar(10), train_loader, val_loader, config, checkpoint_path, device)
