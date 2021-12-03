import medmnist
from medmnist import INFO

from omegaconf import DictConfig

import torch.utils.data as data
import torchvision.transforms as transforms

from hydra.utils import instantiate

import logging
logger = logging.getLogger(__name__)

def get_data_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    return transform

def get_train_data(cfg: DictConfig):
    data_transform = get_data_transform()
    download = cfg.train.download

    train_dataset = instantiate(cfg.train.dataset, split='train', transform=data_transform, download=download)
    val_dataset = instantiate(cfg.train.dataset, split='val', transform=data_transform, download=download)

    logger.info(f'Dataset size, train: {len(train_dataset)}, valid: {len(val_dataset)}')

    dataloader_workers = cfg.train.dataloader_workers
    pin_memory = False

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size_train,
        shuffle=True,
        num_workers=dataloader_workers,
        pin_memory=pin_memory
    )

    val_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size_val,
        shuffle=False,
        num_workers=dataloader_workers,
        pin_memory=pin_memory
    )

    logger.info(f'Dataloader size, train: {len(train_loader)}, val: {len(val_loader)}')

    return train_loader, val_loader

def get_test_data(cfg: DictConfig):
    info = INFO[cfg.train.data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    data_transform = get_data_transform()
    test_dataset = DataClass(split='test', transform=data_transform, download=cfg.train.download)

    logger.info(f'Dataset size, test: {len(test_dataset)}')

    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=cfg.train.batch_size_val,
        shuffle=False,
        num_workers=cfg.train.dataloader_workers
    )

    logger.info(f'Dataloader size, test: {len(test_loader)}')

    return test_loader

