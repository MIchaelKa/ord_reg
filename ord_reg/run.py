from omegaconf import DictConfig
from hydra.utils import instantiate

from train import Trainer
from utils import seed_everything, get_device
from data import get_train_data, get_test_data

import logging
logger = logging.getLogger(__name__)

import torch.nn as nn

from medmnist import Evaluator

def run_training(cfg: DictConfig):
    
    seed_everything(cfg.seed)
    device = get_device()

    n_classes, train_loader, val_loader = get_train_data(cfg)
 
    model = instantiate(cfg.model, n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    evaluator = Evaluator(cfg.train.data_flag, split='val')

    trainer = Trainer(cfg, model, device, criterion, optimizer, scheduler, evaluator)
    trainer.fit(train_loader, val_loader, cfg.train.num_epochs)

    # trainer.predict(val_loader)

    # test_loader = get_test_data(cfg)
    # trainer.predict(test_loader)