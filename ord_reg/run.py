from omegaconf import DictConfig
from hydra.utils import instantiate

from train import Trainer
from utils import seed_everything, get_device
from data import get_train_data, get_test_data
from model import get_model

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

import logging
logger = logging.getLogger(__name__)

def get_writer(cfg: DictConfig):
    experiment_name = f'{cfg.experiment_name}_{cfg.model.encoder.model_name}_lr:{cfg.optimizer.lr}'
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    tensorboard_folder = f'{experiment_name}_{current_time}'
    logger.info(f'experiment: {experiment_name}')
    writer = SummaryWriter(f'../../runs/{tensorboard_folder}/')
    return writer

def run_training(cfg: DictConfig):
    seed_everything(cfg.seed)

    device = get_device()
    writer = get_writer(cfg)
    train_loader, val_loader = get_train_data(cfg)
    model = get_model(cfg.model).to(device)

    criterion = instantiate(cfg.criterion)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
    evaluator = instantiate(cfg.evaluator, writer=writer)

    trainer = Trainer(cfg, model, device, criterion, optimizer, scheduler, evaluator, writer)
    trainer.fit(train_loader, val_loader, cfg.train.num_epochs)

    # trainer.predict(val_loader)

    if cfg.save_checkpoint:
        test_loader = get_test_data(cfg)
        trainer.predict(test_loader)