from omegaconf import DictConfig
from hydra.utils import instantiate

from train import Trainer
from utils import seed_everything, get_device
from data import get_train_data, get_test_data

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

import logging
logger = logging.getLogger(__name__)


from medmnist import Evaluator


def run_training(cfg: DictConfig):
    
    seed_everything(cfg.seed)
    device = get_device()

    train_loader, val_loader = get_train_data(cfg)

    model = instantiate(cfg.model).to(device)
    criterion = instantiate(cfg.criterion)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    writer = SummaryWriter(f'../../runs/{current_time}/')

    evaluator = instantiate(cfg.evaluator, writer=writer)

    trainer = Trainer(cfg, model, device, criterion, optimizer, scheduler, evaluator, writer)
    trainer.fit(train_loader, val_loader, cfg.train.num_epochs)

    # trainer.predict(val_loader)

    # test_loader = get_test_data(cfg)
    # trainer.predict(test_loader)