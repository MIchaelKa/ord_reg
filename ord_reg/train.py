import torch
import numpy as np

import logging
logger = logging.getLogger(__name__)

class Trainer():
    def __init__(self, config, model, device, criterion, optimizer, scheduler, evaluator, writer):
        self.config = config
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.writer = writer

        self.model_save_name = f'{self.config.model.encoder.model_name}.pth'
        self.print_every = -1 # for debug purposes

    def fit(self, train_loader, val_loader, num_epochs):

        logger.info('Start training...')

        val_best_score = 0
        val_best_epoch = -1

        for epoch in range(num_epochs):
            self.train_epoch(epoch, train_loader)
            loss, acc, qwk, _ = self.val_epoch(val_loader)

            self.writer.add_scalar('val/acc', acc, epoch)
            self.writer.add_scalar('val/qwk', qwk, epoch)
            self.writer.add_scalar('val/loss', loss, epoch)

            logger.info('Epoch: {:>2d}, loss = {:.5f}, acc = {:.5f}, qwk = {:.5f}'.
                format(epoch, loss, acc, qwk))

            # we can choose any of those [-loss, acc, qwk] for saving best model
            score = qwk

            if score > val_best_score or val_best_epoch == -1:
                val_best_score = score
                val_best_epoch = epoch

                if self.config.save_checkpoint:
                    torch.save(self.model.state_dict(), self.model_save_name)
                
            self.update_scheduler(epoch)

        logger.info('Best epoch: {:>2d}, score = {:.5f}'.format(val_best_epoch, val_best_score))

        self.writer.close()

    def update_scheduler(self, epoch):
        last_lr = self.scheduler.get_last_lr()[0]
        self.writer.add_scalar('lr', last_lr, epoch)
        self.scheduler.step()

    def train_epoch(self, epoch, train_loader):
        
        self.model.train()

        iter_start = epoch * len(train_loader)

        for iter_num, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(self.device, dtype=torch.float32)
            y_batch = y_batch.to(self.device)

            output = self.model(x_batch)

            loss = self.criterion(output, y_batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_item = loss.item()

            self.writer.add_scalar('train/loss', loss_item, iter_start+iter_num)

            if self.print_every > 0 and iter_num % self.print_every == 0:
                logger.info('iter: {:>4d}, loss = {:.5f}'.format(iter_num, loss_item))

    def predict(self, loader):
        self.model.load_state_dict(torch.load(self.model_save_name))
        self.model.eval()
        loss, acc, qwk, cm = self.val_epoch(loader)

        logger.info('[Test] loss = {:.5f}, acc = {:.5f}, qwk = {:.5f}'.format(loss, acc, qwk))
        logger.info(f'[Test] cm:\n {cm}')

    def val_epoch(self, val_loader):

        self.model.eval()

        y_true = []
        outputs = []
        losses = []

        with torch.no_grad():
            for _, (x_batch, y_batch) in enumerate(val_loader):
                x_batch = x_batch.to(self.device, dtype=torch.float32)
                y_batch = y_batch.to(self.device)
                
                output = self.model(x_batch)
                loss = self.criterion(output, y_batch)

                outputs.append(output)
                y_true.append(y_batch)
                losses.append(loss.item())

        outputs = torch.cat(outputs, 0)
        y_true = torch.cat(y_true, 0)

        acc, qwk, cm = self.evaluator.evaluate(outputs, y_true)
        loss = np.mean(losses)

        return loss, acc, qwk, cm

