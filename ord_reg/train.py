import torch
import numpy as np

from datetime import datetime

import logging
logger = logging.getLogger(__name__)

from torch.utils.tensorboard import SummaryWriter

class Trainer():
    def __init__(self, config, model, device, criterion, optimizer, scheduler, evaluator):
        self.config = config
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # TODO: what about test?
        self.evaluator = evaluator
 
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.writer = SummaryWriter(f'../../runs/{current_time}/')

    def fit(self, train_loader, val_loader, num_epochs):

        logger.info('Start training...')

        val_best_score = 0
        val_best_epoch = 0

        for epoch in range(num_epochs):
            self.train_epoch(epoch, train_loader)
            val_score = self.val_epoch(epoch, val_loader)

            if val_score > val_best_score:
                val_best_score = val_score
                val_best_epoch = epoch

                if self.config.train.save_checkpoint:
                    model_save_name = f'{self.config.model.model_name}.pth'
                    torch.save(self.model.state_dict(), model_save_name)
                
            self.update_scheduler(epoch)

        logger.info('best epoch {:>2d}, score = {:.5f}'.format(val_best_epoch, val_best_score))

        self.writer.close()

    def update_scheduler(self, epoch):
        last_lr = self.scheduler.get_last_lr()[0]
        self.writer.add_scalar('lr', last_lr, epoch)
        self.scheduler.step()

    def train_epoch(self, epoch, train_loader):
        
        self.model.train()

        print_every=-1

        for iter_num, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(self.device, dtype=torch.float32)
            y_batch = y_batch.to(self.device, dtype=torch.long)

            y_batch = y_batch.squeeze()

            # if iter_num == 1:
            #     print(y_batch)
            #     print(x_batch)
            #     print(self.model.fc.weight)

            output = self.model(x_batch)
            loss = self.criterion(output, y_batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_item = loss.item()

            self.writer.add_scalar('train/loss', loss_item, epoch*len(train_loader)+iter_num)

            if print_every > 0 and iter_num % print_every == 0:
                logger.info('iter: {:>4d}, loss = {:.5f}'.format(iter_num, loss_item))

    def predict(self, loader):
        model_save_name = f'{self.config.model.model_name}.pth'
        self.model.load_state_dict(torch.load(model_save_name))
        self.model.eval()
        self.val_epoch(0, loader)

    def val_epoch(self, epoch, val_loader):

        self.model.eval()

        y_true = []
        outputs = []
        losses = []

        with torch.no_grad():
            for _, (x_batch, y_batch) in enumerate(val_loader):
                x_batch = x_batch.to(self.device, dtype=torch.float32)
                y_batch = y_batch.to(self.device, dtype=torch.long)

                # y_batch = y_batch.squeeze()
                
                output = self.model(x_batch)
                loss = self.criterion(output, y_batch.squeeze())

                outputs.append(output)
                y_true.append(y_batch)
                losses.append(loss.item())


        outputs = torch.cat(outputs, 0)
        y_true = torch.cat(y_true, 0)

        y_prob = outputs.softmax(dim=-1)
        y_pred = torch.argmax(y_prob, 1)

        y_prob = y_prob.detach().cpu().numpy()

        # print(outputs.shape, y_true.shape, y_prob.shape, y_pred.shape)

        auc, acc = self.evaluator.evaluate(y_prob)

        logger.info('epoch: {:>4d}, auc = {:.5f}, acc = {:.5f}'.format(epoch, auc, acc))

        # correct_samples = torch.sum(y_pred == y_true.squeeze())
        # print(correct_samples, y_pred.shape[0])
        # accuracy = float(correct_samples) / y_pred.shape[0]
        # print(accuracy)

        self.writer.add_scalar('val/acc', acc, epoch)
        self.writer.add_scalar('val/auc', auc, epoch)
        self.writer.add_scalar('val/loss', np.mean(losses), epoch)

        score = auc
            
        return score

