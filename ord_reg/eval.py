
import torch

from sklearn.metrics import accuracy_score, cohen_kappa_score

import logging
logger = logging.getLogger(__name__)

class BaseEvaluator():
    def __init__(self, writer):
        self.writer = writer

    def compute_score(self, epoch, y_true, y_pred):
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()

        acc = accuracy_score(y_true, y_pred)
        qwk = cohen_kappa_score(y_pred, y_true, weights='quadratic')

        logger.info('Epoch: {:>2d}, acc = {:.5f}, qwk = {:.5f}'.format(epoch, acc, qwk))

        if epoch != -1:
            self.writer.add_scalar('val/acc', acc, epoch)
            self.writer.add_scalar('val/qwk', qwk, epoch)

        return qwk

class BaselineEvaluator(BaseEvaluator):
    def evaluate(self, epoch, outputs, y_true):
        y_prob = outputs.softmax(dim=-1)
        y_pred = torch.argmax(y_prob, 1)

        return self.compute_score(epoch, y_true, y_pred)

class LabelBinEvaluator(BaseEvaluator):
    def evaluate(self, epoch, outputs, y_true):
        # TODO: > 0.5 after sigmoid is makes sense?
        # TODO: sum until first zero
        y_pred = outputs.sigmoid().sum(1).round()
        y_true = y_true.sum(1)

        return self.compute_score(epoch, y_true, y_pred)
