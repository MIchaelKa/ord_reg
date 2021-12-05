
import torch

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

import logging
logger = logging.getLogger(__name__)

class BaseEvaluator():
    def detect_inconsistency(self, y_prob):
        inconsistency = ((y_prob[:, :-1] - y_prob[:, 1:]) < 0).sum(1)
        total = inconsistency.sum()
        count = (inconsistency > 0).sum()
        logger.info(f'Inconsistency in predictions: {count}/{y_prob.shape[0]}, total: {total}')

    def compute_score(self, y_true, y_pred):
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()

        acc = accuracy_score(y_true, y_pred)
        qwk = cohen_kappa_score(y_pred, y_true, weights='quadratic')
        cm = confusion_matrix(y_true, y_pred)
            
        return acc, qwk, cm

class BaselineEvaluator(BaseEvaluator):
    def evaluate(self, outputs, y_true):
        y_prob = outputs.softmax(dim=-1)
        y_pred = torch.argmax(y_prob, 1)
        self.detect_inconsistency(y_prob)   
        return self.compute_score(y_true, y_pred)

class LabelBinEvaluator(BaseEvaluator):
    def evaluate(self, outputs, y_true):
        y_prob = outputs.sigmoid()
        self.detect_inconsistency(y_prob)

        y_pred = (outputs > 0).sum(1)
        # Different ways of computing the same thing,
        # it can be usefull if we want to tune the threshold
        # y_pred = (y_prob > 0.5).sum(1)
        # y_pred = y_prob.round().sum(1)

        y_true = y_true.sum(1)
        return self.compute_score(y_true, y_pred)

class CoralEvaluator(BaseEvaluator):
    def evaluate(self, outputs, y_true):
        y_prob = outputs.sigmoid()
        self.detect_inconsistency(y_prob)

        y_pred = y_prob.sum(1).round()

        y_true = y_true.sum(1)
        return self.compute_score(y_true, y_pred)
