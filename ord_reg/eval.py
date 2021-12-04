
import torch

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix


class BaseEvaluator():
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

        return self.compute_score(y_true, y_pred)

class LabelBinEvaluator(BaseEvaluator):
    def evaluate(self, outputs, y_true):
        # TODO: > 0.5 after sigmoid is makes sense?
        # TODO: sum until first zero
        y_pred = outputs.sigmoid().sum(1).round()
        y_true = y_true.sum(1)

        return self.compute_score(y_true, y_pred)
