import torch
import torch.nn as nn
import torch.nn.functional as F

class CoralLoss(nn.Module):
    '''
    Computes the CORAL loss
    Look at arxiv paper for more details: https://arxiv.org/abs/1901.07884
    This class is an adapted version from
    https://github.com/Raschka-research-group/coral-pytorch

    Parameters:
    - input : torch.tensor, shape (num_examples, num_classes-1)
        Output of the CORAL layer.

    - target : torch.tensor, shape (num_examples, num_classes-1)
        True labels represented as label bins.

    Returns:
    - loss : torch.tensor
        A torch.tensor containing a single loss value.
    '''
    def forward(self, input, target):
        loss = F.logsigmoid(input) * target + (F.logsigmoid(input) - input) * (1 - target)
        loss = (-torch.sum(loss, dim=1))
        return loss.mean()