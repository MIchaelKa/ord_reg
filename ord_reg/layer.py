import torch
import torch.nn as nn

class CoralLayer(nn.Module):
    '''
    Implements CORAL layer.
    Look at arxiv paper for more details: https://arxiv.org/abs/1901.07884
    This class is an adapted version from
    https://github.com/Raschka-research-group/coral-pytorch
    
    Parameters:
    - in_features: int
        Number of features for the inputs to the forward method, which
        are expected to have shape (num_examples, in_features).

    - out_features: int
        Number of bins, it equals to the (num_classes - 1).

    - preinit_bias: bool (default=True)
        If true, it will pre-initialize the biases to descending values in
        [0, 1] range instead of initializing it to all zeros. This pre-
        initialization scheme results in faster learning and better
        generalization performance in practice.
    '''

    def __init__(self, in_features, out_features, preinit_bias=True):
        super().__init__()
        
        self.coral_weights = torch.nn.Linear(in_features, 1, bias=False)
        
        if preinit_bias:
            coral_bias = torch.arange(out_features, 0, -1).float() / out_features
        else:
            coral_bias = torch.zeros(out_features).float()
        
        self.coral_bias = torch.nn.Parameter(coral_bias)

    def forward(self, x):
        return self.coral_weights(x) + self.coral_bias