import medmnist
import numpy as np

class BaselineRetinaMNIST(medmnist.dataset.RetinaMNIST):
    '''
    Helper class to get rid of additional dimension in the target field.
    It helps to avoid calling squeeze() in the future and reuse the common codebase.
    '''
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        target = target[0] # target: np.array of L (L=1 for single-label)
        return img, target


class LabelBinRetinaMNIST(medmnist.dataset.RetinaMNIST):
    '''
    Implements label binning for RetinaMNIST

    Preprocess the target data as the following:
        Grade – 0: label = [0,0,0,0]
        Grade – 1: label = [1,0,0,0]
        Grade – 2: label = [1,1,0,0]
        Grade – 3: label = [1,1,1,0]
        Grade – 4: label = [1,1,1,1]
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        n_classes = len(self.info['label'])
        self.n_bins = n_classes - 1
        
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        target = target[0] # target: np.array of L (L=1 for single-label)
        label = np.zeros(self.n_bins).astype(np.float32)
        label[:target] = 1.
        return img, label