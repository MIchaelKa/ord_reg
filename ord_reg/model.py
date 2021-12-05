import torch
import torch.nn as nn
import timm

from omegaconf import DictConfig
from hydra.utils import instantiate

def get_model(cfg: DictConfig):
    encoder = instantiate(cfg.encoder)
    fc = instantiate(cfg.last_fc, in_features=encoder.out_features)
    model = ImageModel(encoder, fc)
    return model

class ImageModel(nn.Module):
    def __init__(self, backbone, fc):
        super().__init__()
        self.backbone = backbone
        self.fc = fc

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class TimmEncoder(nn.Module):
    def __init__(self, model_name, pretrained):
        super().__init__()

        model = timm.create_model(model_name, pretrained=pretrained)
        layers = list(model.children())
        
        self.out_features = layers[-1].in_features
        self.backbone = nn.Sequential(*layers[:-1])

    def forward(self, x):       
        x = self.backbone(x)
        return x
        

class SimpleEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        channels = [3, 64, 128, 256]

        self.backbone = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(channels[-1]*3*3, 512),
            # nn.ReLU(),
        )

        self.out_features = 512
        
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)      
        return x