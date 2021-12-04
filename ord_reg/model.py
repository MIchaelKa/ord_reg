import torch.nn as nn
import timm

from omegaconf import DictConfig
from hydra.utils import instantiate

def get_encoder(model_name, pretrained):
    model = timm.create_model(model_name, pretrained=pretrained)
    layers = list(model.children())   
    in_features = layers[-1].in_features
    encoder = nn.Sequential(*layers[:-1])
    return encoder, in_features

def get_model(cfg: DictConfig):
    encoder, in_features = get_encoder(**cfg.encoder)
    fc = instantiate(cfg.last_fc, in_features=in_features)
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