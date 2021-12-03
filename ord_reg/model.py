import torch.nn as nn
import timm

class ImageModel(nn.Module):
    def __init__(self, model_name, pretrained, out_dim):
        super().__init__()
        
        model = timm.create_model(model_name, pretrained=pretrained)
        layers = list(model.children())   
        in_features = layers[-1].in_features

        self.backbone = nn.Sequential(*layers[:-1])

        self.fc = nn.Linear(in_features, out_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x