import torch.nn as nn
import timm

# class ImageModel(nn.Module):
#     def __init__(self, model_name, pretrained, n_classes):
#         super().__init__()
#         self.model = timm.create_model(model_name, pretrained=pretrained)
#         in_features = self.model.fc.in_features
#         self.model.fc = nn.Linear(in_features, n_classes)

#     def forward(self, x):
#         x = self.model(x)
#         return x

class ImageModel(nn.Module):
    def __init__(self, model_name, pretrained, n_classes):
        super().__init__()
        
        model = timm.create_model(model_name, pretrained=pretrained)
        layers = list(model.children())   
        in_features = layers[-1].in_features

        self.backbone = nn.Sequential(*layers[:-1])
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x