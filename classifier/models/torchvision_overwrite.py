import torch
import torch.nn as nn
import torchvision.models as torch_models


class TorchVisionOverwrite(nn.Module):
    def __init__(self, arch='vgg16', pretrained=False, num_classes=1000, in_chans=3):
        super().__init__()

        model = torch_models.__dict__[arch](pretrained=pretrained)

        # Layers are per default initializied (e.g. Linear per default with kaiming_uniform_)

        # Replace first layer
        out_channels = list(model.features.children())[0].out_channels
        model.features = nn.Sequential(nn.Conv2d(in_chans, out_channels, kernel_size=3, padding=1),
                                       *list(model.features.children())[1:])

        # Replace last layer
        in_features = list(model.classifier.children())[-1].in_features
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1],
                                         nn.Linear(in_features, num_classes))
        self.model = model

    def forward(self, x):
        return self.model(x)
