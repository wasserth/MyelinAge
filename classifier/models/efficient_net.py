import sys
from pathlib import Path
p_dir = str(Path(__file__).absolute().parents[2])
if p_dir not in sys.path: sys.path.insert(0, p_dir)

import os
from os.path import join
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from libs.pytorch_utils import AdaptiveConcatPool2d


class enet(nn.Module):
    def __init__(self, backbone, in_channels, out_channels,
                 concat_pool=False, dropout=0.2, pretrained=True):
        from efficientnet_pytorch import model as enet
        from efficientnet_pytorch import EfficientNet

        super(enet, self).__init__()

        if pretrained:
            self.enet = EfficientNet.from_pretrained(backbone, in_channels=in_channels, num_classes=out_channels)
        else:
            self.enet = EfficientNet.from_name(backbone, in_channels=in_channels, num_classes=out_channels)
            
        if concat_pool:
            # see EfficientNet.forward for how it is applied
            self.enet._avg_pooling = AdaptiveConcatPool2d(1)  # will double features dim -> will anyways get flattened
            nr_features_in = self.enet._fc.in_features * 2  # needs 500MB more GPU mem
        else:
            # Default is nn.AdaptiveAvgPool2d(1): Will reduce spatial dimension to 1x1
            # -> only the feature maps will go into FC layer
            # Shape before pooling: [1, 1280, 96, 96]
            nr_features_in = self.enet._fc.in_features  # 1280

        self.myfc = nn.Linear(nr_features_in, out_channels)  # this will be the new FC layer

        self.enet._fc = nn.Identity()  # replace full connected layer with dummy

        self.enet._dropout = nn.Dropout(dropout)


    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x
