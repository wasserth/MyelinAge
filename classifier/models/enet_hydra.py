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
import numpy as np


class enet_hydra(nn.Module):
    def __init__(self, backbone, in_channels, out_channels,
                 concat_pool=False, dropout=0.2, pretrained=True):
        super(enet_hydra, self).__init__()
        import timm
        
        self.backbone_1 = timm.create_model(backbone, pretrained=pretrained,
                                            num_classes=0, in_chans=1,
                                            drop_rate=dropout, global_pool='avg')
        self.backbone_2 = timm.create_model(backbone, pretrained=pretrained,
                                            num_classes=0, in_chans=1,
                                            drop_rate=dropout, global_pool='avg')
        avg_pool_size = [1, 1]
        final_nr_filt = 1280

        self.fc = nn.Linear(final_nr_filt*np.prod(avg_pool_size)*2, out_channels)

    def forward(self, x):
        x_1 = x[:, 0:1, :, :]   # [bs, 1, x, y]
        x_2 = x[:, 1:2, :, :]
        # print(f"x_1.shape = {x_1.shape}")
        # print(f"x_2.shape = {x_2.shape}")
        x_1 = self.backbone_1(x_1)  # [bs, 1280]
        # print(f"x_1 b.shape = {x_1.shape}")
        x_2 = self.backbone_2(x_2)
        # print(f"x_2 b.shape = {x_2.shape}")
        x = torch.cat([x_1, x_2], dim=1)  # [bs, 2560]
        # print(f"x cat.shape = {x.shape}")
        x = self.fc(x)
        return x
