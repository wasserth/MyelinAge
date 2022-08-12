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


class cbr_tiny_hydra(nn.Module):
    def __init__(self, crop_size, dim, num_classes, in_chans, nr_filt):
        super(cbr_tiny_hydra, self).__init__()

        from classifier.models.cbr_tiny import CbrTiny
        
        self.backbone_1 = CbrTiny(crop_size, dim=dim, num_classes=num_classes,
                                  in_chans=1, nr_filt=nr_filt, return_features=True)
        self.backbone_2 = CbrTiny(crop_size, dim=dim, num_classes=num_classes,
                                  in_chans=1, nr_filt=nr_filt, return_features=True)
        self.backbone_3 = CbrTiny(crop_size, dim=dim, num_classes=num_classes,
                                  in_chans=1, nr_filt=nr_filt, return_features=True)
        self.backbone_4 = CbrTiny(crop_size, dim=dim, num_classes=num_classes,
                                  in_chans=1, nr_filt=nr_filt, return_features=True)
        avg_pool_size = [4, 4, 4]
        final_nr_filt = nr_filt*8
        nr_arms = 4
        self.fc = nn.Linear(final_nr_filt*np.prod(avg_pool_size)*nr_arms, num_classes)

    def forward(self, x):
        x_1 = x[:, 0:1, :, :]   # [bs, 1, x, y]
        x_2 = x[:, 1:2, :, :]
        x_3 = x[:, 2:3, :, :]
        x_4 = x[:, 3:4, :, :]
        # print(f"x_1.shape = {x_1.shape}")
        x_1 = self.backbone_1(x_1)  # [bs, 4096]
        # print(f"x_1 b.shape = {x_1.shape}")
        x_2 = self.backbone_2(x_2)
        x_3 = self.backbone_3(x_3)
        x_4 = self.backbone_4(x_4)
        x = torch.cat([x_1, x_2, x_3, x_4], dim=1)  # [bs, 4*4096]
        # print(f"x cat.shape = {x.shape}")
        x = self.fc(x)
        return x
