import numpy as np
import torch
from torch import nn

class BasicCNN(nn.Module):
    """
    Network from here: https://github.com/moboehle/Pytorch-LRP/blob/master/jrieke/models.py
    lr: 0.0001
    weight decay: 0.0001
    https://arxiv.org/abs/1903.07317
    """

    def __init__(self, crop_size, num_classes=2, in_chans=1, dropout=0.4, dim=2):

        if min(crop_size) < 40:
            raise ValueError("BasicCnn does not work with images with any dimension smaller than 40.")

        if dim == 2:
            Conv = nn.Conv2d
            BatchNorm = nn.BatchNorm2d
            MaxPool = nn.MaxPool2d
            AdaptiveAvgPool = nn.AdaptiveAvgPool2d
            avg_pool_size = [8, 8]
        else:
            Conv = nn.Conv3d
            BatchNorm = nn.BatchNorm3d
            MaxPool = nn.MaxPool3d
            AdaptiveAvgPool = nn.AdaptiveAvgPool3d
            # avg_pool_size = [8, 8, 4]  # for isotropic images [8, 8, 8] might be better
            avg_pool_size = [8, 8, 8]

        nn.Module.__init__(self)
        self.Conv_1 = Conv(in_chans, 8, 3)
        self.Conv_1_bn = BatchNorm(8)
        self.Conv_1_mp = MaxPool(2)
        self.Conv_2 = Conv(8, 16, 3)
        self.Conv_2_bn = BatchNorm(16)
        self.Conv_2_mp = MaxPool(2)  # 3
        self.Conv_3 = Conv(16, 32, 3)
        self.Conv_3_bn = BatchNorm(32)
        self.Conv_3_mp = MaxPool(2)
        self.Conv_4 = Conv(32, 64, 3)
        self.Conv_4_bn = BatchNorm(64)
        # self.Conv_4_mp = MaxPool(3)
        # self.dense_1 = nn.Linear(16384, 128)  # 16384 is the shape for rapmed/pneu 650x650
        self.Conv_4_ap = AdaptiveAvgPool(avg_pool_size)  # 6
        self.dense_1 = nn.Linear(64*np.prod(avg_pool_size), 128)
        self.dense_2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        x = self.Conv_1_mp(x)
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        x = self.Conv_2_mp(x)
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        x = self.Conv_3_mp(x)
        x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
        # x = self.Conv_4_mp(x)
        # x = x.view(x.size(0), -1)
        # print(f"Shape before avg pooling: {x.shape}")
        x = self.Conv_4_ap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.relu(self.dense_1(x))
        x = self.dropout2(x)
        x = self.dense_2(x)
        # Note that no sigmoid is applied here, because the network is used in combination with BCEWithLogitsLoss,
        # which applies sigmoid and BCELoss at the same time to make it numerically stable.
        return x