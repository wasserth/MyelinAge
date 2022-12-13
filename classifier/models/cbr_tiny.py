import numpy as np
import torch
from torch import nn

class CbrTiny(nn.Module):
    """
    Network from here: https://arxiv.org/pdf/1902.07208.pdf
    lr: 0.001
    bs: 32
    optimizer: Adam on Retina; SGD+Momentum with 0.9 on ChestXRay

    Main difference to basic_cnn:
    - only 1 dense layer (instead of 2)
    - kernel size 5 for conv (instead of 3)
    - kernel size 3 for maxpool (instead of 2)

    For my small datasets nr_filt > 8 not really necessary.

    With nr_filt=8: roughly 4MB of weights. basiccnn has roughly 26MB, mainly because of 
    2 instead of 1 dense layers.

    Bigger receptive window (5 here compared to 3 for basic_cnn (or 3 vs 2 in maxpool) will make
    image shrink even faster)

    For 3D 2-3x faster than 2d efficientnet_b0 with tiles
    """

    def __init__(self, crop_size, num_classes=2, in_chans=1, dim=2, nr_filt=8, return_features=False):

        if min(crop_size) < 40:
            raise ValueError("cbr_tiny does not work with images with any dimension smaller than 40.")

        self.return_features = return_features

        if dim == 2:
            Conv = nn.Conv2d
            BatchNorm = nn.BatchNorm2d
            MaxPool = nn.MaxPool2d
            AdaptiveAvgPool = nn.AdaptiveAvgPool2d
            avg_pool_size = [4, 4]
        else:
            Conv = nn.Conv3d
            BatchNorm = nn.BatchNorm3d
            MaxPool = nn.MaxPool3d
            AdaptiveAvgPool = nn.AdaptiveAvgPool3d
            # Rough heuristic: 
            # shape before avp_pool: s = int((input_size / 2**nr_maxpool) - 5)
            # s should be slightly bigger than avg_pool_size
            avg_pool_size = [4, 4, 4]  # good for 3d isotropic with 4x maxpool
            # avg_pool_size = [8, 8, 1]  # good for 3d anisotropic with 3x maxpool

        nn.Module.__init__(self)
        
        # Good for 3d isotropic
        # kernel_size = 5
        # kernel_size_mp = 3
        
        # Better for 3d with small z
        kernel_size = 3
        kernel_size_mp = 2

        self.Conv_1 = Conv(in_chans, nr_filt, kernel_size)
        self.Conv_1_bn = BatchNorm(nr_filt)
        self.Conv_1_mp = MaxPool(kernel_size_mp, 2)
        self.Conv_2 = Conv(nr_filt, nr_filt*2, kernel_size)
        self.Conv_2_bn = BatchNorm(nr_filt*2)
        self.Conv_2_mp = MaxPool(kernel_size_mp, 2)
        self.Conv_3 = Conv(nr_filt*2, nr_filt*4, kernel_size)
        self.Conv_3_bn = BatchNorm(nr_filt*4)
        self.Conv_3_mp = MaxPool(kernel_size_mp, 2)
        self.Conv_4 = Conv(nr_filt*4, nr_filt*8, kernel_size)
        self.Conv_4_bn = BatchNorm(nr_filt*8)
        # self.Conv_4_mp = MaxPool(kernel_size_mp, 2)

        # Jump to AvgPool should not be too big (in terms of shape) to avoid performance loss
        # (12->4: performance loss, 12->8: ok, 12->maxpool to 5->4: ok)
        #
        # TIMM efficientnet afaik uses AdaptiveAvgPool.
        # (see SelectAdaptivePool2d(nn.Module) in timm.models.layers.adaptive_avgmax_pool.py)
        self.Conv_4_ap = AdaptiveAvgPool(avg_pool_size)  # specify output shape (instead of stride+kernel size)
        if not self.return_features:
            self.dense_1 = nn.Linear(nr_filt*8*np.prod(avg_pool_size), num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        x = self.Conv_1_mp(x)
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        x = self.Conv_2_mp(x)
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        x = self.Conv_3_mp(x)
        x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
        # x = self.Conv_4_mp(x)
        # print(f"Shape before avg pooling: {x.shape}")
        x = self.Conv_4_ap(x)
        x = torch.flatten(x, 1)
        if not self.return_features:
            x = self.dense_1(x)
        # Do not use any final activation function, because often part of loss function to make it
        # numerically more stable.
        return x