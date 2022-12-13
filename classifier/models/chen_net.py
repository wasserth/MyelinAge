import numpy as np
import torch
from torch import nn

class ChenNet(nn.Module):
    """
    Network from here: https://pubs.rsna.org/doi/10.1148/radiol.211860

    init: He
    epochs: 50
    bs: 16
    optimizer: adam
    dropout: 0.5
    lr: 0.001 at start then decaying

        stopping = EarlyStopping(patience=30)
        reduce_lr = ReduceLROnPlateau(
            factor=0.1,
            patience=8,
            min_lr=configs['learning_rate'] * 0.001)

    input image size: 86, 110, 78   (roughly downsampled by 2x from original size)

    (my images after resampling to 1.5mm isotropic: 86, 94, 84)

    Data was put into 10 bins by age. Then during training each bin was sampled equally often (inverse
    frequency).

    Data augmentation: 
    - radom rotation by up to 10 degree
    - translation up to 3 voxels
    - flipping along sagittal plane
    """
    def __init__(self, crop_size, num_classes=2, in_chans=1, dropout=0.5, dim=2, nr_filt=16):

        # if min(crop_size) < 40:
        #     raise ValueError("BasicCnn does not work with images with any dimension smaller than 40.")

        if dim == 2:
            Conv = nn.Conv2d
            BatchNorm = nn.BatchNorm2d
            MaxPool = nn.MaxPool2d
            AdaptiveAvgPool = nn.AdaptiveAvgPool2d
            avg_pool_size = [2, 2]
        else:
            Conv = nn.Conv3d
            BatchNorm = nn.BatchNorm3d
            MaxPool = nn.MaxPool3d
            AdaptiveAvgPool = nn.AdaptiveAvgPool3d
            avg_pool_size = [2, 2, 2]

        nn.Module.__init__(self)
        self.Conv_1 = Conv(in_chans, nr_filt, 3, 1, "same")
        self.Conv_1_bn = BatchNorm(nr_filt)
        self.Conv_1_mp = MaxPool(2)
        self.Conv_2 = Conv(nr_filt, nr_filt*2, 3, 1, "same")
        self.Conv_2_bn = BatchNorm(nr_filt*2)
        self.Conv_2_mp = MaxPool(2)
        self.Conv_3 = Conv(nr_filt*2, nr_filt*4, 3, 1, "same")
        self.Conv_3_bn = BatchNorm(nr_filt*4)
        self.Conv_3_mp = MaxPool(2)
        self.Conv_4 = Conv(nr_filt*4, nr_filt*8, 3, 1, "same")
        self.Conv_4_bn = BatchNorm(nr_filt*8)
        self.Conv_4_mp = MaxPool(2)
        self.Conv_5 = Conv(nr_filt*8, nr_filt*16, 3, 1, "same")
        self.Conv_5_bn = BatchNorm(nr_filt*16)
        self.Conv_5_mp = MaxPool(2)

        self.Conv_6 = Conv(nr_filt*16, nr_filt*4, 1, 1, "same")
        self.Conv_6_bn = BatchNorm(nr_filt*4)
        self.Conv_6_ap = AdaptiveAvgPool(avg_pool_size)  # 6

        self.dropout = nn.Dropout(dropout)
        self.dense_1 = nn.Linear(nr_filt*4*np.prod(avg_pool_size), 512)
        self.dense_2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        x = self.Conv_1_mp(x)
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        x = self.Conv_2_mp(x)
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        x = self.Conv_3_mp(x)
        x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
        x = self.Conv_4_mp(x)
        x = self.relu(self.Conv_5_bn(self.Conv_5(x)))
        x = self.Conv_5_mp(x)
        x = self.relu(self.Conv_6_bn(self.Conv_6(x)))
        # print(f"Shape before avg pooling: {x.shape}")
        x = self.Conv_6_ap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.relu(self.dense_1(x))
        x = self.dense_2(x)
        # Note that no sigmoid is applied here, because the network is used in combination with BCEWithLogitsLoss,
        # which applies sigmoid and BCELoss at the same time to make it numerically stable.
        return x