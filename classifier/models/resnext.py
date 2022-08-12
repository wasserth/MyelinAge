import sys
from pathlib import Path
p_dir = str(Path(__file__).absolute().parents[2])
if p_dir not in sys.path: sys.path.insert(0, p_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.pytorch_utils import Mish, AdaptiveConcatPool2d, Flatten

# This is from iafoss code
class ResNext(nn.Module):
    def __init__(self, arch='resnext50_32x4d_ssl', n=6):
        super().__init__()
        m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(AdaptiveConcatPool2d(), Flatten(), nn.Linear(2 * nc, 512),
                                  Mish(), nn.BatchNorm1d(512), nn.Dropout(0.5), nn.Linear(512, n))

    def forward(self, *x):
        shape = x[0].shape
        n = len(x)
        x = torch.stack(x, 1).view(-1, shape[1], shape[2], shape[3])
        # x: bs*N x 3 x 128 x 128
        x = self.enc(x)
        # x: bs*N x C x 4 x 4
        shape = x.shape
        # concatenate the output for tiles into a single map
        x = x.view(-1, n, shape[1], shape[2], shape[3]).permute(0, 2, 1, 3, 4).contiguous() \
            .view(-1, shape[1], shape[2] * n, shape[3])
        # x: bs x C x N*4 x 4
        x = self.head(x)
        # x: bs x n
        return x

