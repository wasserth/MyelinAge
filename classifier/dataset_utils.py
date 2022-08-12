import sys
from pathlib import Path
p_dir = str(Path(__file__).absolute().parents[1])
if p_dir not in sys.path: sys.path.insert(0, p_dir)

import random
import time

import torch
import nibabel as nib
import numpy as np
import pandas as pd


def brainage_month_to_class(month):
    if type(month) == np.ndarray:
        out = month.copy()
    else:
        out = month.clone()  # by using clone out is on same device as month

    # 5 classes
    # out[:] = 4
    # out[month <= 24] = 3
    # out[month <= 12] = 2
    # out[month <= 9] = 1
    # out[month <= 4] = 0

    # 4 classes
    out[:] = 3
    out[month <= 24] = 2
    out[month <= 12] = 1
    out[month <= 4] = 0
    
    return out.astype(np.uint8) if type(month) == np.ndarray else out.int()
