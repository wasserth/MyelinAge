import sys
from pathlib import Path
p_dir = str(Path(__file__).absolute().parents[1])
if p_dir not in sys.path: sys.path.insert(0, p_dir)

import random
import time
import os

import torch
from torch.nn import functional as F
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from scipy import ndimage

from torch.utils.data import DataLoader, Dataset, random_split

from classifier.utils import get_slices_from_3D_img, get_slices_from_3D_img_subset, img_3d_to_tiles_2d


def nifti_loader(file_path):
    return nib.load(file_path).get_fdata()


class NiftiDataset(Dataset):
    def __init__(self, data_path, subjects, labels, transform=None, dim=3, nr_slices=3,
                 multi_orientation=0, crop_size=(170,170), slice_subset=0, img_files="ct_05.npy", 
                 loss="ce", nr_classes=2, deterministic=False, tiles=False, 
                 file_loader=nifti_loader, tta_nr=1, tiles_subsample=3, tiles_start=4, zoom=[1.0, 1.0, 1.0],
                 slice_orientation="z", unpack_to_npy=True):
        self.data_path = data_path
        self.labels = labels
        self.transform = transform
        self.dim = dim
        self.subjects = subjects
        self.nr_slices = nr_slices
        self.multi_orientation = multi_orientation
        self.crop_size = crop_size
        self.slice_subset = slice_subset
        self.img_files = img_files
        self.loss = loss
        self.nr_classes = nr_classes
        self.deterministic = deterministic
        self.tiles = tiles
        self.file_loader=file_loader
        self.tta_nr = tta_nr
        self.tiles_subsample = tiles_subsample
        self.tiles_start = tiles_start
        self.zoom = zoom
        self.slice_orientation = slice_orientation
        self.unpack_to_npy = unpack_to_npy

    def get_data(img_path):
        # todo: move loading of x here so we can reuse it in inference
        pass

    def __getitem__(self, index: int):
        subject = self.subjects[index]
        subject_path = (self.data_path / subject).resolve()  # resolve allows ".." in the path

        rnd_size = None  # initialize here to make the same for all modalities
        x_all = []
        for img_file in self.img_files:  # iterate over modalities
            if os.path.exists(subject_path / f"{img_file.split('.')[0]}.npy"):
                # for 2D loading entire 3D volume is very slow
                # load from npy with mmap=r: will only read the slices that are really required -> a lot faster for 2D
                mmap_mode = "r" if self.dim == 2 else None
                x = np.load(subject_path / f"{img_file.split('.')[0]}.npy", mmap_mode=mmap_mode) 
            else:
                # if self.unpack_to_npy:
                #     print("WARNING: loading nifti file. Should load npy. Why was npy not created?")
                x = self.file_loader(subject_path / img_file)
            y = self.labels[index]

            # Work with smaller image size
            if np.any(np.array(self.zoom) != 1.0):
                # print(f"INFO: reducing image size by zoom {self.zoom}")
                # could try higher order resampling here
                x = ndimage.zoom(x, self.zoom, order=0)

            if rnd_size is None:
                if self.slice_orientation == "x":
                    rnd_size = int(x.shape[0] / 8)
                elif self.slice_orientation == "y":
                    rnd_size = int(x.shape[1] / 8)
                elif self.slice_orientation == "z":
                    rnd_size = int(x.shape[2] / 8)
                # this is not affected by np multiprocessing random seed bug, because is native python random
                rnd = 0 if self.deterministic else random.randint(-rnd_size, rnd_size)

            if self.dim == 2:
                if self.slice_subset:
                    seg = nib.load(subject_path / "lung_pleu_05.nii.gz").get_fdata() == 2
                    x = get_slices_from_3D_img_subset(x, seg, self.nr_slices, rnd, rnd_size)
                else:                    
                    if self.tiles:
                        # three_quarter = x.shape[2] - x.shape[2]//4
                        # x = img_3d_to_tiles_2d(x[:,:,abs(rnd):three_quarter:3])[None,...]  # used for brainage
                        # four_fifth = x.shape[2] - x.shape[2]//5
                        # x = img_3d_to_tiles_2d(x[:,:,abs(rnd):four_fifth:1])[None,...]
                        # x = img_3d_to_tiles_2d(x[:,:,abs(rnd):-2:3])[None,...]  # use this in the long-term
                        first_quarter = x.shape[2]//self.tiles_start
                        three_quarter = x.shape[2] - x.shape[2]//self.tiles_start
                        x = img_3d_to_tiles_2d(x[:,:,abs(rnd):three_quarter:self.tiles_subsample])[None,...]
                        # x = img_3d_to_tiles_2d(x[:,:,first_quarter+abs(rnd):three_quarter:self.tiles_subsample])[None,...]
                    else:
                        x = get_slices_from_3D_img(x, self.nr_slices, self.multi_orientation,
                                                   self.crop_size, rnd, rnd_size, self.slice_orientation)
            else:
                x = x[None,...]  # add channel dim

            x_all.append(x)

        x = np.array(x_all)
        # Merge files dim (e.g. multiple modalities) with channel dim to combined channel dim
        if self.dim == 2:
            x = x.reshape([x.shape[0]*x.shape[1], x.shape[2], x.shape[3]])
        else:
            x = x.reshape([x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4]])
            # x = x[:, :, : , ::2]  # for GRU/LSTM

        x_all = []
        for tta_idx in range(self.tta_nr):
            x_tf = self.transform(x) if self.transform else x
            # x_tf in pytorch cpu tensor
            x_tf = x_tf.float()
            x_all.append(x_tf)
        x = x_all[0] if self.tta_nr == 1 else x_all

        if self.loss == "ce":
            label = int(y)
        elif self.loss == "bce":
            # one less than classes, because all 0 is first class
            label = np.zeros(self.nr_classes-1).astype(np.float32)
            label[:int(y)] = 1.
        elif self.loss.startswith("mse"):
            label = np.array(y).astype(np.float32)

        return x, label

    def __len__(self) -> int:
        return len(self.subjects)


if __name__ == "__main__":
    data_path = Path("/mnt/nvme/data/brainage/classify_5_groups")
    subjects = [p.name for p in data_path.glob("*")]
    # from classifier.transformations import get_transforms
    # tfs_train, tfs_val = get_transforms(hparams)
    # dataset = NiftiDataset(subjects, transform=tfs_train, dim=3)
    # train_loader = DataLoader(dataset, batch_size=2, num_workers=3)  # pin_memory=True
    # for batch in tqdm(train_loader):
    #     x, y = batch
    #     print(x.shape)
