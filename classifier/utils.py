import sys
from pathlib import Path
p_dir = str(Path(__file__).absolute().parents[1])
if p_dir not in sys.path: sys.path.insert(0, p_dir)

import os
import pickle
import random
import math
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import nibabel as nib
from monai.transforms import ResizeWithPadOrCrop
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt

from libs.utils import get_bbox_from_mask, flatten


def _get_slice_idxs(mid, nr_slices, offset=25):
    if nr_slices == 1:
        slice_idxs = [mid]
    if nr_slices == 3:
        slice_idxs = [mid-offset, mid, mid+offset]
    if nr_slices == 5:
        slice_idxs = [mid-offset, mid-int(offset/2), mid, mid+int(offset/2), mid+offset]
    return slice_idxs

    
def get_slices_from_3D_img(x, nr_slices, multi_orientation, crop_size, rand_int, offset, orientation="z"):
    """
    x: numpy array with dimensions [x,y,z]
    nr_slices: how many slices to select in each orientation
    multi_orientation: if true samples slices not only from z orientation but from x & y & z orientation
                       resulting number of channels: 3*nr_slices
    return: numpy array with dimensions [c,w,h]
    """

    if multi_orientation:
        cropper = ResizeWithPadOrCrop(crop_size)
        mid = (np.array(x.shape) / 2).astype(np.int)
        mid += np.array([rand_int, rand_int, rand_int])

        ns = nr_slices
        slices = np.zeros((3*ns,) + crop_size)
        slices[0*ns:1*ns, :, :] = cropper(x[_get_slice_idxs(mid[0], nr_slices, offset), :, :].transpose(0,1,2))
        slices[1*ns:2*ns, :, :] = cropper(x[:, _get_slice_idxs(mid[1], nr_slices, offset), :].transpose(1,0,2))
        slices[2*ns:3*ns, :, :] = cropper(x[:, :, _get_slice_idxs(mid[2], nr_slices, offset)].transpose(2,0,1))
    else:
        if orientation == "x":
            mid = int(x.shape[0] / 2)  # sagittal (side view)
            if x.shape[0] > 1:  # do not add random factor for 2D images
                mid += rand_int
            slices = x[_get_slice_idxs(mid, nr_slices, offset), :, :]
        elif orientation == "y":  # coronal (front view)
            mid = int(x.shape[1] / 2) 
            if x.shape[1] > 1:  # do not add random factor for 2D images
                mid += rand_int
            slices = x[:, _get_slice_idxs(mid, nr_slices, offset), :].transpose(1,0,2)
        elif orientation == "z":  # axial (top view)
            mid = int(x.shape[2] / 2) 
            if x.shape[2] > 1:  # do not add random factor for 2D images
                mid += rand_int
            slices = x[:, :, _get_slice_idxs(mid, nr_slices, offset)].transpose(2,0,1)

    return slices


def get_slices_from_3D_img_subset(x, seg, nr_slices, rand_int):
    """
    like get_slices_from_3D_img() but will derive the middle slice not from the entire image but 
    will take the middle of segmentation, provided in a separate file.

    Does not support multi_orientation or an orientation other than z.
    """
    if seg.sum() > 0:
        bb_x, bb_y, bb_z = get_bbox_from_mask(seg, outside_value=0)
    else:
        bb_x, bb_y, bb_z = [0, seg.shape[0]], [0, seg.shape[1]], [0, seg.shape[2]]

    # x-slices
    # mid = int((bb_x[0] + bb_x[1]) / 2) + rand_int
    # x = x[_get_slice_idxs(mid, nr_slices, 7), :, :].transpose(0,1,2)

    # z-slices
    mid = int((bb_z[0] + bb_z[1]) / 2) + rand_int
    x = x[:, :, _get_slice_idxs(mid, nr_slices, 7)].transpose(2,0,1)
    return x


def img_3d_to_tiles_2d(img):
    sh = img.shape
    h, w = sh[0], sh[1]
    rows = math.ceil(math.sqrt(sh[2]))
    cols = math.ceil(sh[2]/rows)
    data = np.zeros([cols*h, rows*w])

    ctr = 0
    for c_idx in range(cols):
        for r_idx in range(rows):
            if ctr < sh[2]:
                data[c_idx*h:(c_idx*h)+h, r_idx*w:(r_idx*w)+w] = img[:, :, ctr]
            ctr += 1
    return data


def get_meta_df(path, index_col="AccessionNumber"):
    if (path.parent / f"{path.stem}.pkl").exists():
        return pickle.load(open(path.parent / f"{path.stem}.pkl", "rb"))
    else:
        meta = pd.read_excel(path, dtype={index_col: str})
        # meta[index_col] = meta[index_col].astype(str)  # this can not restore leading 0s
        meta = meta.set_index(index_col)
        print("Saving meta data as pickle for faster loading...")
        pickle.dump(meta, open(path.parent / f"{path.stem}.pkl", "wb"))
    return meta


def stratified_kfold_train_val_test(x, y, fold, test_split=False, unit_test=False,
                                    return_idx=False, save_path=None, skip_stratification=False):
    """
    This will do a stratified split into k folds. In contrast to sklearn.StratifiedKFold it can 
    also return a split into train-val-test set (60%-20%-20%). If test_split=False 
    then a train-val set (80%-20%) will be returned.

    returns: x_train, x_val, x_test, y_train, y_val, y_test   
                [if test_split=False: x_test and y_test are empty lists]
    """
    x, y = np.array(x), np.array(y)
    
    nf = 5  # nr of folds
    if nf % 5 != 0: raise ValueError("number of folds must be multiple of 5.")

    if unit_test:
        kf = KFold(nf)
    else:
        # Somehow (Stratified)KFold is not shuffling the data. If not shuffling here, then labels will not be
        # mixed (if they were not mixed in meta.xlsx)
        x, y = shuffle(x, y)
        if skip_stratification:
            kf = KFold(nf, shuffle=True)
        else:
            kf = StratifiedKFold(nf, shuffle=True)  # deterministic because of global seed
    
    splits = []
    for idx, f in enumerate(kf.split(x, y)):
        train_idx, val_idx = f
        splits.append(val_idx.tolist())  # tolist() makes it real non-numpy; needed for json dump

    if save_path is not None:
        if save_path.exists():
            print("Loading existing cv_split")
            splits = json.load(open(save_path, "r"))
        else:
            print("Creating and saving new cv_split")
            print(f"Size of each split: {[len(s) for s in splits]}")
            json.dump(splits, open(save_path, "w"), indent=4)

    if test_split:
        train_size, val_size, test_size = int(nf * 0.6), int(nf * 0.2), int(nf * 0.2)
        train_idxs = flatten([splits[(fold+idx)%nf] for idx in range(train_size)])
        val_idxs = flatten([splits[(fold+train_size+idx)%nf] for idx in range(val_size)])
        test_idxs = flatten([splits[(fold+train_size+val_size+idx)%nf] for idx in range(test_size)])
        if return_idx:
            return np.array(train_idxs).tolist(), np.array(val_idxs).tolist(), np.array(test_idxs).tolist()
        else:
            return x[train_idxs].tolist(), x[val_idxs].tolist(), x[test_idxs].tolist(), y[train_idxs].tolist(), y[val_idxs].tolist(), y[test_idxs].tolist()
    else:
        train_size, val_size = int(nf * 0.8), int(nf * 0.2)
        train_idxs = flatten([splits[(fold+idx)%nf] for idx in range(4)])
        val_idxs = flatten([splits[(fold+train_size+idx)%nf] for idx in range(val_size)])
        if return_idx:
            return np.array(train_idxs).tolist(), np.array(val_idxs).tolist(), []
        else:
            return x[train_idxs].tolist(), x[val_idxs].tolist(), [], y[train_idxs].tolist(), y[val_idxs].tolist(), []
    

def TEST_stratified_kfold_train_val_test():
    
    subjects = [1,2,3,4,5,6,7,8,9,10]
    labels = [True, True, True, True, True, True, False, False, False, False]

    for fold in range(5):
        subjects_train, subjects_val, subjects_test, labels_train, labels_val, labels_test = \
            stratified_kfold_train_val_test(subjects, labels, fold, test_split=True, unit_test=True)  
        print(f"fold: {fold}")
        print(subjects_train)
        print(subjects_val)
        print(subjects_test)


def confusion_matrix_to_figure(confusion_matrix, nr_classes):
    df_cm = pd.DataFrame(confusion_matrix.astype(np.int),
                         index=range(nr_classes),
                         columns=range(nr_classes))
    plt.figure(figsize = (5,4))
    sns.set(font_scale=1.2)
    fig_ = sns.heatmap(df_cm, annot=True, cmap='YlGnBu', fmt='d').get_figure()
    fig_.axes[0].set(xlabel='predicted', ylabel='groundtruth')
    plt.tight_layout()
    plt.close(fig_)
    return fig_


def unpack_to_npy(subject, data_path=None, img_files=None):
    for img_file in img_files:
        basename = img_file.split('.')[0]
        if not os.path.exists(data_path / subject / f"{basename}.npy"):
            print(f"unpacking {subject}")
            data = nib.load(data_path / subject / f"{basename}.nii.gz").get_fdata()
            np.save(data_path / subject / f"{basename}.npy", data)


def get_mean_stats_over_img_files(dataset_stats, img_files):
    """
    Average the intensitiy statistics of each img_file (each channel / modality).
    """
    metrics = dataset_stats[img_files[0]].keys()
    metrics_mean = defaultdict(list)
    for img_file in img_files:
        for metric in metrics:
            metrics_mean[metric].append(dataset_stats[img_file][metric])
    metrics_mean = {k:np.array(v).mean() for k,v in metrics_mean.items()}
    return metrics_mean


if __name__ == "__main__":
    TEST_stratified_kfold_train_val_test()
