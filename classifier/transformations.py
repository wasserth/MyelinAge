import sys
from pathlib import Path
p_dir = str(Path(__file__).absolute().parents[1])
if p_dir not in sys.path: sys.path.insert(0, p_dir)

import random
import time 

import numpy as np
import nibabel as nib

from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, ToTensor
from monai.transforms import ResizeWithPadOrCrop, RandGaussianNoise, RandShiftIntensity, RandScaleIntensity
from monai.transforms import RandAdjustContrast, RandFlip, RandGaussianSmooth, RandZoom, ThresholdIntensity
from monai.transforms import NormalizeIntensity, RandSpatialCrop, CenterSpatialCrop
from monai.transforms import Rand3DElastic, Rand2DElastic
from monai.transforms import RandBiasField, RandGaussianSharpen, RandGibbsNoise, RandKSpaceSpikeNoise, RandHistogramShift
from monai.transforms import RandCoarseDropout, RandCoarseShuffle  # not yet in monai 0.6.0 (used for kaggle)

from libs.random_erasing import RandomErasing

"""
todo:
Try the following augmentations:
(https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/194145)

albumentations.RandomContrast(limit=0.2, p=1.0),
albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=1.0),
albumentations.Cutout(num_holes=2, max_h_size=int(0.4*image_size), max_w_size=int(0.4*image_size), fill_value=0, always_apply=True, p=1.0),
"""

class RandPosterize():
    def __init__(self, factor=[1.2, 1.8], prob=1.0):
        self.prob = prob
        self.factor = factor

    def __call__(self, data):
        if np.random.RandomState().rand() < self.prob:
            thr = data.mean() + data.std() * np.random.uniform(self.factor[0], self.factor[1])
            data[data>thr] = thr
        return data


def get_transforms(hparams):
    p = hparams.daug_prob
    # Monai uses np.random.RandomState() for randomness. Experimentally this resulted in different seed
    # for each subprocess (unlike np.random.randint which has the seed bug)

    # Augmentations in auto_augment: 
    # equalize, invert, contrast, brightness, sharpen, posterize, solarize, shear, translate, rotate
    #
    # Senseless for grayscale: 
    # ColorJitter
    #
    # Not so meaningful for grayscale:
    # solarize

    # Training transforms
    train_tfs = []
    # train_tfs += [ResizeWithPadOrCrop(hparams.crop_size)]

    # All of these transformations are the same for each channel
    if hparams.clip: 
        train_tfs += [ThresholdIntensity(hparams.clip_high, above=False, cval=hparams.clip_high)]
        train_tfs += [ThresholdIntensity(hparams.clip_low, above=True, cval=hparams.clip_low)]

    if hparams.normalize:
        # nnUNet normalization: 
        # CT: get global mean + std + percentile; then clip to 95% and -global_mean /global_std
        # MRI: -mean /std per image individually
        # RGB/XRAY/unknown: like MRI
        if hparams.norm_global:
            if hparams.global_mean == 0:
                raise ValueError("norm_global is set, but global_mean is not. (is 0 but should be list).")
            # global_mean and args.global_std are lists with one element for each modality
            train_tfs += [NormalizeIntensity(subtrahend=hparams.global_mean, divisor=hparams.global_std,
                                             channel_wise=hparams.norm_channel_wise)]  # -mean /std
        else:
            # Channelwise good for multi-modality, but bad for multi-slice
            train_tfs += [NormalizeIntensity(channel_wise=hparams.norm_channel_wise,
                                             nonzero=hparams.norm_ignore_zero)]  # -mean /std

    if hparams.daug_zoom: 
        # Zoom will keep the image size the same. Will add padding if making image smaller.
        # This is reason why black borders: Zoom in will cut image and later on ResizeWithPadOrCrop will add
        # padding (black borders) to make it fit to crop_size. RandSpatialCrop can make black border even larger.
        train_tfs += [RandZoom(min_zoom=0.7, max_zoom=1.2, padding_mode="constant", prob=p)]
    if hparams.daug_shift: 
        train_tfs += [RandShiftIntensity([-10, 10], prob=p)]    # default false
    if hparams.daug_scale: 
        train_tfs += [RandScaleIntensity([-0.3, 0.3], prob=p)] # 4: unrealistic; 1.5: strong
    if hparams.daug_contrast: 
        # train_tfs += [RandAdjustContrast(gamma=[0.5, 4.5], prob=p)]  # high intensity
        train_tfs += [RandAdjustContrast(gamma=[0.5, 1.2], prob=p)]  # low intensity
    if hparams.daug_noise: 
        # train_tfs += [RandGaussianNoise(mean=0.0, std=50, prob=p)]  # 150: strong noise; this is for pre-normalization
        train_tfs += [RandGaussianNoise(mean=0.0, std=0.1, prob=p)]  # this for after normalization; 0.1: medium; 0.2: strong
    if hparams.daug_smooth: 
        train_tfs += [RandGaussianSmooth(sigma_x=(.1, .8), sigma_y=(.1, .8), sigma_z=(.1, .8), prob=p)]  # used as default for brats_genom so far
        # train_tfs += [RandGaussianSmooth(sigma_x=(.1, .4), sigma_y=(.1, .4), sigma_z=(.1, .4), prob=p)]  # light blurring; [.1, .8]: medium bluring; [.5, 1.5]: strong bluring
    if hparams.daug_rotate: 
        train_tfs += [RandRotate90(prob=p)]
    if hparams.daug_flip: 
        train_tfs += [RandFlip(prob=p)]
    if hparams.daug_bias_field:
        # way too strong -> can not be reduced by params somehow
        train_tfs += [RandBiasField(degree=3, coeff_range=(0., .1), prob=p)]  # realistic range; [0, 0.2] would be intense
    if hparams.daug_sharpen:
        train_tfs += [RandGaussianSharpen(sigma1_x=(.1, .4), sigma1_y=(.1, .4), sigma1_z=(.1, .4), 
                                        sigma2_x=0.3, sigma2_y=0.3, sigma2_z=0.3, 
                                        alpha=(5., 10.), prob=p)]  # depending on random value minor to major changes; medium runtime
    if hparams.daug_histogram_shift:
        train_tfs += [RandHistogramShift(prob=p)]  # medium change
    if hparams.daug_gibbs_noise:
        train_tfs += [RandGibbsNoise(alpha=(.0, .6), as_tensor_output=False, prob=p)]  # 0.6: quite strong and blurry
    if hparams.daug_spike_noise:
        # only good in small range; might have to tune for each dataset??
        train_tfs += [RandKSpaceSpikeNoise(intensity_range=(10,12), as_tensor_output=False, prob=p)]  
    if hparams.daug_cutout:
        train_tfs += [RandCoarseDropout(10, 3, max_holes=20, max_spatial_size=(20, 30, 20), prob=p)]  # fast
        # train_tfs += [RandCoarseShuffle(10, 3, max_holes=20, max_spatial_size=(20, 30, 20), prob=p)]  # still too easy for network; can remember intensity statistics
    if hparams.daug_posterize:
        train_tfs += [RandPosterize([1.6, 2.0], prob=p)]  # higher values means less augmentation

    if hparams.daug_elastic:  # high CPU requirement
        # Higher sigma increases runtime a lot; higher magnitude only increases slightly.
        # For bigger deformations: lower sigma and higher magnitude.
        #
        # sigma=[5, 7], magnitude=[50, 70]: minor visible deformations
        # sigma=[1, 2], magnitude=[5, 10]: minor visible deformations, faster
        # sigma=[2, 5], magnitude=[70, 90]: major visible deformations
        # sigma=[2, 3], magnitude=[10, 20]: medium visible deformations, faster  (same speed as minor fast)
        # values <1.0 for sigma and magnitude will lead to strange artifacts
        #
        # rotate_range=[0.3, 0.3, 0.3]: roughly 20degree along each axis; good: clear rotation, but overally still same orientation
        # shear_range=[0.2, 0.2, 0.2]: obvious shearing, but overal orientation still the same
        # translate_range=[10, 10, 10]: simply add offset to image in pixel
        # scale_range=[0.2, 0.2, 0.2]: scaling by 20% -> good
        #
        # todo: make magnitude dependend on image size
        
        # low intensity
        if hparams.dim == 2:
            train_tfs += [Rand2DElastic([20, 20], [1, 2],
                                        prob=p, rotate_range=[0.1, 0.1, 0.1],
                                        shear_range=[0.1, 0.1, 0.1], translate_range=[5, 5, 5],
                                        scale_range=[0.1, 0.1, 0.1], padding_mode="zeros")]
        else:
            # train_tfs += [Rand3DElastic([2, 3], [5, 20],  # slightly higher deform
            train_tfs += [Rand3DElastic([1, 2], [5, 10],
                                        prob=p, rotate_range=[0.1, 0.1, 0.1],
                                        shear_range=[0.1, 0.1, 0.1], translate_range=[5, 5, 5],
                                        scale_range=[0.1, 0.1, 0.1], padding_mode="zeros")]
        # medium intensity
        # train_tfs += [Rand3DElastic([2, 3], [10, 20],
        #                            prob=p, rotate_range=[0.3, 0.3, 0.3],
        #                            shear_range=[0.2, 0.2, 0.2], translate_range=[10, 10, 10],
        #                            scale_range=[0.2, 0.2, 0.2], padding_mode="zeros")]  # can run on GPU

    if hparams.daug_elastic:
        train_tfs += [ResizeWithPadOrCrop(hparams.crop_size)]
    else:
        # zoom_for_rand_crop = 1.2  # leads to larger black borders
        zoom_for_rand_crop = 1.05
        train_tfs += [ResizeWithPadOrCrop((np.array(hparams.crop_size)*zoom_for_rand_crop).astype(int))]
        train_tfs += [RandSpatialCrop(hparams.crop_size, random_center=True, random_size=False)]

    # not needed if using -mean /std normalization
    if hparams.daug_scale_itensity:
        train_tfs += [ScaleIntensity(minv=-1, maxv=1)]
    
    train_tfs += [ToTensor()]

    if hparams.daug_random_erasing:
        train_tfs += [RandomErasing(max_area=hparams.daug_re_area,
                                    min_count=1, max_count=hparams.daug_re_count, 
                                    mode=hparams.daug_re_mode, probability=p)]  # working on tensors

    # Validation transforms
    val_tfs = []
    # val_tfs += [ResizeWithPadOrCrop(hparams.crop_size)]
    if hparams.clip: 
        val_tfs += [ThresholdIntensity(hparams.clip_high, above=False, cval=hparams.clip_high)]
        val_tfs += [ThresholdIntensity(hparams.clip_low, above=True, cval=hparams.clip_low)]
    
    val_tfs += [ResizeWithPadOrCrop(hparams.crop_size)]

    if hparams.daug_scale_itensity:
        val_tfs += [ScaleIntensity(minv=-1, maxv=1)]
    if hparams.normalize:
        if hparams.norm_global:
            val_tfs += [NormalizeIntensity(subtrahend=hparams.global_mean, divisor=hparams.global_std,
                                           channel_wise=hparams.norm_channel_wise)]
        else:
            val_tfs += [NormalizeIntensity(channel_wise=hparams.norm_channel_wise,
                                           nonzero=hparams.norm_ignore_zero)]
    val_tfs += [ToTensor()]

    # if hparams.dim == 3:
    #     train_tfs = [AddChannel()] + train_tfs
    #     val_tfs = [AddChannel()] + val_tfs

    return Compose(train_tfs), Compose(val_tfs)


if __name__ == "__main__":
    from argparse import Namespace

    file_in = sys.argv[1]
    file_out = sys.argv[2]

    img_in = nib.load(file_in)
    data = img_in.get_fdata()
    data = data[None, :, :, :]

    hparam_keys = ["daug_prob", "daug_zoom", "daug_elastic", "daug_shift", "daug_scale", "daug_contrast",
                   "daug_noise", "daug_smooth", "daug_rotate", "daug_flip", "daug_scale_itensity",
                   "daug_random_erasing", "daug_re_area", "daug_re_count", "daug_re_mode", "clip",
                   "clip_low", "clip_high", "normalize", "norm_channel_wise", "norm_global",
                   "norm_ignore_zero", "daug_bias_field", "daug_sharpen", "daug_histogram_shift",
                   "daug_gibbs_noise", "daug_spike_noise", "daug_cutout", "daug_posterize"]
    hparams = Namespace(**{k: 0 for k in hparam_keys})

    # hparams.daug_elastic = 1  # 1.0s

    # hparams.daug_zoom = 1
    # hparams.daug_bias_field = 1  # not good
    # hparams.daug_sharpen = 1
    # hparams.daug_histogram_shift = 1
    # hparams.daug_gibbs_noise = 1  # 0.7s  # hardy visible if used with noise and smooth -> not worth the long runtime
    # hparams.daug_spike_noise = 1  # not good
    # hparams.daug_posterize = 1

    hparams.daug_contrast = 1  # 0.12s
    
    # hparams.daug_noise = 1  # 0.18s
    # hparams.daug_smooth = 1  # 0.20s
    # hparams.daug_flip = 1 

    # hparams.daug_random_erasing = 1  # 0.10s
    # hparams.daug_re_area = 0.1
    # hparams.daug_re_count = 8
    # hparams.daug_re_mode = "const"

    # hparams.daug_flip = 1

    hparams.daug_prob = 1.0
    # hparams.daug_prob = 0.2

    # hparams.crop_size = [140,170,140]
    hparams.crop_size = [86,94,86]
    modality = "mri"
    
    if modality == "ct":
        print("Use CT normalization: -global_mean /global_std")
        hparams.normalize = 1
        hparams.norm_global = 1
        hparams.clip = 1
        hparams.clip_low = 200  # intensity_stats["perc02"]
        hparams.clip_high = 2000 # intensity_stats["perc98"]
        hparams.global_mean = 1000
        hparams.global_std = 300
    else:
        print("Use MRI/other normalization: -mean /std per image and channel (and no clipping)")
        hparams.normalize = 1
        hparams.norm_channel_wise = 1
        hparams.clip = 0

    # hparams.    
    train_tfs, val_tfs = get_transforms(hparams)

    # transform
    st = time.time()
    data = train_tfs(data)
    data = data.numpy()
    print(f"took: {time.time()-st:.2f}s")

    nib.save(nib.Nifti1Image(data[0,...], img_in.affine), file_out)
