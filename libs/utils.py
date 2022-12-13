import os
from pathlib import Path
import pickle

import numpy as np
import nibabel as nib
from scipy import ndimage
import psutil
from joblib import Parallel, delayed
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_closing
from sklearn.metrics import confusion_matrix



def keep_largest_blob(data, debug=False):
    blob_map, _ = ndimage.label(data)
    counts = list(np.bincount(blob_map.flatten()))  # number of pixels in each blob
    if len(counts) <= 1: return data  # no foreground
    if debug: print(f"size of second largest blob: {sorted(counts)[-2]}")
    key_second = counts.index(sorted(counts)[-2])
    return (blob_map == key_second).astype(np.uint8)


def remove_small_blobs(img: np.ndarray, interval=[10, 30], debug=False) -> np.ndarray:
    """
    Find blobs/clusters of same label. Remove all blobs which have a size which is outside of the interval.

    Args:
        img: Binary image.
        interval: Boundaries of the sizes to remove.
        debug: Show debug information.
    Returns:
        Detected blobs.
    """
    mask, number_of_blobs = ndimage.label(img)
    if debug: print('Number of blobs before: ' + str(number_of_blobs))
    counts = np.bincount(mask.flatten())  # number of pixels in each blob

    # If only one blob (only background) abort because nothing to remove
    if len(counts) <= 1: return img

    remove = np.where((counts <= interval[0]) | (counts > interval[1]), True, False)
    remove_idx = np.nonzero(remove)[0]
    mask[np.isin(mask, remove_idx)] = 0
    mask[mask > 0] = 1  # set everything else to 1

    if debug:
        print(f"counts: {sorted(counts)[::-1]}")
        _, number_of_blobs_after = ndimage.label(mask)
        print('Number of blobs after: ' + str(number_of_blobs_after))

    return mask


def get_bbox_from_mask(mask, outside_value=-900, addon=0):
    if type(addon) is int:
        addon = [addon] * 3
    if (mask > outside_value).sum() == 0: 
        print("WARNING: Could not crop because no foreground detected")
        minzidx, maxzidx = 0, mask.shape[0]
        minxidx, maxxidx = 0, mask.shape[1]
        minyidx, maxyidx = 0, mask.shape[2]
    else:
        mask_voxel_coords = np.where(mask > outside_value)
        minzidx = int(np.min(mask_voxel_coords[0])) - addon[0]
        maxzidx = int(np.max(mask_voxel_coords[0])) + 1 + addon[0]
        minxidx = int(np.min(mask_voxel_coords[1])) - addon[1]
        maxxidx = int(np.max(mask_voxel_coords[1])) + 1 + addon[1]
        minyidx = int(np.min(mask_voxel_coords[2])) - addon[2]
        maxyidx = int(np.max(mask_voxel_coords[2])) + 1 + addon[2]

    # Avoid bbox to get out of image size
    s = mask.shape
    minzidx = max(0, minzidx)
    maxzidx = min(s[0], maxzidx)
    minxidx = max(0, minxidx)
    maxxidx = min(s[1], maxxidx)
    minyidx = max(0, minyidx)
    maxyidx = min(s[2], maxyidx)

    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    return image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]


def crop_to_bbox_nifti(image: nib.Nifti1Image, bbox) -> nib.Nifti1Image:
    """
    Crop nifti image to bounding box and adapt affine accordingly

    image: nib.Nifti1Image
    bbox: list of lists [[minx_idx, maxx_idx], [miny_idx, maxy_idx], [minz_idx, maxz_idx]]
          Indices of bbox must be in voxel coordinates  (not in world space)

    returns: nib.Nifti1Image
    """
    assert len(image.shape) == 3, "only supports 3d images"
    data = image.get_fdata()

    # Crop the image
    data_cropped = data[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]

    # Update the affine matrix
    affine = np.copy(image.affine)
    affine[:3, 3] = np.dot(affine, np.array([bbox[0][0], bbox[1][0], bbox[2][0], 1]))[:3]

    return nib.Nifti1Image(data_cropped.astype(image.dataobj.dtype), affine)


def crop_to_foreground(data, seg=None, bbox=None, addon=0, bg_val=-900):
    """
    data: [x,y,z]
    seg: will get cropped by same bbox if provided
    bbox: if not provided will be generated around all foreground (everything >bg_val )
    addon: int or triple; enlarge bbox by that amount (only works if bbox=None)
    """
    original_shape = data.shape
    if bbox is None:
        bbox = get_bbox_from_mask(data, bg_val, addon=addon)

    cropped_data = []
    if len(data.shape) == 3: 
        data = data[..., None]  # make 4d
    for c in range(data.shape[3]):
        cropped = crop_to_bbox(data[:,:,:,c], bbox)
        cropped_data.append(cropped)
    data = np.array(cropped_data).transpose(1,2,3,0)
    data = data[:,:,:,0] if data.shape[3] == 1 else data  # remove channel dim if only one channel

    if seg is not None:
        cropped_seg = []
        if len(seg.shape) == 3: 
            seg = seg[..., None]  # make 4d
        for c in range(seg.shape[3]):
            cropped = crop_to_bbox(seg[:,:,:,c], bbox)
            cropped_seg.append(cropped)
        seg = np.array(cropped_seg).transpose(1, 2, 3, 0)
        seg = seg[:,:,:,0] if seg.shape[3] == 1 else seg  # remove channel dim if only one channel

    return data, seg, bbox, original_shape


def crop_to_foreground_nifti(img_in, seg=None, bbox=None, addon=0, bg_val=-900, dtype=None):
    """
    Just as crop_to_foreground, but takes as input a NiftiImage and correctly changes the affine.
    Also returns a nifti image.
    """
    data = img_in.get_fdata()
    data_out, _, bbox, _ = crop_to_foreground(data, seg, bbox, addon, bg_val)

    # Update the affine matrix
    affine = np.copy(img_in.affine)
    affine[:3, 3] = np.dot(affine, np.array([bbox[0][0], bbox[1][0], bbox[2][0], 1]))[:3]

    data_type = img_in.dataobj.dtype if dtype is None else dtype
    return nib.Nifti1Image(data_out.astype(data_type), affine)


def remove_negative_values(img_in, img_out, dtype=np.float32):
    img_in = nib.load(img_in)
    data = img_in.get_fdata()
    data[data < 0] = 0
    nib.save(nib.Nifti1Image(data.astype(dtype), img_in.affine), img_out)


def get_full_task_name(task_id: int, src: str="raw", dim: str="2d"):
    if src == "raw":
        base = Path(os.environ['nnUNet_raw_data_base']) / "nnUNet_raw_data"
    elif src == "preprocessed":
        base = Path(os.environ['nnUNet_preprocessed'])
    elif src == "results":
        base = Path(os.environ['RESULTS_FOLDER']) / "nnUNet" / dim
    dirs = [str(dir).split("/")[-1] for dir in base.glob("*")]
    for dir in dirs:
        if f"Task{task_id:03d}" in dir:
            return dir
    raise ValueError(f"task_id {task_id} not found")


def get_split_of_subject(task_id: int, study_id: str):

    task_name = get_full_task_name(task_id, src="preprocessed")
    splits = pickle.load(open(Path(os.getenv("nnUNet_preprocessed")) / task_name / "splits_final.pkl", "rb"))

    fold_found = 0  # use 0 as default fold for subjects which were not part of CV set
    for fold in range(len(splits)):
        for subject_id in splits[fold]["val"]:
            if subject_id == study_id:
                fold_found = fold

    return fold_found


def my_f1_score(y_true, y_pred):
    """
    Binary f1. Same results as sklearn f1 binary.
    """
    intersect = np.sum(y_true * y_pred)  # works because all multiplied by 0 gets 0
    denominator = np.sum(y_true) + np.sum(y_pred)  # works because all multiplied by 0 gets 0
    f1 = (2 * intersect) / (denominator + 1e-6)
    return f1


def get_all_metrics(y_true, y_pred, average="micro"):
    """
    y_true: 1d array    
    y_pred: 1d array
    average: None | "micro" | "macro"
        Micro is less affected by class imbalances than macro. For macro one small class with a bad score
        will make entire result bad, even if only a few samples.
        Macro is more interpretable in case of multiclass: Calc score for each class, then average
        None will return one score for each class (like sklearn average=None)

    Returns dict with metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cnf_matrix = confusion_matrix(y_true, y_pred)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    if average == "micro":
        FP = FP.sum()
        FN = FN.sum()
        TP = TP.sum()
        TN = TN.sum()

    # Sensitivity, hit rate, recall, or true positive rate
    sensitivity = TP/(TP+FN+1e-10)
    # Specificity or true negative rate
    specificity = TN/(TN+FP+1e-10) 
    # Precision or positive predictive value
    precision = TP/(TP+FP+1e-10)
    # Negative predictive value
    npv = TN/(TN+FN)
    # Fall out or false positive rate
    fpr = FP/(FP+TN)
    # False negative rate
    fnr = FN/(TP+FN)
    # False discovery rate
    fdr = FP/(TP+FP)
    # Overall accuracy for each class
    # acc = (TP+TN)/(TP+FP+FN+TN)  # deals with each class independently (like f1)
    acc = (y_true == y_pred).sum() / len(y_true)  # this is the way sklearn calculates it for multiclass
    # F1 score
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    # MCC (Matthews correlation coefficient)   (better estimation if strong class imbalance)
    # micro average leads to different results than sklearn mcc. sklearn same as macro or None here.
    nr_classes = len(TP) if type(TP) is np.ndarray else 1
    mcc = (TP*TN-FP*FN) / np.max([np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)), 
                                  np.ones(nr_classes)], axis=0)  # If denominator is 0, it can be set to 1

    r = {
        "fp": FP,
        "fn": FN,
        "TP": TP,
        "TN": TN,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "fpr": fpr,
        "fnr": fnr,
        "fdr": fdr,
        "acc": acc,
        "f1": f1,
        "mcc": mcc
    }

    if average == "macro":
        r = {k: v.mean() for k, v in r.items()}

    # convert type from numpy to native python to make it json serializable
    r = {k: v.tolist() for k, v in r.items()}

    return r


def flatten(l):
    return [item for sublist in l for item in sublist]


def maybe_mkdir_p(directory):
    directory = os.path.abspath(directory)
    splits = directory.split("/")[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[:i+1])):
            try:
                os.mkdir(os.path.join("/", *splits[:i+1]))
            except FileExistsError:
                # this can sometimes happen when two jobs try to create the same directory at the same time,
                # especially on network drives.
                print("WARNING: Folder %s already existed and does not need to be created" % directory)


def scale_to_range(data, range=(0, 255), integer=True):
    """
    Works for pytorch and numpy
    """
    if integer:
        return (range[1] - range[0]) * (data - data.min()) // (data.max() - data.min()) + range[0]
    else:
        return (range[1] - range[0]) * (data - data.min()) / (data.max() - data.min()) + range[0]


def scale_to_range_global(data, range=(0, 255), global_min_max=(0, 0.03)):
    """
    Rescale to range, but take min_max of entire dataset instead of only this sample.
    """
    return (range[1] - range[0]) * (data - global_min_max[0]) / (global_min_max[1] - global_min_max[0]) + range[0]


def segment_lungs_by_threshold(data, debug=False):
    data_thr = (data > -900) & (data < -130)  # segment lung (and minor other things)

    # border = 20
    # data_thr[:border, :border, :border] = 0
    # data_thr[-border:, -border:, -border:] = 0

    data_thr = binary_erosion(data_thr, iterations=3)  # remove contour lines 
    data_thr = binary_closing(data_thr, structure=np.ones([3]*3))
    data_thr = binary_dilation(data_thr, iterations=2)

    mask, _ = ndimage.label(data_thr)
    counts = list(np.bincount(mask.flatten()))  # number of pixels in each blob
    if debug: print(f"size of second largest blob: {sorted(counts)[-2]}")
    key_second = counts.index(sorted(counts)[-2])
    key_third = counts.index(sorted(counts)[-3])
    lung_mask = (mask == key_second) | (mask == key_third)

    lung_mask = binary_closing(lung_mask, structure=np.ones([10]*3))

    return lung_mask


def rm_substr_from_list(l, substrs):
    """
    Will check each elem of l if it contains as substring any element from substrs. If true then deletes
    the elem from l.

    l: list of strings
    substrs: list of strings

    returns: subset of l
    """
    r = []
    for e in l:
        delete = False
        for substr in substrs:
            if substr in e:
                delete = True
        if not delete:
            r.append(e)
    return r


def crop_multiple_to_foreground(images, mask, out_dir=None):
    """
    Crop all images to foreground using the bounding box from the first image in the list

    images: list of image pathes
    mask: path to brain mask
    """
    mask_img = nib.load(mask)
    mask = mask_img.get_fdata() > 0

    img = nib.load(str(images[0])).get_fdata()
    img[~mask] = 0
    _, _, bbox, _ = crop_to_foreground(img, None, None, 0, 0)

    # Update the affine matrix
    affine = np.copy(mask_img.affine)
    affine[:3, 3] = np.dot(affine, np.array([bbox[0][0], bbox[1][0], bbox[2][0], 1]))[:3]

    for img_path in images:
        img = nib.load(img_path).get_fdata()
        img[~mask] = 0
        img, _, _, _ = crop_to_foreground(img, None, bbox, 0, 0)
        if out_dir is None:
            nib.save(nib.Nifti1Image(img, affine), str(img_path)[:-7] + "_crop.nii.gz")
        else:
            nib.save(nib.Nifti1Image(img, affine), out_dir / (str(img_path.name)[:-7] + "_crop.nii.gz"))
            