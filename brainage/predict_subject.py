import sys
from pathlib import Path
p_dir = str(Path(__file__).absolute().parents[1])
if p_dir not in sys.path: sys.path.insert(0, p_dir)

import os
import tempfile
import shutil
import subprocess 

import numpy as np
import nibabel as nib
from scipy.ndimage.morphology import binary_dilation, binary_closing
from environs import Env

from libs.utils import remove_negative_values
from libs.utils import crop_multiple_to_foreground
from libs.alignment import transform_image
from libs.resampling import resample_img, change_spacing_nifti
from classifier.run_inference import run_inference_single_file

"""
This requires flirt.

When deploying: adapt brainage/.env 
"""


# todo: set correct path for nnunet weights and predict image script (Task160)
#       (create env file for this)


def brain_mask_nnUNet(img_path, out_path):
    subprocess.call(f"python ~/dev/nnunet_docker/nnUNet_predict_image.py " +
                    f"-i {img_path} " +
                    f"-o {out_path} " +
                    f"-w /mnt/nvme/nnunet/results/nnUNet " +
                    f"-t 160 -m 2d -f 0 -ori -1 1 1 -q -nd -tta", shell=True)
    img = nib.load(out_path)
    data_cl = binary_closing(img.get_fdata(), structure=np.ones([3]*3))    
    nib.save(nib.Nifti1Image(data_cl.astype(np.uint8), img.affine), out_path)


def register_to_atlas(dir):
    atlas_dir = Path("/mnt/nvme/data/brainage/atlas")

    subprocess.call(f"flirt -ref {atlas_dir}/t1_reg_crop_rsp_24272080.nii.gz " + 
                    f"-in {dir}/t1_reg_crop_rsp.nii.gz -out {dir}/t1_atlas.nii.gz " +
                    f"-omat {dir}/t1_2_atlas.mat -dof 12 " +
                    f"-interp spline", shell=True)  # dof6: 10s, dof12: 25s
    remove_negative_values(dir / f"t1_atlas.nii.gz", dir / f"t1_atlas.nii.gz", dtype=np.uint16)

    subprocess.call(f"flirt -ref {atlas_dir}/t1_reg_crop_rsp_24272080.nii.gz " + 
                    f"-in {dir}/t2_crop_rsp.nii.gz -out {dir}/t2_atlas.nii.gz " +
                    f"-applyxfm -init {dir}/t1_2_atlas.mat -dof 12 " +
                    f"-interp spline", shell=True)
    remove_negative_values(dir / f"t2_atlas.nii.gz", dir / f"t2_atlas.nii.gz", dtype=np.uint16)


def preprocess_image(t1_path, t2_path, tmp_dir):
    print(f"Registering images...")
    # transform_image(t2_path, t1_path, f"{tmp_dir}/t1_reg.nii.gz", dtype=np.float32)
    print(f"Generating brainmask...")
    # brain_mask_nnUNet(t2_path, tmp_dir / "nodif_brain_mask_nnunet.nii.gz")
    print(f"Cropping images...")
    # crop_multiple_to_foreground([t2_path, tmp_dir/"t1_reg.nii.gz"], tmp_dir/"nodif_brain_mask_nnunet.nii.gz", tmp_dir)
    print(f"Resampling images...")
    # change_spacing_nifti(tmp_dir / "t1_reg_crop.nii.gz", tmp_dir / "t1_reg_crop_rsp.nii.gz",
    #                      [0.6, 0.6, 3.0], order=3, remove_negative=True, dtype=np.float32)
    # change_spacing_nifti(tmp_dir / "t2_crop.nii.gz", tmp_dir / "t2_crop_rsp.nii.gz",
    #                      [0.6, 0.6, 3.0], order=3, remove_negative=True, dtype=np.float32)
    print(f"Registering to atlas...")
    # register_to_atlas(tmp_dir)


def run_inference_brainage(img_file_pathes):
    env = Env()
    # starts searching for .env in the directory this python file is in
    env.read_env()  # if .env file found the content of the file will be written into os.environ

    exp_name = "d2_2dTi_atlas_hydra_el1_tiss1_tist8"
    project = "brainage/classify_5_groups"
    exps = [f"{exp_name}_f{f}" for f in range(5)]
    _, pred = run_inference_single_file(project, exps, img_file_pathes)
    return pred


if __name__ == "__main__":
    
    # tmp_dir = Path(tempfile.mkdtemp())
    tmp_dir = Path("tmp_dir")
    tmp_dir.mkdir(exist_ok=True)

    t1_path = Path(sys.argv[1])
    t2_path = Path(sys.argv[2])

    preprocess_image(t1_path, t2_path, tmp_dir)
    pred = run_inference_brainage([tmp_dir / "t2_atlas.nii.gz", tmp_dir / "t1_atlas.nii.gz"])

    # shutil.rmtree(tmp_dir)
