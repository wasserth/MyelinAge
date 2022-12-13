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
from libs.alignment import transform_image, rigid_registration_dipy
from libs.resampling import resample_img, change_spacing_nifti
from classifier.run_inference import run_inference_single_file

"""
This requires flirt.
"""

def brain_mask_nnUNet(img_path, out_path):

    # Using nnunet
    # maybe needed: export RESULTS_FOLDER="/mnt/nvme/nnunet/results"
    # subprocess.call(f"python ~/dev/nnunet_docker/nnUNet_predict_image.py " +
    #                 f"-i {img_path} " +
    #                 f"-o {out_path} " +
    #                 f"-t 160 -m 2d -f 0 -q -tta", shell=True)
    # img = nib.load(out_path)
    # data_cl = binary_closing(img.get_fdata(), structure=np.ones([3]*3))    
    # nib.save(nib.Nifti1Image(data_cl.astype(np.uint8), img.affine), out_path)

    # Using synthstrip (faster and no GPU required and more precise)
    subprocess.call(f"docker run -v {img_path.parent}:/workspace " +
                    f"freesurfer/synthstrip:latest " +
                    f"-i /workspace/{img_path.name} " +
                    f"-o /workspace/skull_removed_tmp.nii.gz " +
                    f"-m /workspace/nodif_brain_mask_synthstrip.nii.gz > /dev/null", shell=True)
    os.remove(img_path.parent / "skull_removed_tmp.nii.gz")
    shutil.move(img_path.parent / "nodif_brain_mask_synthstrip.nii.gz", out_path)


def register_to_atlas(dir, atlas_res="06mm"):
    atlas_dir = str(Path(__file__).absolute().parent / "atlas")

    subprocess.call(f"flirt -ref {atlas_dir}/t1_reference_subject_{atlas_res}.nii.gz " + 
                    f"-in {dir}/t1_reg_crop.nii.gz -out {dir}/t1_atlas_{atlas_res}.nii.gz " +
                    f"-omat {dir}/t1_2_atlas_{atlas_res}.mat -dof 12 " +
                    f"-interp spline", shell=True)  # dof6: 10s, dof12: 25s
    remove_negative_values(dir / f"t1_atlas_{atlas_res}.nii.gz", dir / f"t1_atlas_{atlas_res}.nii.gz", dtype=np.uint16)

    subprocess.call(f"flirt -ref {atlas_dir}/t1_reference_subject_{atlas_res}.nii.gz " + 
                    f"-in {dir}/t2_crop.nii.gz -out {dir}/t2_atlas_{atlas_res}.nii.gz " +
                    f"-applyxfm -init {dir}/t1_2_atlas_{atlas_res}.mat -dof 12 " +
                    f"-interp spline", shell=True)
    remove_negative_values(dir / f"t2_atlas_{atlas_res}.nii.gz", dir / f"t2_atlas_{atlas_res}.nii.gz", dtype=np.uint16)


def preprocess_image(t1_path, t2_path, tmp_dir, rm_intermediate_files=False, resample=[0.6, 0.6, 3.0], copy_t2=False):
    if copy_t2:
        print(f"Rename T2...")
        shutil.copy(t2_path, tmp_dir / "t2.nii.gz")  # standardise naming because in crop_multiple_to_foreground will take this name
        t2_path = tmp_dir / "t2.nii.gz"
    print(f"Registering images...")
    transform_image(t2_path, t1_path, f"{tmp_dir}/t1_reg.nii.gz", dtype=np.float32)  # t1 to t2
    print(f"Generating brainmask...")
    brain_mask_nnUNet(t2_path, tmp_dir / "nodif_brain_mask_nnunet.nii.gz")
    print(f"Cropping images...")
    crop_multiple_to_foreground([t2_path, tmp_dir/"t1_reg.nii.gz"], tmp_dir/"nodif_brain_mask_nnunet.nii.gz", tmp_dir)

    # Not needed because automatically done by atlas
    # print(f"Resampling images...")
    # change_spacing_nifti(tmp_dir / "t1_reg_crop.nii.gz", tmp_dir / "t1_reg_crop_rsp.nii.gz",
    #                      resample, order=3, remove_negative=True, dtype=np.float32)
    # change_spacing_nifti(tmp_dir / "t2_crop.nii.gz", tmp_dir / "t2_crop_rsp.nii.gz",
    #                      resample, order=3, remove_negative=True, dtype=np.float32)

    # This did not help for the 3 images which were not properly registered
    # # t1
    # subprocess.call(f"fslorient -copysform2qform {tmp_dir}/t1_reg_crop.nii.gz", shell=True)
    # subprocess.call(f"fslreorient2std {tmp_dir}/t1_reg_crop.nii.gz  {tmp_dir}/t1_reg_crop.nii.gz", shell=True)
    # # t2
    # subprocess.call(f"fslorient -copysform2qform {tmp_dir}/t2_crop.nii.gz", shell=True)
    # subprocess.call(f"fslreorient2std {tmp_dir}/t2_crop.nii.gz  {tmp_dir}/t2_crop.nii.gz", shell=True)

    print(f"Registering to atlas...")
    atlas_res = "15mm" if resample[0] == 1.5 else "06mm"
    register_to_atlas(tmp_dir, atlas_res=atlas_res)

    if rm_intermediate_files:
        print(f"Removing intermediate files...")
        # os.remove(tmp_dir / "t1.nii.gz")
        os.remove(tmp_dir / "t2.nii.gz")
        os.remove(tmp_dir / "t2_crop.nii.gz")
        # os.remove(tmp_dir / "t2_crop_rsp.nii.gz")
        os.remove(tmp_dir / "t1_reg.nii.gz")
        os.remove(tmp_dir / "t1_reg_crop.nii.gz")
        # os.remove(tmp_dir / "t1_reg_crop_rsp.nii.gz")
        os.remove(tmp_dir / f"t1_2_atlas_{atlas_res}.mat")
        os.remove(tmp_dir / "nodif_brain_mask_nnunet.nii.gz")


def run_inference_brainage(img_file_pathes):
    # env = Env()
    # # starts searching for .env in the directory this python file is in
    # env.read_env()  # if .env file found the content of the file will be written into os.environ; only if env var does not exist yet
    
    exps = ["d2_2dTi_final", "d2_2dTi_final_run2", "d2_2dTi_final_run3",
            "d2_3d_chen_ep120", "d2_3d_chen_ep120_run2", "d2_3d_chen_ep120_run3"]
    exps = [f"{exp}_f{f}" for f in range(5) for exp in exps]
    project = "brainage/classify_5_groups"
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
