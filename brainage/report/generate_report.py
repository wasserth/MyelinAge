import sys
from pathlib import Path
p_dir = str(Path(__file__).absolute().parents[2])
if p_dir not in sys.path: sys.path.insert(0, p_dir)

import argparse
import json
from datetime import datetime
import tempfile
import shutil

from environs import Env
from jinja2 import Environment, FileSystemLoader, select_autoescape
import imgkit
import numpy as np
import nibabel as nib
from skimage.io import imsave

from brainage.predict_subject import preprocess_image, run_inference_brainage
from libs.utils import scale_to_range_global


def to_date(date_as_int):
    if date_as_int:
        return datetime.strptime(str(date_as_int), "%Y%m%d").strftime("%d.%m.%Y")
    else:
        return ""


def now():
    return datetime.now().strftime("%d.%m.%Y %H:%M:%S")


def generate_report(env, t1_slices, t2_slices, metadata, output_file):
    t = env.get_template("template.html")
    output_html = t.render(
        t1_slices = t1_slices,
        t2_slices = t2_slices,
        metadata=metadata,
    )

    print("Converting to PNG...")
    imgkit.from_string(output_html, str(output_file), 
                       options={"xvfb": "", "format": "png", "width": "1200", "height": "1200"})
    print(f"PNG saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t1", dest="t1_path", type=lambda p: Path(p).absolute(), required=True)
    parser.add_argument("-t2", dest="t2_path", type=lambda p: Path(p).absolute(), required=True)
    parser.add_argument("-o", dest="output_file", type=lambda p: Path(p).absolute(), required=True)
    args = parser.parse_args()
  
    templates_path = str(Path(__file__).absolute().parent)
    env = Environment(
        loader=FileSystemLoader(templates_path),
        autoescape=select_autoescape(["html", "xml"]),
    )
    env.filters["to_date"] = to_date
    env.globals["now"] = now
    version = "1.0"

    # todo
    # tmp_dir = Path(tempfile.mkdtemp())
    tmp_dir = Path("tmp_dir").absolute()
    tmp_dir.mkdir(exist_ok=True)
    print(f"tmp_dir: {tmp_dir}")

    print("Preprocessing....")
    preprocess_image(args.t1_path, args.t2_path, tmp_dir)

    print("Running inference....")
    predicted_age = run_inference_brainage([tmp_dir / "t2_atlas.nii.gz", tmp_dir / "t1_atlas.nii.gz"])
    predicted_age = round(predicted_age, 1)
    print(f"predicted age: {predicted_age}")

    biological_age = 31.0  # todo

    untypical = False if abs(biological_age - predicted_age) < 2 else True

    print("Getting metadata...")
    if (Path(args.t1_path).parent.parent / "nodeinfo.json").exists():
        with open(Path(args.t1_path).parent.parent / "nodeinfo.json", "r") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "PatientName": "Unknown",
            "PatientSex": "Unknown",
            "PatientBirthDate": "19000101",
            "PatientID": "Unknown",
            "StudyID": "Unknown",
            "StudyDescription": "Unknown",
            "StudyDate": "19000101",
        }

    metadata["biological_age"] = biological_age
    metadata["predicted_age"] = predicted_age
    metadata["untypical"] = untypical
    metadata["version"] = version

    print("Slicing images...")
    t1 = nib.load(tmp_dir / "t1_atlas.nii.gz").get_fdata()
    t2 = nib.load(tmp_dir / "t2_atlas.nii.gz").get_fdata()

    t1_min, t1_max = t1.min(), t1.max()
    t2_min, t2_max = t2.min(), t2.max()

    ss = t1.shape
    nr_slices = 4
    slice_distance = 6   # higher means lower distance
    t1_slices = []
    t2_slices = []
    for idx in range(nr_slices):
        slice_idx = 5 + (ss[2]//slice_distance) * idx
        t1_slice = t1[:,:,slice_idx].transpose(1,0)[::-1,:]
        t2_slice = t2[:,:,slice_idx].transpose(1,0)[::-1,:]
        
        # Normalize all slices to same intensity
        t1_slice = scale_to_range_global(t1_slice, (0, 255), (t1_min, t1_max))
        t2_slice = scale_to_range_global(t2_slice, (0, 255), (t2_min, t2_max))

        #todo: make sure left/right is correct
        imsave(tmp_dir/f"t1_slice_{idx}.png", t1_slice.astype(np.uint8))
        imsave(tmp_dir/f"t2_slice_{idx}.png", t2_slice.astype(np.uint8))
        t1_slices.append(tmp_dir/f"t1_slice_{idx}.png")
        t2_slices.append(tmp_dir/f"t2_slice_{idx}.png")
    
    print("Generating report...")
    generate_report(env, t1_slices, t2_slices,
                    metadata, args.output_file)

    # shutil.rmtree(tmp_dir)  # todo
