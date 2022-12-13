# MyelinAge

Model to predict the myelin age in infant MRI images.

This is the code for the paper: TODO

The training data can be found here: https://zenodo.org/record/6556135


Install dependencies:
* Docker (needed for brain extraction with `freesurfer/synthstrip:latest`)
* pytorch

Install MyelinAge:
```
git clone https://github.com/wasserth/MyelinAge.git
cd MyelinAge
pip install -r requirements.txt
```

Run:
```
python brainage/report/generate_report.py -t1 t1.nii.gz -t2 t2.nii.gz -o myelinage_report.png
```
