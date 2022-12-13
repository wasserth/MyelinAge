import sys
from pathlib import Path
p_dir = str(Path(__file__).absolute().parents[1])
if p_dir not in sys.path: sys.path.insert(0, p_dir)

import os
import sys
from pathlib import Path
import json
import warnings
import shutil
import random
import string

import torch
from torch.nn import functional as F
import nibabel as nib
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, r2_score
from scipy.stats import pearsonr, spearmanr

from classifier.lit_classifier import LitClassifier
from classifier.datasets import NiftiDataset, nifti_loader
from classifier.utils import confusion_matrix_to_figure
from classifier.utils_loading import load_lightning_model
from classifier.transformations import get_transforms
from libs.utils import get_all_metrics

warnings.filterwarnings("ignore")
pl.seed_everything(1234)

"""
Standalone prediction of a sample with a trained lightning model.

Does not work properly for mse loss. Use mse_cat los.
(it expects the _decode_y functions to convert predictions to classes, which is not the case for
mse loss)
"""

def run_inference_single_file(project, exps, img_file_pathes):

    project_name, task_name = project.split("/")
    data_path = Path(os.environ['lightning_data_jakob']) / project_name

    # Creating tmp data directory
    tmp_dir = ("classifier_tmp_" + ''.join(random.Random().choices(string.ascii_uppercase + string.digits, k=8)))
    (data_path / tmp_dir / "s01").mkdir(exist_ok=True, parents=True)
    data_folder = tmp_dir

    # Copy files to tmp directory
    subjects_val = ["s01"]
    for img_file, file_name in img_file_pathes:
        shutil.copy(img_file, data_path / tmp_dir / "s01" / file_name)
    
    # Create fake label
    labels_val = np.array([0], dtype=np.float32)

    # working with multiple dataloaders
    preds, preds_class = [], []
    for exp in exps:
        exp = exp if type(exp) is list else [exp]
        # img_files need to have same naming as in training data
        subject, gt_class, pred_class, gt, pred = run_inference(project, exp, subjects_val, labels_val,
                                                                data_folder=data_folder, tta_nr=1)
        preds.append(pred)
        preds_class.append(pred_class)
    preds = np.array(preds).mean(axis=0)
    preds_class = np.array(preds_class).mean(axis=0).round()

    shutil.rmtree(data_path / tmp_dir)

    return preds_class[0], preds[0]


def run_inference(project, exps, subjects_val=None, labels_val=None, img_files=None,
                  data_folder="data", file_loader=nifti_loader, data_path=None,
                  probs=False, tta_nr=1):
    """
    Returns 
        subjects_val: subject ids 
        gts_class: ground truth classes
        preds_class: predicted classes
        gts: raw ground truth (relevant for mse: the raw float values before mapping them to classes)
                (for ce: same as gts_class)
        preds: raw predictions (relevant for mse) 
                (for ce: filled with 0s)
    """
    batch_size = 16

    # Setup
    device = torch.device('cuda')

    project_name, task_name = project.split("/")
    if data_path is None:
        data_path_raw = Path(os.environ['lightning_data_jakob'])
        data_path = data_path_raw / project_name

    # Load model from checkpoint
    log_path = Path(os.environ['lightning_results_jakob'])
    models = []
    for exp in exps:    
        model, _ = load_lightning_model(log_path / project / exp)
        model.eval().to(device)
        if "lightning_quiet" not in os.environ:
            print(f"Exp name: {model.hparams.exp_name}")
        models.append(model)

    model = models[0]  # model to get hparams for preprocessing from

    # Load subjects from hparams
    if subjects_val is None or labels_val is None:
        subjects_val = model.hparams.subjects_val
        labels_val = model.hparams.labels_val

    if img_files is None:
        if hasattr(model.hparams, "img_files"):
            img_files = model.hparams.img_files  
        else:
            config = json.load(open(data_path / task_name / "dataset.json"))
            config["img_files"]
        img_files = [f.replace(".npy", ".nii.gz") for f in img_files]

    # Load subjects from files
    # data_path = data_path_raw / "siim"
    # subjects_val = [s.name for s in (data_path / "data").glob("*")]#[:1000]
    # labels_val = [0 if os.path.exists(data_path / "data" / s / "empty_mask.txt") else 1 for s in subjects_val]#[:1000]
    # img_files = "ct_025.nii.gz"  # ct_025.nii.gz | ct.nii.gz

    if "lightning_quiet" not in os.environ:
        print(f"Using img_files: {img_files}")

    tfs_train, tfs_val = get_transforms(model.hparams)

    tfs_used = tfs_val
    # tfs_used = tfs_train

    dataset_val = NiftiDataset(data_path / data_folder, subjects_val, labels_val,
                               transform=tfs_used,
                               dim=model.hparams.dim,
                               nr_slices=model.hparams.nr_slices,
                               multi_orientation=model.hparams.multi_orientation,
                               crop_size=model.hparams.crop_size,
                               slice_subset=model.hparams.slice_subset,
                               img_files=img_files,
                               loss=model.hparams.loss,
                               nr_classes=model.hparams.nr_classes,
                               deterministic=True,
                               tiles=model.hparams.tiles,
                               file_loader=file_loader,
                               tta_nr=tta_nr,
                               tiles_subsample=model.hparams.tiles_subsample,
                               tiles_start=model.hparams.tiles_start,
                               zoom=model.hparams.zoom,
                               slice_orientation=model.hparams.slice_orientation)

    # Per default DataLoader not doing any shuffling even with num_workers > 1
    loader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=10, pin_memory=True)


    gts = []
    preds = []
    gts_class = []
    preds_class = []
    with torch.no_grad():
        for idx, data in enumerate(tqdm(loader_val)):
            x, y, = data
            gts += list(y)
            gts_class += list(model._decode_y(y).numpy())
            # Ensemble different models
            preds_ens = []
            preds_class_ens = []
            for m in models:
                for tta_idx in range(tta_nr):
                    x_tta = x[tta_idx].to(device) if tta_nr > 1 else x.to(device)
                    y_hat = m(x_tta)
                    y_hat = y_hat[:,0] if m.hparams.loss.startswith("mse") else y_hat
                    y_hat = y_hat.detach().cpu()
                    y_hat = m._y_hat_to_prob(y_hat)
                    preds_ens.append(y_hat)
                    preds_class_ens.append(m._decode_y_hat(y_hat).float())
            # mean over models and tta (=ensemble)
            preds_ens = torch.stack(preds_ens).mean(axis=0)  # keep as torch tensor for _decode_y_hat
            preds_class += list(m._decode_y_hat(preds_ens).numpy())
            # Mean over preds_class instead of y_hat (probs) might be better (for AUROC this seems better, 
            # for F1 not much difference)
            # Using this is the same on CV, because per fold only one model (no ensemble). But on ext_data if 
            # using ensemble of folds this could result in better AUROC.
            # preds_class += list(torch.stack(preds_class_ens).mean(axis=0).numpy())
            preds += list(preds_ens.numpy())

    gts = np.array(gts)
    preds = np.array(preds)
    gts_class = np.array(gts_class)
    preds_class = np.array(preds_class)

    preds_class = np.rint(preds_class)

    return subjects_val, gts_class, preds_class, gts, preds


def run_inference_multi_dataloader(project, exps, subjects_val=None, labels_val=None, img_files=None,
                                   data_folder="data", file_loader=nifti_loader, data_path=None,
                                   probs=False, tta_nr=1):
    """ Will run inference but for each experiment use a new dataloader. 
    (run_inference() will reuse the same dataloader for all experiments to make runtime faster.)

    exps: list of experiments or list of lists of experiments (inner list will be passed to same dataloader). 
    """
    subjects, gts, preds, gts_class, preds_class = [], [], [], [], []
    for exp in exps:
        # print(f"Processing exp {exp}...")
        exp = exp if type(exp) is list else [exp]
        subject, gt_class, pred_class, gt, pred = run_inference(project, exp, subjects_val, labels_val, img_files,
                                                                data_folder, file_loader, data_path,
                                                                probs, tta_nr)
        subjects.append(subject)
        gts.append(gt)
        preds.append(pred)
        gts_class.append(gt_class)
        preds_class.append(pred_class)
    subjects = np.array(subjects)[0]
    gts = np.array(gts)[0]
    preds = np.array(preds).mean(axis=0)
    gts_class = np.array(gts_class)[0]
    preds_class = np.array(preds_class).mean(axis=0).round()
    # Keep mean, instead of rounding to full class. Better probs for AUROC.
    # (for brats_genom this can makes big difference if only ensemble of modalities. If ensemble of 
    # several models (exp_groups), then this make less of a difference)
    # preds_class = np.array(preds_class).mean(axis=0)
    return subjects, gts_class, preds_class, gts, preds


def eval_predictions(gts_class, preds_class, gts, preds):
    from libs.statistics import bootstrap_statistics
    
    # todo for each dataset: set appropriate averaging
    average="macro"  # None|micro|macro

    print(f"First 5 preds: {preds_class[:5]}")
    print(f"\nAll predictions shape: {preds.shape}")
    metrics = get_all_metrics(gts_class, preds_class, average=average)

    bootstrap_n = 10000

    # Calculate confidence intervals
    for metric in ["sensitivity", "specificity", "mcc", "f1"]:
        ci_tmp = bootstrap_statistics(np.stack([gts_class, preds_class], axis=1),
                                        lambda x: get_all_metrics(x[:,0], x[:,1], average=average)[metric],
                                        n_cores=4, n_samples=bootstrap_n)
        metrics[f"{metric}_ci_lower"] = ci_tmp[0]
        metrics[f"{metric}_ci_upper"] = ci_tmp[1]

    # Additional metrics
    if len(preds.shape) == 1:
        # Regression Metrics

        # MAE
        metrics["mae"] = float(np.abs(gts-preds).mean())
        ci_mae = bootstrap_statistics(np.stack([gts, preds], axis=1),
                                      lambda x: np.abs(x[:,0]-x[:,1]).mean(), n_samples=bootstrap_n)
        metrics["mae_ci_lower"] = float(ci_mae[0])
        metrics["mae_ci_upper"] = float(ci_mae[1])

        # RMSE
        metrics["rmse"] = float(np.sqrt(np.square(gts-preds).mean()))
        ci_rmse = bootstrap_statistics(np.stack([gts, preds], axis=1),
                                      lambda x: np.sqrt(np.square(x[:,0]-x[:,1]).mean()), n_samples=bootstrap_n)
        metrics["rmse_ci_lower"] = float(ci_rmse[0])
        metrics["rmse_ci_upper"] = float(ci_rmse[1])

        # Pearson correlation
        metrics["corr"] = float(pearsonr(gts, preds)[0])
        ci_corr = bootstrap_statistics(np.stack([gts, preds], axis=1),
                                       lambda x: pearsonr(x[:, 0], x[:, 1])[0], n_samples=bootstrap_n)
        metrics["corr_ci_lower"] = float(ci_corr[0])
        metrics["corr_ci_upper"] = float(ci_corr[1])

        # Spearman correlation
        metrics["corrspear"] = float(spearmanr(gts, preds)[0])
        ci_corr = bootstrap_statistics(np.stack([gts, preds], axis=1),
                                       lambda x: spearmanr(x[:, 0], x[:, 1])[0], n_samples=bootstrap_n)
        metrics["corrspear_ci_lower"] = float(ci_corr[0])
        metrics["corrspear_ci_upper"] = float(ci_corr[1])

        # R2 score
        metrics["r2"] = float(r2_score(gts, preds))
        ci_r2 = bootstrap_statistics(np.stack([gts, preds], axis=1),
                                       lambda x: r2_score(x[:, 0], x[:, 1]), n_samples=bootstrap_n)
        metrics["r2_ci_lower"] = float(ci_r2[0])
        metrics["r2_ci_upper"] = float(ci_r2[1])
        
    if len(preds.shape) == 2:
        # Raw results to proper probabilities
        if preds.shape[1] == 1:  # from BCE loss: [nr_samples, 1]
            preds_flat = preds[:, 0]
            # somehow in bce case gts in list of tensors, but what roc_auc_score needs numpy array
            print("CONVERTING TENSOR TO NUMPY")
            gts = np.array([e.numpy() for e in gts]).flatten()
        else:  # from CE loss: [nr_samples, 2]
            if preds.shape[1] > 2: raise ValueError("At moment only implemented for 2 classes")
            preds_flat = preds[:, 1]
        # For brats_genom this could be helpful, if also changing how to take mean 
        # in run_inference_multi_dataloader().
        # preds_flat = preds_class  
        metrics["auroc_binary"] = roc_auc_score(gts, preds_flat, average=average)  # only works for binary case

    for k, v in metrics.items():
        if average is None:
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.4f}")
    return metrics


def save_predictions(subjects, gts_class, preds_class, gts, preds, metrics, project, fn):
    output_path = Path(os.environ['lightning_data_jakob']) / project.split("/")[0] / "predictions"

    # Save predictions
    if len(preds.shape) == 1:
        print("Regression: Saving raw output.")
    if len(preds.shape) == 2:
        print(f"Classification: Apply sigmoid or softmax before saving raw predictions ({preds.shape})")
        if preds.shape[1] == 1:  # from BCE loss: [nr_samples, 1]
            # somehow in bce case gts in list of tensors, but what roc_auc_score needs numpy array
            print("CONVERTING TENSOR TO NUMPY 2")
            gts = np.array([e.numpy() for e in gts]).flatten()
            preds = preds[:, 0]
        else:  # from CE loss: [nr_samples, 2]
            preds = preds[:, 1]
    
    df = pd.DataFrame(data=np.array([subjects, gts_class, preds_class, gts, preds.round(2)]).T,
                        columns=["AccessionNumber", "target", "target_predicted", "age", "age_predicted"])
    df.to_excel(output_path / fn, index=False, float_format="%.2f")

    # Save metrics
    metrics_rounded = {}
    for k, v in metrics.items():
        metrics_rounded[k] = [round(e, 3) for e in v] if type(v) is list else round(v, 3)
    json.dump(metrics_rounded, open(output_path / fn.replace(".xlsx", ".json"), "w"), indent=4)
    
    # Save confusion matrix
    # fig = confusion_matrix_to_figure(confusion_matrix(gts_class, preds_class), len(np.unique(gts_class)))
    # fig.savefig(output_path / fn.replace(".xlsx", ".png"), dpi=300)


def append_predictions_to_table(metrics, project, exp_name, fn):
    output_path = Path(os.environ['lightning_data_jakob']) / project.split("/")[0] / "predictions"
    output_path = output_path / fn

    # Only keep subset of metrics
    metrics = {k: v for k, v in metrics.items() if k in ["f1", "auroc_binary", "mae"]}

    if os.path.exists(output_path):
        df = pd.read_excel(output_path)
    else:
        df = pd.DataFrame(columns=["exp_name",]+list(metrics.keys())+["comment",])
    metrics_rounded = {}
    for k, v in metrics.items():
        metrics_rounded[k] = [round(e, 3) for e in v] if type(v) is list else round(v, 3)
    metrics_rounded["exp_name"] = exp_name
    metrics_rounded["comment"] = ""

    if exp_name in df["exp_name"].values:
        print("Experiment already exists in table.")
    else:
        next_index = 0 if df.index.max() is np.nan else df.index.max()+1
        df.loc[next_index] = metrics_rounded
        df.to_excel(output_path, index=False, float_format="%.3f")


if __name__ == '__main__':
    exp_name = sys.argv[1]
    exps = [exp_name]

    # project = "covid/covid_vs_healthy"
    # project = "covid/covid_vs_disease"
    project = "brainage/classify_5_groups"
    # project = "ICB/regress_age"

    subjects, gts_class, preds_class, gts, preds = run_inference(project, exps)

    metrics = eval_predictions(gts_class, preds_class, gts, preds)
    print(metrics)
    # save_predictions(subjects, gts_class, preds_class, gts, preds, metrics, project,
                    #  f"predictions_{exp_name}_TMP.xlsx")


    # Show difference as percentage of value size
    # mask = gts > 0
    # perc = list((np.abs(gts[mask]-preds[mask]) / (gts[mask])).round(2))
    # print(list(zip(gts[mask], preds[mask].round(1), perc)))
