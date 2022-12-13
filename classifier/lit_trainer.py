import sys
from pathlib import Path
p_dir = str(Path(__file__).absolute().parents[1])
if p_dir not in sys.path: sys.path.insert(0, p_dir)

import os
import time
from argparse import ArgumentParser, Namespace
import json
import logging
import random
from functools import partial
import warnings

# from rich.traceback import install
# install(show_locals=False)

import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.utils import shuffle
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data import RandomSampler, WeightedRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
# from pytorch_lightning.callbacks import GPUStatsMonitor  # deprecated
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from p_tqdm import p_map

from datasets import NiftiDataset
from libs.pytorch_utils import get_sample_weights
from lit_classifier import LitClassifier
from transformations import get_transforms
from classifier.utils import get_meta_df, stratified_kfold_train_val_test, unpack_to_npy, get_mean_stats_over_img_files
from classifier.utils_loading import load_lightning_model, load_weights_from_checkpoint
from libs.utils import flatten
from classifier.utils_hparams import args_short_to_long

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # set to 2 to hide all tf warnings
# set to logging.INFO for more infos e.g. nr of model parameters
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# Hide annoying warnings. Remove when installing new versions.
warnings.filterwarnings("ignore")  

def cli_main():
    pl.seed_everything(1234)
    # torch.multiprocessing.set_start_method('spawn')  # needed if Rand3DElastic with cuda

    log_path = Path(os.environ['lightning_results_jakob'])  # "/mnt/nvme/data/lightning_results"
    data_path = Path(os.environ['lightning_data_jakob'])  # "/mnt/nvme/data"

    # args
    parser = ArgumentParser()
    parser.add_argument("-en", "--exp_name", type=str, default="my_test")
    parser.add_argument("-p", "--project", type=str, default="covid/covid_vs_healthy")
    parser.add_argument("-l", "--loss", type=str, default="ce", choices=["ce", "bce", "mse", "mse_cat"])  # mse_cat: train as regression but eval as classification
    parser.add_argument("-bs", "--batch_size", default=32, type=int)
    parser.add_argument("-do", "--dropout", type=float, default=0.1)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)  # 1e-4 1e-6
    parser.add_argument("-dim", "--dim", type=int, default=2, choices=[2, 3])
    parser.add_argument("-ti", "--tiles", type=int, default=0)
    parser.add_argument("-tiss", "--tiles_subsample", type=int, default=3)  # select every nth slice
    parser.add_argument("-tist", "--tiles_start", type=int, default=4)  # skip first and last 1/4 of slices  (higher number means less skipping)
    parser.add_argument("-z", "--zoom", type=str, default="1.0,1.0,1.0")  # use smaller image size; one value per dimension
    parser.add_argument("-ep", "--max_epochs", type=int, default=50)
    parser.add_argument("-nf", "--nr_filt", type=int, default=8)
    parser.add_argument("-nl", "--nr_levels", type=int, default=4)  # unused at the moment
    parser.add_argument("-m", "--model", type=str, default="tf_efficientnet_b0_ns",
                        choices=["densenet", "efficientnet", "resnext", "tf_efficientnet_b7_ns",
                                "tf_efficientnet_b3_ns", "tf_efficientnet_b0_ns", "mobilenetv3_large_100",
                                "efficientnet_3D", "resnet152d", "resnet18", "alexnet", "vgg16",
                                "basiccnn", "cbr_tiny", "cnn_rnn", "cnn_transformer", "enet_hydra",
                                "cbr_tiny_hydra", "chen_net"])
    parser.add_argument("-lr", "--learning_rate", type=float, default=5e-4)
    parser.add_argument("-pre", "--pretrain", type=int, default=1)  # pretrained with imagenet
    parser.add_argument("-pre_exp", "--pretrain_exp", type=str, default="")  # pretrained with specific exp
    parser.add_argument("-sh", "--scheduler", type=str, default="cosine", choices=["cosine", "plateau"])
    parser.add_argument("-es", "--early_stopping", type=int, default=0)
    parser.add_argument("-op", "--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("-mm", "--momentum", type=float, default=0.9)
    parser.add_argument("-wt", "--weight_type", type=str, default="None", choices=["None", "inverse", "manual_2_8", "manual_3_7"])
    parser.add_argument("-ns", "--nr_slices", type=int, default=1)
    parser.add_argument("-mo", "--multi_orientation", type=int, default=0)
    parser.add_argument("-li", "--log_images", type=int, default=0)    
    parser.add_argument("-ss", "--slice_subset", type=int, default=0)    
    # when sampling 2d slices from which orientation to sample
    parser.add_argument("-so", "--slice_orientation", type=str, default="z", choices=["x", "y", "z"])
    parser.add_argument("-det", "--deterministic", type=int, default=0)    

    # Data augmentation
    parser.add_argument("-prob", "--daug_prob", type=float, default=0.3)
    parser.add_argument("-zo", "--daug_zoom", type=int, default=1)
    parser.add_argument("-el", "--daug_elastic", type=int, default=0)
    parser.add_argument("-sft", "--daug_shift", type=int, default=0)
    parser.add_argument("-sc", "--daug_scale", type=int, default=0)
    parser.add_argument("-co", "--daug_contrast", type=int, default=1)
    parser.add_argument("-no", "--daug_noise", type=int, default=0)
    parser.add_argument("-sm", "--daug_smooth", type=int, default=1)
    parser.add_argument("-ro", "--daug_rotate", type=int, default=0)  # requires square crops
    parser.add_argument("-fl", "--daug_flip", type=int, default=1)
    parser.add_argument("-si", "--daug_scale_itensity", type=int, default=0)
    parser.add_argument("-bf", "--daug_bias_field", type=int, default=0)
    parser.add_argument("-sa", "--daug_sharpen", type=int, default=0)
    parser.add_argument("-hi", "--daug_histogram_shift", type=int, default=0)
    parser.add_argument("-gi", "--daug_gibbs_noise", type=int, default=0)
    parser.add_argument("-sn", "--daug_spike_noise", type=int, default=0)
    parser.add_argument("-cu", "--daug_cutout", type=int, default=0)  # throws error with newer monai
    parser.add_argument("-po", "--daug_posterize", type=int, default=0)
    parser.add_argument("-re", "--daug_random_erasing", type=int, default=0)
    parser.add_argument("-rea", "--daug_re_area", type=float, default=0.1)
    parser.add_argument("-rec", "--daug_re_count", type=int, default=5)
    parser.add_argument("-rem", "--daug_re_mode", type=str, default="const", choices=["const", "pixel"])
    parser.add_argument("-cp", "--clip", type=int, default=0)  # will be overwritten by autoset by modality
    parser.add_argument("-cl", "--clip_low", type=int, default=-1000)
    parser.add_argument("-ch", "--clip_high", type=int, default=500)
    parser.add_argument("-n", "--normalize", type=int, default=1)  # will be overwritten by autoset by modality
    parser.add_argument("-nc", "--norm_channel_wise", type=int, default=1)
    parser.add_argument("-gn", "--norm_global", type=int, default=0)   # this also requires normalize=1 
    parser.add_argument("-iz", "--norm_ignore_zero", type=int, default=0)

    # Trainer 
    parser.add_argument("-c", "--cont", type=int, default=0)
    parser.add_argument("-f", "--fold", type=int, default=0)
    parser.add_argument("-ts", "--test_split", type=int, default=0)
    parser.add_argument("--profiler", type=str, default=None, choices=["simple"])
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default=None, choices=[None, "dp", "ddp", "ddp2"])
    parser.add_argument("-npy", "--unpack_to_npy", type=int, default=1)
    parser.add_argument("-if", "--img_files", type=str, nargs='+')  # will overwrite config.json
    parser.add_argument("-cm", "--checkpoint_metric", type=str, default="f1/val", choices=["f1/val", "loss/val", "misc_val/auroc"])
    parser.add_argument("-cd", "--checkpoint_mode", type=str, default="max", choices=["min", "max"])

    # More experimental params for transformers
    # transformer not as robustly working as GRU
    parser.add_argument("-tf_nl", "--tf_nr_layers", type=int, default=1)
    parser.add_argument("-tf_nh", "--tf_nr_heads", type=int, default=4)
    parser.add_argument("-tf_em", "--tf_embed_multiplier", type=float, default=1.0)  # 0.25 or 0.5 for GRU; 1.0 for Transformer
    parser.add_argument("-tf_sm", "--tf_seq_model", type=str, default="GRU", choices=["RNN", "GRU", "LSTM"])
    

    # parser = pl.Trainer.add_argparse_args(parser)  # adds too many options and can not set defaults
    parser = LitClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    # args.gpus = [1]  # select specific gpu by id
    args.accelerator = "gpu"
    args.devices = args.gpus
    del args.gpus
    
    args.deterministic = bool(args.deterministic)

    # Parse project identifier
    project_name, task_name = args.project.split("/")
    data_path = data_path / project_name

    args_set = [args_short_to_long[arg][0] for arg in sys.argv if arg in args_short_to_long]

    if os.path.exists(data_path / task_name / "default_params.json"):
        default_args = json.load(open(data_path / task_name / "default_params.json"))
        print("INFO: Overwriting with default_params.json:")
        for k,v in default_args.items():
            if hasattr(args, k):
                if k in args_set:
                    print(f"\tSkipping '{k}' because is manually set.")
                else:
                    print(f"\t{k}: {v}")
                    setattr(args, k, v)

    # Continue training
    if args.cont:
        model_hp, best_weights = load_lightning_model(log_path / args.project / args.exp_name)
        args = Namespace(**model_hp.hparams)  # this is only needed for dataloaders
        args.resume_from_checkpoint = best_weights  # this will be passed to trainer and make it load checkpoint

    # Load meta data 
    config = json.load(open(data_path / task_name / "dataset.json"))
    meta = get_meta_df(data_path / task_name/  f"meta.xlsx", index_col=config["subject_col"])
    label_col_name = "label_col_reg" if args.loss.startswith("mse") else "label_col_clas"
    meta[config[label_col_name]] = meta[config[label_col_name]].values.astype(str)
    subjects = meta.index.values
    
    if args.loss.startswith("mse"):
        labels = meta[config[label_col_name]].values.astype(np.float32)
    else:
        labels = np.zeros(len(subjects)).astype(np.int32)
        for idx, cl in enumerate(config["classes"].keys()):
            # All values in config are string
            mask = (meta[config[label_col_name]].astype(str) == config["classes"][cl]).values
            labels[mask] = idx

    # Train/val/test split    
    splits_file = data_path / task_name / f"cv_splits.json"
    train_idxs, val_idxs, test_idxs = \
            stratified_kfold_train_val_test(subjects, labels, args.fold, test_split=args.test_split, 
                                            return_idx=True, save_path=splits_file, 
                                            skip_stratification=args.loss.startswith("mse"))

    # S
    # train_idxs = train_idxs[:500]
    # val_idxs = val_idxs[:100]

    # M
    # train_idxs = train_idxs[:1000]
    # val_idxs = val_idxs[:200]

    # Save train/val/test split to hparams for inference
    args.subjects_train, args.subjects_val, args.subjects_test = subjects[train_idxs], subjects[val_idxs], subjects[test_idxs]
    args.labels_train, args.labels_val, args.labels_test = labels[train_idxs], labels[val_idxs], labels[test_idxs]
    if args.img_files is None:
        args.img_files = config["img_files"]
    args.float_to_class_fn = config["float_to_class_fn"] if "float_to_class_fn" in config else None

    # Those are not randomly ordered but in order of the indices in meta.xlsx (because kf.split will
    # return the indices in order even if input unordered). But no problem because random sampler will
    # shuffle it during training and for validation no shuffling needed.
    print(f"First 10 val subjects: {args.subjects_val[:10]}")
    print(f"First 10 val labels: {args.labels_val[:10]}")
    print(f"Nr of train subjects: {len(args.subjects_train)}")

    args.crop_size = config[f"crop_size_{args.dim}d_tiles"] if args.tiles and args.dim == 2 else config[f"crop_size_{args.dim}d"]
    args.zoom = [float(e) for e in args.zoom.split(",")]
    if np.any(np.array(args.zoom) != 1.0):
        print(f"INFO: reducing image size by zoom {args.zoom}")
        args.crop_size = [int(e * args.zoom[idx]) for idx, e in enumerate(args.crop_size)]
        print(f"new crop_size: {args.crop_size}")
    if len(args.crop_size) != args.dim:
        raise ValueError(f"Len of crop_size is not same as dim. (crop_size: {args.crop_size}, dim: {args.dim})")
    args.nr_classes_config = len(config["classes"])  # needed for mse_cat loss
    args.nr_classes = len(config["classes"])
    args.nr_classes = args.nr_classes if not args.loss.startswith("mse") else 1

    if args.dim == 2 and not args.tiles:
        nr_channel_per_img = args.nr_slices*3 if args.multi_orientation else args.nr_slices
    else:
        nr_channel_per_img = 1
    args.nr_channels = nr_channel_per_img * len(args.img_files)  # multipy by nr of modalities

    # Convert to npy if needed
    data_dir_name = "data"
    if args.unpack_to_npy:
        img_name = f"{args.img_files[0].split('.')[0]}.npy"
        all_unpacked = True
        st = time.time()
        for s in subjects:
            if not (data_path / data_dir_name / s / img_name).exists():
                all_unpacked = False
        # print(f"Check unpacking took {time.time()-st:.2f}s")
        if not all_unpacked:
            print("Unpacking...")
            p_map(partial(unpack_to_npy, data_path=data_path/data_dir_name, img_files=args.img_files),
                  subjects, num_cpus=10, disable=True)

    # Transformations
    if "modality" in config:
        dataset_stats = json.load(open(data_path / task_name / "dataset_stats.json"))
        intensity_stats = get_mean_stats_over_img_files(dataset_stats, args.img_files)
        if config["modality"] == "ct":
            print("Use CT normalization: -global_mean /global_std")
            args.normalize = 1
            args.norm_global = 1
            # note: clipping is same for each channel. If channels very different intensities then this must 
            # be changed! But for ct images normally only one channel.
            args.clip = 1
            # args.clip_low = intensity_stats["perc05"]
            # args.clip_high = intensity_stats["perc95"]
            args.clip_low = intensity_stats["perc02"]
            args.clip_high = intensity_stats["perc98"]
            args.global_mean = [[dataset_stats[img_file]["mean"]] * nr_channel_per_img 
                                for img_file in args.img_files]
            args.global_mean = flatten(args.global_mean)
            args.global_std = [[dataset_stats[img_file]["std"]] * nr_channel_per_img
                                for img_file in args.img_files]
            args.global_std = flatten(args.global_std)
        else:
            print("Use MRI/other normalization: -mean /std per image and channel (and no clipping)")
            args.normalize = 1
            args.norm_channel_wise = 1
            args.clip = 0
    tfs_train, tfs_val = get_transforms(args)

    # Datasets
    dataset_train = NiftiDataset(data_path / data_dir_name, args.subjects_train, args.labels_train,
                                 transform=tfs_train, dim=args.dim,
                                 nr_slices=args.nr_slices, multi_orientation=args.multi_orientation,
                                 crop_size=args.crop_size, slice_subset=args.slice_subset,
                                 img_files=args.img_files, loss=args.loss, nr_classes=args.nr_classes,
                                 deterministic=args.deterministic, tiles=args.tiles, 
                                 tiles_subsample=args.tiles_subsample, tiles_start=args.tiles_start,
                                 zoom=args.zoom, slice_orientation=args.slice_orientation,
                                 unpack_to_npy=args.unpack_to_npy)
    dataset_val = NiftiDataset(data_path / data_dir_name, args.subjects_val, args.labels_val,
                               transform=tfs_val, dim=args.dim,
                               nr_slices=args.nr_slices, multi_orientation=args.multi_orientation,
                               crop_size=args.crop_size, slice_subset=args.slice_subset,
                               img_files=args.img_files, loss=args.loss, nr_classes=args.nr_classes,
                               deterministic=True, tiles=args.tiles, 
                               tiles_subsample=args.tiles_subsample, tiles_start=args.tiles_start,
                               zoom=args.zoom, slice_orientation=args.slice_orientation,
                               unpack_to_npy=args.unpack_to_npy)

    if args.weight_type != "None":
        sample_weights = get_sample_weights(args.labels_train, weight_type=args.weight_type)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    else:
        sampler = RandomSampler(dataset_train)

    def worker_init_fn(worker_id):
        # Torch gives unique seed to every worker. By passing this to numpy we can avoid the np random
        # seed error (which would use same seed in each process).
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # If 3D and large batch size than nr_workers needs to be lower; otherwise something is messed up
    # and training is very slow even while utility of CPU and GPU are very low.
    # nr_workers = 10 if args.dim == 2 else 3
    nr_workers = 10
    loader_train = DataLoader(dataset_train, sampler=sampler,
                              batch_size=args.batch_size, num_workers=nr_workers, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    loader_val = DataLoader(dataset_val,
                            batch_size=args.batch_size, num_workers=nr_workers, pin_memory=True,
                            worker_init_fn=worker_init_fn)

    # Lower learning rate for fine-tuning sometimes improves results
    # This has to be before passing args to LitClassifier
    # if args.pretrain_exp != "":
    #     args.learning_rate /= 10
    #     print(f"For Finetuning reducing lr by 10x to {args.learning_rate}")

    # Model
    model = LitClassifier(**vars(args))  # do this before adding the logger to args

    # Pretraining on a significantly larger dataset with a different task seems to help.
    if args.pretrain_exp != "":
        print("Loading existing weights...")
        load_weights_from_checkpoint(model, log_path / args.project / args.pretrain_exp,
                                     del_classifier=True, model_type=args.model)
        # load_weights_from_checkpoint(model, log_path / "ICB/pretrain" / args.pretrain_exp,
                                    #  del_classifier=True, model_type=args.model)
        # backbone.freeze() ?

    # Logging
    log_path = log_path / project_name / task_name
    exp_path = log_path / args.exp_name
    exp_path.mkdir(parents=True, exist_ok=True)  # create dir here otherwise checkpoint stored in parent dir
    logger = TensorBoardLogger(save_dir=log_path, name=args.exp_name)  # run: "tensorboard --logdir=."

    # if args.loss.startswith("mse"):
    #     checkpoint_callback = ModelCheckpoint(dirpath=None, filename='{epoch:03d}', save_top_k=1, monitor='loss/val', mode='min')
    # else:
    #     checkpoint_callback = ModelCheckpoint(dirpath=None, filename='{epoch:03d}', save_top_k=1, monitor='f1/val', mode='max')
    checkpoint_callback = ModelCheckpoint(dirpath=None, filename='{epoch:03d}', save_top_k=1,
                                          monitor=args.checkpoint_metric, mode=args.checkpoint_mode)

    args.logger=logger
    # args.checkpoint_callback=checkpoint_callback
    args.callbacks = [checkpoint_callback]

    if args.early_stopping:
        early_stop_callback=EarlyStopping(monitor='loss/val', min_delta=0, patience=10, mode="min")
        args.callbacks += [early_stop_callback]

    # lr_monitor = LearningRateMonitor(logging_interval='step')
    # gpu_stats = GPUStatsMonitor(memory_utilization=False)  # deprecated
    # gpu_stats = DeviceStatsMonitor()
    # args.callbacks += [gpu_stats]

    # Training
    # torch.use_deterministic_algorithms(True)  # not tried yet (available since pytorch 1.9)
    # args.deterministic = True
    trainer = pl.Trainer.from_argparse_args(args)  # default: shuffle true in train and false in val
    trainer.fit(model, loader_train, loader_val)

    # Final evaluation
    # This will run test with latest epoch, but not with best epoch. This is wrong.
    # trainer.test(dataloaders=loader_val, verbose=False)
    
    # Even after loading checkpoint, self.trainer.checkpoint_callback.best_model_score will properly
    # be set. So this is working fine.
    trainer.test(dataloaders=loader_val, ckpt_path="best", verbose=False)

    # If want to run test without previous fit:
    # (also have to comment 'val = self.trainer.checkpoint_callback.best_model_score' in test_epoch_end() )
    # trainer.test(model=model, dataloaders=loader_val, verbose=False)


if __name__ == '__main__':
    cli_main()
