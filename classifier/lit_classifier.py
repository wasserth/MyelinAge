import sys
from pathlib import Path
p_dir = str(Path(__file__).absolute().parents[1])
if p_dir not in sys.path: sys.path.insert(0, p_dir)

from argparse import ArgumentParser

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import pytorch_lightning as pl
import torchmetrics
import pandas as pd
import numpy as np
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from libs.utils import scale_to_range  
from classifier.models.alexnet import alexnet
from classifier.models.torchvision_overwrite import TorchVisionOverwrite
from classifier.models.basic_cnn import BasicCNN
from classifier.models.chen_net import ChenNet
from classifier.models.cnn_rnn import CNNRNN
from classifier.models.cbr_tiny import CbrTiny
from classifier.models.cnn_transformer import CNNTransformer
from classifier.models.enet_hydra import enet_hydra
from classifier.models.cbr_tiny_hydra import cbr_tiny_hydra
from classifier.utils import confusion_matrix_to_figure
import classifier.dataset_utils


class LitClassifier(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # will save all kwargs to hparams.yml + store into self.hparams
        nr_channels = self.hparams.nr_channels
        # bce: one less than classes, because all 0 is first class
        nr_classes = self.hparams.nr_classes-1 if self.hparams.loss == "bce" else self.hparams.nr_classes
        spatial_dims = self.hparams.dim

        if self.hparams.dim == 3 and not self.hparams.model in ["densenet", "efficientnet_3D", "basiccnn",
                                                                "cbr_tiny", "cnn_rnn", "cnn_transformer",
                                                                "cbr_tiny_hydra", "chen_net"]:
            raise ValueError(f"This model ({self.hparams.model}) does not support 3D.")

        if self.hparams.model == "densenet":
            from monai.networks.nets.densenet import DenseNet121
            self.backbone = densenet121(spatial_dims=spatial_dims, in_channels=nr_channels,
                                        out_channels=nr_classes, dropout_prob=self.hparams.dropout,
                                        pretrained=self.hparams.pretrain)
        elif self.hparams.model == "efficientnet":
            from classifier.models.efficient_net import enet
            self.backbone = enet("efficientnet-b0", in_channels=nr_channels,
                                 out_channels=nr_classes, dropout=self.hparams.dropout)
        elif self.hparams.model == "efficientnet_3D":
            from classifier.models.efficient_net_3d import efficient_net_3d
            self.backbone = efficient_net_3d("efficientnet-b0", in_channels=nr_channels,
                                             out_channels=nr_classes, pretrained=self.hparams.pretrain)
        elif self.hparams.model == "resnext":
            from classifier.models.resnext import ResNext
            self.backbone = ResNext("resnext50_32x4d_ssl", 2)
        elif self.hparams.model == "resnet18":
            self.backbone = TorchVisionOverwrite("resnet18", pretrained=self.hparams.pretrain,
                                                 num_classes=nr_classes, in_chans=nr_channels)
        elif self.hparams.model == "vgg16":
            self.backbone = TorchVisionOverwrite("vgg16", pretrained=self.hparams.pretrain,
                                                 num_classes=nr_classes, in_chans=nr_channels)
        elif self.hparams.model == "alexnet":
            self.backbone = alexnet(
                num_classes=nr_classes, in_chans=nr_channels)
        elif self.hparams.model == "basiccnn":
            self.backbone = BasicCNN(self.hparams.crop_size, dim=spatial_dims, num_classes=nr_classes,
                                     in_chans=nr_channels, dropout=self.hparams.dropout)
        elif self.hparams.model == "chen_net":
            self.backbone = ChenNet(self.hparams.crop_size, dim=spatial_dims, num_classes=nr_classes,
                                     in_chans=nr_channels, dropout=self.hparams.dropout, nr_filt=self.hparams.nr_filt)
        elif self.hparams.model == "cnn_rnn":
            self.backbone = CNNRNN(self.hparams.crop_size, pretrained=self.hparams.pretrain,
                                   num_classes=nr_classes, in_chans=nr_channels,
                                   dropout=self.hparams.dropout, hparams=self.hparams)
        elif self.hparams.model == "cnn_transformer":
            self.backbone = CNNTransformer(self.hparams.crop_size, pretrained=self.hparams.pretrain,
                                           num_classes=nr_classes, in_chans=nr_channels, 
                                           dropout=self.hparams.dropout, hparams=self.hparams)
        elif self.hparams.model == "cbr_tiny":
            self.backbone = CbrTiny(self.hparams.crop_size, dim=spatial_dims, num_classes=nr_classes,
                                    in_chans=nr_channels, nr_filt=self.hparams.nr_filt)
        elif self.hparams.model == "enet_hydra":
            self.backbone = enet_hydra("tf_efficientnet_b0_ns", in_channels=nr_channels,
                                       out_channels=nr_classes, dropout=self.hparams.dropout)
        elif self.hparams.model == "cbr_tiny_hydra":
            self.backbone = cbr_tiny_hydra(self.hparams.crop_size, dim=spatial_dims, num_classes=nr_classes,
                                           in_chans=nr_channels, nr_filt=self.hparams.nr_filt)
        else:
            import timm
            # see available models: timm.list_models("*efficientnet*", pretrained=True)
            self.backbone = timm.create_model(self.hparams.model, pretrained=self.hparams.pretrain,
                                              num_classes=nr_classes, in_chans=nr_channels)

        if self.hparams.loss == "ce":
            # weights = torch.tensor([0.01, 0.1, 3., 3., 8.])  # reweighting of smaller classes
            weights = None
            self.loss_func = torch.nn.CrossEntropyLoss(weight=weights)
        elif self.hparams.loss == "bce":
            self.loss_func = torch.nn.BCEWithLogitsLoss()  # min: 0, max: +inf
        elif self.hparams.loss.startswith("mse"):
            self.loss_func = torch.nn.MSELoss()

        if self.hparams.loss != "mse":
            # different results depending on num_classes:
            # num_classes=1: like sklearn f1_score binary
            # num_classes=2: like sklearn f1_score micro   
            # (for Recall&Precision + lightning>=1.2.0 only num_classes=2 is working)
            self.f1 = torchmetrics.F1Score(num_classes=self.hparams.nr_classes_config)
            self.accuracy_m = torchmetrics.Accuracy()
            self.recall_m = torchmetrics.Recall(num_classes=self.hparams.nr_classes_config)
            self.precision_m = torchmetrics.Precision(num_classes=self.hparams.nr_classes_config)
            # self.auroc_m = torchmetrics.AUROC(num_classes=self.hparams.nr_classes_config)

        self.max_f1 = 0

    def forward(self, x):
        return self.backbone(x)  # use forward for inference/predictions

    def _decode_y(self, y):
        if self.hparams.loss == "ce":
            label = y
        elif self.hparams.loss == "bce":
            label = y.sum(1).int()  # works for ordinal classes as well as binary
        elif self.hparams.loss == "mse":
            label = y
        elif self.hparams.loss == "mse_cat":
            label = getattr(classifier.dataset_utils, self.hparams.float_to_class_fn)(y)
        return label

    def _y_hat_to_prob(self, y):
        if self.hparams.loss == "ce":
            label = y.detach().softmax(dim=1)
        elif self.hparams.loss == "bce":
            label = y.detach().sigmoid()
        elif self.hparams.loss == "mse":
            label = y.detach()  # [bs], float16
        elif self.hparams.loss == "mse_cat":
            label = y.detach()
        return label

    def _decode_y_hat(self, y):
        if self.hparams.loss == "ce":
            # Add softmax before argmax? Only relevant for loss and there it is added automatically; 
            # for argmax results is identical (?)
            label = torch.argmax(y.detach(), dim=1)
        elif self.hparams.loss == "bce":
            label = y.detach().sum(1).round().int()  # works for ordinal classes as well as binary
        elif self.hparams.loss == "mse":
            label = y.detach()  # [bs], float16
        elif self.hparams.loss == "mse_cat":
            label = getattr(classifier.dataset_utils, self.hparams.float_to_class_fn)(y.detach())
        return label

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.current_epoch == 0 and batch_idx == 0:
            print(f"x.shape: {x.shape}")
        y_hat = self.backbone(x)  # [bs, nr_classes]
        y_hat = y_hat[:, 0] if self.hparams.loss.startswith("mse") else y_hat
        loss = self.loss_func(y_hat, y)
        y = self._decode_y(y)
        y_hat = self._y_hat_to_prob(y_hat)
        y_hat = self._decode_y_hat(y_hat)
        self.log('loss/train', loss, on_step=False, on_epoch=True)
        if self.hparams.loss != "mse":
            self.log('f1/train', self.f1(y_hat, y), on_step=False, on_epoch=True)

        if self.hparams.log_images:
            print("WARNING: Logging images")
            nr_imgs = 8  # nr of imgs from batch
            if x.shape[2] > 200:
                x = F.interpolate(x, scale_factor=0.5, mode="nearest")  # reduce img size to make event file smaller
            # remove the rescaling to properly display binary masks
            img = 255 - scale_to_range(x[:nr_imgs], (0, 254))  # [nr_imgs, channels, x, y, (z)]
            if self.hparams.dim == 2:
                # Show channels underneath each other
                img = img.reshape([img.shape[0], 1, img.shape[1]*img.shape[2], img.shape[3]])
            else:
                img = img[:, :, :, :, img.shape[-1] // 2]  # select middle slice in z
                # Show channels underneath each other
                img = img.reshape([img.shape[0], 1, img.shape[1]*img.shape[2], img.shape[3]])
            self.logger.experiment.add_images('data/train', img, dataformats='NCHW')

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat_prob = self.backbone(x)
        y_hat_prob = y_hat_prob[:, 0] if self.hparams.loss.startswith("mse") else y_hat_prob
        loss = self.loss_func(y_hat_prob, y)
        y = self._decode_y(y)
        y_hat_prob = self._y_hat_to_prob(y_hat_prob)
        y_hat = self._decode_y_hat(y_hat_prob)
        self.log('loss/val', loss, on_step=False, on_epoch=True)
        if self.hparams.loss != "mse":
            # This is identical results to first aggregating all y and y_hat in validation_epoch_end and
            # then calculating f1. Not sure how it works. Should give different result when calculating f1
            # per batch and then mean?
            self.log('f1/val', self.f1(y_hat, y), on_step=False, on_epoch=True)
            self.log('misc_val/accuracy', self.accuracy_m(y_hat, y), on_step=False, on_epoch=True)
            self.log('misc_val/sensitivity', self.recall_m(y_hat, y), on_step=False, on_epoch=True)
            self.log('misc_val/precision', self.precision_m(y_hat, y), on_step=False, on_epoch=True)
            # if y.min() == 1:
            #     print("WARNING: All labels 1, therefore setting one to 0 for AUROC to work.")
            #     y[0] = 0

            # This is working but deactivated because shows many warnings. Activate if needed.
            # if self.hparams.loss == "ce":
            #     self.log('misc_val/auroc', self.auroc_m(F.softmax(y_hat_prob, dim=1), y), on_step=False, on_epoch=True)
            # if self.hparams.loss == "bce":
            #     self.log('misc_val/auroc', self.auroc_m(F.sigmoid(y_hat_prob)[:, 0], y), on_step=False, on_epoch=True)

            # Same result as self.auroc_m
            # a = F.softmax(y_hat_prob, dim=1).detach().cpu().numpy()[:, 1]
            # b = y.detach().cpu().numpy()
            # auroc = roc_auc_score(b, a)
            # self.log('misc_val/auroc_skl', auroc, on_step=False, on_epoch=True)
            
        return {"y": y, "y_hat": y_hat}

    def validation_epoch_end(self, outputs):
        current_lr = torch.tensor(self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log("loss/current_lr", current_lr)

        if self.hparams.loss != "mse":
            # Important note: this does not correctly work for Multi-GPU training
            preds = torch.cat([step['y_hat'] for step in outputs])
            targets = torch.cat([step['y'] for step in outputs])
            f1 = torchmetrics.functional.f1_score(preds, targets, num_classes=self.hparams.nr_classes_config).item()
            if self.current_epoch > 0:  # skip validation sanity check (and first epoch)
                self.max_f1 = max(self.max_f1, f1)
            self.log("f1/val_max", self.max_f1)

    def test_step(self, batch, batch_idx):
        x, y_raw = batch
        y_hat_raw = self.backbone(x)
        y_hat_raw = y_hat_raw[:, 0] if self.hparams.loss.startswith("mse") else y_hat_raw
        loss = self.loss_func(y_hat_raw, y_raw)
        y = self._decode_y(y_raw)
        y_hat_raw = self._y_hat_to_prob(y_hat_raw)
        y_hat = self._decode_y_hat(y_hat_raw)

        if self.hparams.log_images:
            print("WARNING: Logging images")
            x = x[y_hat != y]  # select wrongly classified samples
            print(f"Found {x.shape[0]} wrong classified images in batch.")
            nr_imgs = 5  # nr of imgs from batch
            x = F.interpolate(x, scale_factor=0.5, mode="nearest")  # reduce img size to make event file smaller
            img = 255 - scale_to_range(x[:nr_imgs], (0, 255))
            if self.hparams.dim == 2:
                # Show channels underneath each other
                img = img.reshape([img.shape[0], 1, img.shape[1]*img.shape[2], img.shape[3]])
            else:
                img = img[:, :1, :, :, img.shape[-1] // 2] 
            self.logger.experiment.add_images('wrongly_classified', img, dataformats='NCHW', global_step=batch_idx)

        return {"y_raw": y_raw, "y_hat_raw": y_hat_raw, "y": y, "y_hat": y_hat}

    def test_epoch_end(self, outputs):
        # All the evaluations we want to do on the final model. This is also run on the validation set.
        # 
        # Important note: This only works properly for single GPU training. For Multi-GPU we would have
        # to aggregate results accross GPUs. (e.g. try to use the metrics class API)

        y_hat_raw = torch.cat([step['y_hat_raw'] for step in outputs])
        y_raw = torch.cat([step['y_raw'] for step in outputs])
        y_hat = torch.cat([step['y_hat'] for step in outputs])
        y = torch.cat([step['y'] for step in outputs])

        if self.hparams.loss != "mse":
            # Plot confusion matrix
            confusion_matrix = torchmetrics.functional.confusion_matrix(y_hat, y, num_classes=self.hparams.nr_classes_config)
            fig_ = confusion_matrix_to_figure(confusion_matrix.detach().cpu().numpy(), self.hparams.nr_classes_config)
            self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

            # Print f1 per class
            s = ""
            f1_all = torchmetrics.functional.f1_score(y_hat, y, num_classes=self.hparams.nr_classes_config).item()
            s += f"All: {f1_all:.3f}\n  "
            for class_idx in range(self.hparams.nr_classes_config):
                # different results depending on num_classes:
                # num_classes=1: like sklearn f1_score binary
                # num_classes=2: like sklearn f1_score micro  
                f1 = torchmetrics.functional.f1_score((y_hat == class_idx).int(),
                                                (y == class_idx).int(), num_classes=2).item()
                s += f"{class_idx}: {f1:.3f}\n  "
            self.logger.experiment.add_text("F1 per class", s, self.current_epoch)
        
        if self.hparams.loss.startswith("mse"):
            mse = torchmetrics.functional.mean_squared_error(y_hat_raw, y_raw).item()
            # This is very close to run_inference.py -> mae, but not completely identical -> why?
            mae = torchmetrics.functional.mean_absolute_error(y_hat_raw, y_raw).item()  # identical to: torch.abs(y_hat_raw-y_raw).mean()
            self.logger.experiment.add_text("MSE", f"MSE: {mse:.4f}, MAE: {mae:.4f}", self.current_epoch)

        # update hp_metric
        # val = self.trainer.checkpoint_callback.best_model_score
        # self.log('hp_metric', val)  # do not remove, because only done here
        if self.hparams.loss.startswith("mse"):
            self.log('hp_metric', mae)  # mae != sqrt(mse)
        else:
            self.log('hp_metric', f1_all)

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate,
                                        weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate,
                                        momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay)

        if self.hparams.scheduler == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5,
                                          cooldown=0, min_lr=1e-8)
            scheduler_config = {
                'scheduler': scheduler,
                'reduce_on_plateau': True,  # For ReduceLROnPlateau scheduler
                'monitor': 'loss/val',  # Metric for ReduceLROnPlateau to monitor
            }
        elif self.hparams.scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, self.hparams.max_epochs)
            scheduler_config = {'scheduler': scheduler}
            
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler_config,
            'monitor': 'loss/val'
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--my_test', type=float, default=1e-4)
        return parser
