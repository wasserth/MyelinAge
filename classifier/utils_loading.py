import os
import sys
from pathlib import Path
p_dir = str(Path(__file__).absolute().parents[1])
if p_dir not in sys.path: sys.path.insert(0, p_dir)

import torch 

from classifier.utils_hparams import args_short_to_long


def _get_ckpt_path_from_dir(exp_path: Path):
    version = "version_0"

    # If running all folds for an experiment they are saved as 5 versions and experiment has to 
    # end with _fAll. In run_inference_cv.py "_fX" will be added to make it 5 experiments. Here
    # we have to transform it to the right format.
    # in:  /my/path/3d_t2_run1_fAll_f2/version_0
    # out: /my/path/3d_t2_run1_fAll/version_2    (this is the real file structure)
    if str(exp_path)[-8:-3] == "_fAll":
        version = f"version_{str(exp_path)[-1:]}"
        exp_path = Path(str(exp_path)[:-3])

    print(f'looking for weights in: {str(exp_path / version / "checkpoints")}')
    best_weights = list((exp_path / version / "checkpoints").glob("epoch*"))
    if len(best_weights) > 1: print(f"Found more than 1 checkpoint: {best_weights}")
    return best_weights[0]


def _add_missing_hparams(model):
    hparams_all = {v[0]: v[1] for k, v in args_short_to_long.items()}  # transform to dict param_long_name:default_value
    for param, default_value in hparams_all.items():
        if param not in model.hparams:
            if "lightning_quiet" not in os.environ:
                print(f"Missing hparam '{param}' in stored model. Adding with value {default_value}.")
            model.hparams[param] = default_value
    return model


def load_lightning_model(exp_path: Path):
    from classifier.lit_classifier import LitClassifier
    best_weights_path = _get_ckpt_path_from_dir(exp_path)
    model = LitClassifier.load_from_checkpoint(best_weights_path)
    model = _add_missing_hparams(model)
    return model, best_weights_path
    

def load_weights_from_checkpoint(model,
                                 exp_path: Path,
                                 del_classifier=True,
                                 model_type="tf_efficientnet_b0_ns") -> None:
    """
    Load the weights from a given checkpoint file.
    
    If the checkpoint model architecture is different then `model`, only
    the common parts will be loaded.

    https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices
    """
    best_weights_path = _get_ckpt_path_from_dir(exp_path)

    # Load pytorch-lightning model which is a dict. One key is state_dict.
    # state_dict is the pytorch model as a dict to make it easily editable.
    state_dict = torch.load(best_weights_path)["state_dict"]

    # Delete final layer
    if del_classifier:
        if model_type == "tf_efficientnet_b0_ns":
            # Last layer of timm models
            del state_dict["backbone.classifier.weight"]
            del state_dict["backbone.classifier.bias"]
        elif model_type == "cbr_tiny":
            del state_dict["backbone.dense_1.weight"]
            del state_dict["backbone.dense_1.bias"]
        else:
            raise ValueError("Unrecognised model_type")

    # Replace the old state_dict with the new state_dict.
    # strict=False will allow the dict keys of new state dict to be different from the keys in the model.
    # All keys which do not exactly match will simply be ignored.
    model.load_state_dict(state_dict, strict=False)
