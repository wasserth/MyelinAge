import sys
from pathlib import Path
p_dir = str(Path(__file__).absolute().parents[1])
if p_dir not in sys.path: sys.path.insert(0, p_dir)

import numpy as np


args_short_to_long = {
    "-l": ("loss", 0),
    "-bs": ("batch_size", 0),
    "-do": ("dropout", 0),
    "-wd": ("weight_decay", 0),
    "-dim": ("dim", 0),
    "-ti": ("tiles", 0),
    "-ep": ("max_epochs", 0),
    "-nf": ("nr_filt", 0),
    "-nl": ("nr_levels", 0),
    "-m": ("model", 0),
    "-lr": ("learning_rate", 0),
    "-pre": ("pretrain", 0),
    "-sh": ("scheduler", 0),
    "-es": ("early_stopping", 0),
    "-op": ("optimizer", 0),
    "-mm": ("momentum", 0),
    "-wt": ("weight_type", 0),
    "-ns": ("nr_slices", 0),
    "-mo": ("multi_orientation", 0),
    "-li": ("log_images", 0),
    "-ss": ("slice_subset", 0),
    "-prob": ("daug_prob", 0),
    "-zo": ("daug_zoom", 0),
    "-el": ("daug_elastic", 0),
    "-sft": ("daug_shift", 0),
    "-sc": ("daug_scale", 0),
    "-co": ("daug_contrast", 0),
    "-no": ("daug_noise", 0),
    "-sm": ("daug_smooth", 0),
    "-ro": ("daug_rotate", 0),
    "-fl": ("daug_flip", 0),
    "-si": ("daug_scale_itensity", 0),
    "-bf": ("daug_bias_field", 0),
    "-sa": ("daug_sharpen", 0),
    "-hi": ("daug_histogram_shift", 0),
    "-gi": ("daug_gibbs_noise", 0),
    "-sn": ("daug_spike_noise", 0),
    "-cu": ("daug_cutout", 0),
    "-po": ("daug_posterize", 0),
    "-si": ("daug_smooth", 0),
    "-cp": ("clip", 0),
    "-cl": ("clip_low", 0),
    "-ch": ("clip_high", 0),
    "-n": ("normalize", 0),
    "-nc": ("norm_channel_wise", 0),
    "-gn": ("norm_global", 0),
    "-gm": ("global_mean", 0),
    "-gs": ("global_std", 0),
    "-npy": ("unpack_to_npy", 0),
    "-if": ("img_files", 0),
    "-cm": ("checkpoint_metric", 0),
    "-cd": ("checkpoint_mode", 0),
    "-tf_nl": ("tf_nr_layers", 0),
    "-tf_nh": ("tf_nr_heads", 0),
    "-tf_em": ("tf_embed_multiplier", 0),
    "-tf_sm": ("tf_seq_model", 0),
    "-tiss": ("tiles_subsample", 3),
    "-tist": ("tiles_start", 4),
    "-z": ("zoom", 1.0),
    "-iz": ("norm_ignore_zero", 0),
    "-re": ("daug_random_erasing", 0),
    "-rea": ("daug_re_area", 0.33),
    "-rec": ("daug_re_count", 3),
    "-rem": ("daug_re_mode", "const")
}
