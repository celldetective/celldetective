"""
Copright © 2023 Laboratoire Adhesion et Inflammation, Authored by Remy Torro.
"""

import argparse
import os
import sys
import shutil
from typing import Dict, Any
from glob import glob
import json
from tqdm import tqdm
import numpy as np
import random

from celldetective.utils.image_augmenters import augmenter
from celldetective.utils.image_loaders import load_image_dataset
from celldetective.utils.image_cleaning import interpolate_nan
from celldetective.utils.normalization import normalize_multichannel
from celldetective.utils.mask_cleaning import fill_label_holes
from celldetective.utils.image_transforms import pad_dataset_to_patch_size
from celldetective.utils.io import make_json_safe
from celldetective.utils.model_loaders import freeze_model_encoder
from art import tprint
from distutils.dir_util import copy_tree
import logging

logger = logging.getLogger("celldetective")


def save_json(data: Dict[str, Any], fpath: str, **kwargs: Any) -> None:
    """
    Save dictionary to JSON file.

    Parameters
    ----------
    data : dict
        Data to save.
    fpath : str
        File path.
    **kwargs
        Additional keyword arguments for json.dumps.
    """
    with open(fpath, "w") as f:
        f.write(json.dumps(data, **kwargs))


tprint("Train")

parser = argparse.ArgumentParser(
    description="Train a signal model from instructions.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("-c", "--config", required=True, help="Training instructions")
parser.add_argument("-g", "--use_gpu", required=True, help="Use GPU")

args = parser.parse_args()
process_arguments = vars(args)
instructions = str(process_arguments["config"])
use_gpu = bool(process_arguments["use_gpu"])

if os.path.exists(instructions):
    with open(instructions, "r") as f:
        training_instructions = json.load(f)
else:
    logger.error("Training instructions could not be found. Abort.")
    sys.exit(1)

model_name = training_instructions["model_name"]
target_directory = training_instructions["target_directory"]
model_type = training_instructions["model_type"]
pretrained = training_instructions["pretrained"]
if pretrained == "":
    pretrained = None

datasets = training_instructions["ds"]

target_channels = training_instructions["channel_option"]
normalization_percentile = training_instructions["normalization_percentile"]
normalization_clip = training_instructions["normalization_clip"]
normalization_values = training_instructions["normalization_values"]
spatial_calibration = training_instructions["spatial_calibration"]

validation_split = training_instructions["validation_split"]
augmentation_factor = training_instructions["augmentation_factor"]

learning_rate = training_instructions["learning_rate"]
epochs = training_instructions["epochs"]
batch_size = training_instructions["batch_size"]


# Load dataset
logger.info(f"Datasets: {datasets}")
X, Y, filenames = load_image_dataset(
    datasets,
    target_channels,
    train_spatial_calibration=spatial_calibration,
    mask_suffix="labelled",
)
logger.info("Dataset loaded...")

values = []
percentiles = []
for k in range(len(normalization_percentile)):
    if normalization_percentile[k]:
        percentiles.append(normalization_values[k])
        values.append(None)
    else:
        percentiles.append(None)
        values.append(normalization_values[k])

X = [
    normalize_multichannel(
        x, **{"percentiles": percentiles, "values": values, "clip": normalization_clip}
    )
    for x in X
]

for k in range(len(X)):
    x = X[k].copy()
    x_interp = np.moveaxis(
        [interpolate_nan(x[:, :, c].copy()) for c in range(x.shape[-1])], 0, -1
    )
    X[k] = x_interp

Y = [fill_label_holes(y) for y in tqdm(Y)]

if len(X) <= 1:
    raise ValueError("not enough training data")
rng = np.random.RandomState()
ind = rng.permutation(len(X))
n_val = max(1, int(round(validation_split * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]

files_train = [filenames[i] for i in ind_train]
files_val = [filenames[i] for i in ind_val]

logger.info(f"number of images: {len(X):3d}")
logger.info(f"- training:       {len(X_trn):3d}")
logger.info(f"- validation:     {len(X_val):3d}")

if model_type == "cellpose":

    # do augmentation in place
    X_aug = []
    Y_aug = []
    n_val = max(1, int(round(augmentation_factor * len(X_trn))))
    indices = random.choices(list(np.arange(len(X_trn))), k=n_val)
    logger.info("Performing image augmentation pre-training...")
    for i in tqdm(indices):
        x_aug, y_aug = augmenter(X_trn[i], Y_trn[i])
        X_aug.append(x_aug)
        Y_aug.append(y_aug)

    # Channel axis in front for cellpose_utils
    X_aug = [np.moveaxis(x, -1, 0) for x in X_aug]
    X_val = [np.moveaxis(x, -1, 0) for x in X_val]
    logger.info(f"number of augmented images: {len(X_aug):3d}")

    from cellpose.models import CellposeModel
    from cellpose.io import logger_setup
    import torch

    if not use_gpu:
        logger.info("Using CPU for training...")
        device = torch.device("cpu")
    else:
        logger.info("Using GPU for training...")

    diam_mean = 30.0
    _cellpose_logger, log_file = logger_setup()
    logger.info(f"Pretrained model: {pretrained}")
    if pretrained is not None:
        if pretrained.endswith("CP_nuclei"):
            diam_mean = 17.0
        pretrained_path = os.sep.join([pretrained, os.path.split(pretrained)[-1]])
    else:
        pretrained_path = pretrained

    model = CellposeModel(
        gpu=use_gpu,
        model_type=None,
        pretrained_model=pretrained_path,
        diam_mean=diam_mean,
        nchan=X_aug[0].shape[0],
    )
    for name, module in model.net.named_children():
        logger.debug(f"{name} {type(module)}")

    if pretrained is not None:
        freeze_model_encoder(model, "cellpose")

    # Now train normally (Cellpose will internally skip frozen params)
    model.train(
        train_data=X_aug,
        train_labels=Y_aug,
        normalize=False,
        channels=None,
        batch_size=batch_size,
        min_train_masks=1,
        save_path=target_directory + os.sep + model_name,
        n_epochs=epochs,
        model_name=model_name,
        learning_rate=learning_rate,
        test_data=X_val,
        test_labels=Y_val,
    )

    file_to_move = glob(os.sep.join([target_directory, model_name, "models", "*"]))[0]
    shutil.move(
        file_to_move,
        os.sep.join([target_directory, model_name, ""])
        + os.path.split(file_to_move)[-1],
    )
    os.rmdir(os.sep.join([target_directory, model_name, "models"]))

    diameter = model.diam_labels

    if pretrained is not None and os.path.split(pretrained)[-1] == "CP_nuclei":
        standard_diameter = 17.0
    else:
        standard_diameter = 30.0

    input_spatial_calibration = spatial_calibration  # *diameter / standard_diameter

    config_inputs = {
        "channels": target_channels,
        "diameter": standard_diameter,
        "cellprob_threshold": 0.0,
        "flow_threshold": 0.4,
        "normalization_percentile": normalization_percentile,
        "normalization_clip": normalization_clip,
        "normalization_values": normalization_values,
        "model_type": "cellpose",
        "spatial_calibration": input_spatial_calibration,
        "cell_size_um": round(diameter * input_spatial_calibration, 4),
        "dataset": {"train": files_train, "validation": files_val},
    }
    json_input_config = json.dumps(config_inputs, indent=4, default=make_json_safe)
    with open(
        os.sep.join([target_directory, model_name, "config_input.json"]), "w"
    ) as outfile:
        outfile.write(json_input_config)

elif model_type == "stardist":

    from stardist import calculate_extents, gputools_available
    from stardist.models import Config2D, StarDist2D

    n_rays = 32
    logger.debug(f"gputools_available={gputools_available()}")

    n_channel = X_trn[0].shape[-1]

    # Predict on subsampled grid for increased efficiency and larger field of view
    grid = (2, 2)
    conf = Config2D(
        n_rays=n_rays,
        grid=grid,
        use_gpu=use_gpu,
        n_channel_in=n_channel,
        train_learning_rate=learning_rate,
        train_patch_size=(256, 256),
        train_epochs=epochs,
        train_reduce_lr={"factor": 0.1, "patience": 30, "min_delta": 0},
        train_batch_size=batch_size,
        train_steps_per_epoch=int(augmentation_factor * len(X_trn)),
    )

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory

        limit_gpu_memory(None, allow_growth=True)

    if pretrained is None:
        model = StarDist2D(conf, name=model_name, basedir=target_directory)
    else:

        os.rename(
            instructions, os.sep.join([target_directory, model_name, "temp.json"])
        )
        copy_tree(pretrained, os.sep.join([target_directory, model_name]))

        if os.path.exists(
            os.sep.join([target_directory, model_name, "training_instructions.json"])
        ):
            os.remove(
                os.sep.join(
                    [target_directory, model_name, "training_instructions.json"]
                )
            )
        if os.path.exists(
            os.sep.join([target_directory, model_name, "config_input.json"])
        ):
            os.remove(os.sep.join([target_directory, model_name, "config_input.json"]))
        if os.path.exists(os.sep.join([target_directory, model_name, "logs" + os.sep])):
            shutil.rmtree(os.sep.join([target_directory, model_name, "logs"]))
        os.rename(
            os.sep.join([target_directory, model_name, "temp.json"]),
            os.sep.join([target_directory, model_name, "training_instructions.json"]),
        )

        # shutil.copytree(pretrained, os.sep.join([target_directory, model_name]))
        model = StarDist2D(None, name=model_name, basedir=target_directory)
        model.config.train_epochs = epochs
        model.config.train_batch_size = min(len(X_trn), batch_size)
        model.config.train_learning_rate = (
            learning_rate  # perf seems bad if lr is changed in transfer
        )
        model.config.use_gpu = use_gpu
        model.config.train_reduce_lr = {"factor": 0.1, "patience": 10, "min_delta": 0}
        logger.debug(f"model.config={model.config}")

        save_json(
            vars(model.config),
            os.sep.join([target_directory, model_name, "config.json"]),
        )

    if pretrained is not None:
        freeze_model_encoder(model, "stardist")

    # Check and pad training/validation images/labels if smaller than patch size
    train_patch_size = getattr(model.config, "train_patch_size", (256, 256))
    patch_h, patch_w = train_patch_size[0], train_patch_size[1]

    X_trn, Y_trn, padded_trn_count = pad_dataset_to_patch_size(
        X_trn, Y_trn, patch_h, patch_w
    )
    X_val, Y_val, padded_val_count = pad_dataset_to_patch_size(
        X_val, Y_val, patch_h, patch_w
    )

    if padded_trn_count > 0 or padded_val_count > 0:
        logger.info(
            f"StarDist training: Padded {padded_trn_count} training images and "
            f"{padded_val_count} validation images to match train_patch_size {train_patch_size} using centered constant padding."
        )

    median_size = calculate_extents(list(Y_trn), np.mean)
    fov = np.array(model._axes_tile_overlap("YX"))
    logger.info(f"median object size:      {median_size}")
    logger.info(f"network field of view :  {fov}")

    current_depth = getattr(model.config, "unet_n_depth", 3)
    initial_depth = current_depth
    max_depth = initial_depth + 3
    while pretrained is None and any(median_size > fov):
        if current_depth >= max_depth:
            break
        new_depth = current_depth + 1
        logger.info(
            f"Auto-adjusting StarDist U-Net depth: median object size {median_size} "
            f"exceeds network field of view {fov}. Increasing unet_n_depth from {current_depth} to {new_depth}."
        )
        conf = Config2D(
            n_rays=n_rays,
            grid=grid,
            use_gpu=use_gpu,
            n_channel_in=n_channel,
            train_learning_rate=learning_rate,
            train_patch_size=tuple(train_patch_size),
            train_epochs=epochs,
            train_reduce_lr={"factor": 0.1, "patience": 30, "min_delta": 0},
            train_batch_size=batch_size,
            train_steps_per_epoch=int(augmentation_factor * len(X_trn)),
            unet_n_depth=new_depth,
        )
        model = StarDist2D(conf, name=model_name, basedir=target_directory)
        fov = np.array(model._axes_tile_overlap("YX"))
        logger.info(f"new network field of view :  {fov}")
        current_depth = new_depth

    if any(median_size > fov):
        logger.warning(
            "median object size larger than field of view of the neural network."
        )

    if pretrained is not None:

        mod = model.keras_model
        encoder_depth = len(mod.layers) // 2

        for layer in mod.layers[:encoder_depth]:
            layer.trainable = False

        # Keep decoder trainable
        for layer in mod.layers[encoder_depth:]:
            layer.trainable = True

    if augmentation_factor == 1.0:
        model.train(X_trn, Y_trn, validation_data=(X_val, Y_val))
    else:
        model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter)
    model.optimize_thresholds(X_val, Y_val)

    if isinstance(median_size, list):
        median_size = np.mean(median_size)

    config_inputs = {
        "channels": target_channels,
        "normalization_percentile": normalization_percentile,
        "normalization_clip": normalization_clip,
        "normalization_values": normalization_values,
        "model_type": "stardist",
        "spatial_calibration": spatial_calibration,
        "cell_size_um": median_size * spatial_calibration,
        "dataset": {"train": files_train, "validation": files_val},
    }

    json_input_config = json.dumps(config_inputs, indent=4, default=make_json_safe)
    with open(
        os.sep.join([target_directory, model_name, "config_input.json"]), "w"
    ) as outfile:
        outfile.write(json_input_config)

logger.info("Done.")
