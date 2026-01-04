"""
Copright © 2023 Laboratoire Adhesion et Inflammation, Authored by Remy Torro.
"""

import argparse
import os
import json
from glob import glob
import numpy as np
from art import tprint
from celldetective.signals import SignalDetectionModel
from celldetective.utils.model_loaders import locate_signal_model

tprint("Train")

parser = argparse.ArgumentParser(
    description="Train a signal model from instructions.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("-c", "--config", required=True, help="Training instructions")

args = parser.parse_args()
process_arguments = vars(args)
instructions = str(process_arguments["config"])

if os.path.exists(instructions):
    with open(instructions, "r") as f:
        threshold_instructions = json.load(f)
        threshold_instructions.update(
            {"n_channels": len(threshold_instructions["channel_option"])}
        )
        if threshold_instructions["augmentation_factor"] > 1.0:
            threshold_instructions.update({"augment": True})
        else:
            threshold_instructions.update({"augment": False})
        threshold_instructions.update({"test_split": 0.0})
else:
    print("The configuration path is not valid. Abort.")
    os.abort()

all_classes = []
for d in threshold_instructions["ds"]:
    datasets = glob(d + os.sep + "*.npy")
    for dd in datasets:
        data = np.load(dd, allow_pickle=True)
        classes = np.unique([ddd["class"] for ddd in data])
        all_classes.extend(classes)
all_classes = np.unique(all_classes)
print(all_classes, len(all_classes))

n_classes = len(all_classes)

model_params = {
    k: threshold_instructions[k]
    for k in (
        "pretrained",
        "model_signal_length",
        "channel_option",
        "n_channels",
        "label",
    )
    if k in threshold_instructions
}
model_params.update({"n_classes": n_classes})

train_params = {
    k: threshold_instructions[k]
    for k in (
        "model_name",
        "target_directory",
        "channel_option",
        "recompile_pretrained",
        "test_split",
        "augment",
        "epochs",
        "learning_rate",
        "batch_size",
        "validation_split",
        "normalization_percentile",
        "normalization_values",
        "normalization_clip",
    )
    if k in threshold_instructions
}

print(f"model params {model_params}")
print(f"train params {train_params}")

model = SignalDetectionModel(**model_params)
print(threshold_instructions["ds"])
model.fit_from_directory(threshold_instructions["ds"], **train_params)


# if neighborhood of interest in training instructions, write it in config!
if "neighborhood_of_interest" in threshold_instructions:
    if threshold_instructions["neighborhood_of_interest"] is not None:

        model_path = locate_signal_model(
            threshold_instructions["model_name"], path=None, pairs=True
        )
        complete_path = model_path  # +model
        complete_path = rf"{complete_path}"
        model_config_path = os.sep.join([complete_path, "config_input.json"])
        model_config_path = rf"{model_config_path}"

        f = open(model_config_path)
        config = json.load(f)
        config.update(
            {
                "neighborhood_of_interest": threshold_instructions[
                    "neighborhood_of_interest"
                ],
                "reference_population": threshold_instructions["reference_population"],
                "neighbor_population": threshold_instructions["neighbor_population"],
            }
        )
        json_string = json.dumps(config)
        with open(model_config_path, "w") as outfile:
            outfile.write(json_string)

print("Done.")
