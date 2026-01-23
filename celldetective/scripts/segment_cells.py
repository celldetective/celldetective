"""
Copright © 2022 Laboratoire Adhesion et Inflammation, Authored by Remy Torro.
"""

import argparse
import datetime
import os
import json
from celldetective.utils.model_loaders import locate_segmentation_model
from celldetective.utils.image_loaders import (
    auto_load_number_of_frames,
    _load_frames_to_segment,
    _get_img_num_per_channel,
)
from celldetective.utils.mask_cleaning import _check_label_dims
from celldetective.utils.experiment import extract_position_name
from celldetective.utils.stardist_utils import (
    _prep_stardist_model,
    _segment_image_with_stardist_model,
)
from celldetective.utils.cellpose_utils import (
    _segment_image_with_cellpose_model,
    _prep_cellpose_model,
)
from celldetective.utils.mask_transforms import _rescale_labels
from celldetective.utils.image_transforms import _estimate_scale_factor
from celldetective.utils.parsing import (
    _get_normalize_kwargs_from_config,
    config_section_to_dict,
    _extract_channel_indices_from_config,
    _extract_nbr_channels_from_config,
)
from pathlib import Path, PurePath
from glob import glob
from shutil import rmtree
from tqdm import tqdm
import numpy as np
from celldetective.utils.io import save_tiff_imagej_compatible
import gc
from art import tprint
import concurrent.futures

tprint("Segment")

parser = argparse.ArgumentParser(
    description="Segment a movie in position with the selected model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("-p", "--position", required=True, help="Path to the position")
parser.add_argument("-m", "--model", required=True, help="Model name")
parser.add_argument(
    "--mode",
    default="target",
    choices=["target", "effector", "targets", "effectors"],
    help="Cell population of interest",
)
parser.add_argument(
    "--use_gpu", default="True", choices=["True", "False"], help="use GPU"
)
parser.add_argument("--threads", default="1", help="Number of parallel threads")

args = parser.parse_args()
process_arguments = vars(args)
pos = str(process_arguments["position"])
mode = str(process_arguments["mode"])
use_gpu = process_arguments["use_gpu"]
n_threads = int(process_arguments["threads"])

if use_gpu == "True" or use_gpu == "true" or use_gpu == "1":
    use_gpu = True
    n_threads = 1  # avoid misbehavior on GPU with multithreading
else:
    use_gpu = False
    # n_threads = 1 # force 1 threads since all CPUs seem to be in use anyway

if not use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

modelname = str(process_arguments["model"])

if mode.lower() == "target" or mode.lower() == "targets":
    label_folder = "labels_targets"
elif mode.lower() == "effector" or mode.lower() == "effectors":
    label_folder = "labels_effectors"

# Locate experiment config
parent1 = Path(pos).parent
expfolder = parent1.parent
config = PurePath(expfolder, Path("config.ini"))
assert os.path.exists(
    config
), "The configuration file for the experiment could not be located. Abort."

print(f"Position: {extract_position_name(pos)}...")
print("Configuration file: ", config)
print(f"Population: {mode}...")

####################################
# Check model requirements #########
####################################

modelpath = os.sep.join(
    [os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "models"]
)
model_complete_path = locate_segmentation_model(modelname)
if model_complete_path is None:
    print("Model could not be found. Abort.")
    os.abort()
else:
    print(f"Model path: {model_complete_path}...")

# load config
assert os.path.exists(
    model_complete_path + "config_input.json"
), "The configuration for the inputs to the model could not be located. Abort."
with open(model_complete_path + "config_input.json") as config_file:
    input_config = json.load(config_file)

# Parse target channels
required_channels = input_config["channels"]

channel_indices = _extract_channel_indices_from_config(config, required_channels)
print(
    f"Required channels: {required_channels} located at channel indices {channel_indices}."
)
required_spatial_calibration = input_config["spatial_calibration"]
print(f"Spatial calibration expected by the model: {required_spatial_calibration}...")

normalize_kwargs = _get_normalize_kwargs_from_config(input_config)

model_type = input_config["model_type"]

movie_prefix = config_section_to_dict(config, "MovieSettings")["movie_prefix"]
spatial_calibration = float(config_section_to_dict(config, "MovieSettings")["pxtoum"])
len_movie = float(config_section_to_dict(config, "MovieSettings")["len_movie"])

# Try to find the file
try:
    file = glob(pos + f"movie/{movie_prefix}*.tif")[0]
except IndexError:
    print("Movie could not be found. Check the prefix.")
    os.abort()

len_movie_auto = auto_load_number_of_frames(file)
if len_movie_auto is not None:
    len_movie = len_movie_auto

if model_type == "cellpose":
    diameter = input_config["diameter"]
    # if diameter!=30:
    # 	required_spatial_calibration = None 	# ignore spatial calibration and use diameter
    cellprob_threshold = input_config["cellprob_threshold"]
    flow_threshold = input_config["flow_threshold"]

scale = _estimate_scale_factor(spatial_calibration, required_spatial_calibration)
print(f"Scale: {scale}...")

nbr_channels = _extract_nbr_channels_from_config(config)
# print(f'Number of channels in the input movie: {nbr_channels}')
img_num_channels = _get_img_num_per_channel(
    channel_indices, int(len_movie), nbr_channels
)

# If everything OK, prepare output, load models
if os.path.exists(pos + label_folder):
    print("Erasing the previous labels folder...")
    rmtree(pos + label_folder)
os.mkdir(pos + label_folder)
print(f"Labels folder successfully generated...")

log = f"segmentation model: {modelname}\n"
with open(pos + f"log_{mode}.json", "a") as f:
    f.write(f"{datetime.datetime.now()} SEGMENT \n")
    f.write(log)


# Loop over all frames and segment
def segment_index(indices):

    if model_type == "stardist":
        model, scale_model = _prep_stardist_model(
            modelname, Path(model_complete_path).parent, use_gpu=use_gpu, scale=scale
        )

    elif model_type == "cellpose":
        model, scale_model = _prep_cellpose_model(
            modelname,
            model_complete_path,
            use_gpu=use_gpu,
            n_channels=len(required_channels),
            scale=scale,
        )

    for t in tqdm(indices, desc="frame"):

        f = _load_frames_to_segment(
            file,
            img_num_channels[:, t],
            scale_model=scale_model,
            normalize_kwargs=normalize_kwargs,
        )

        if model_type == "stardist":
            Y_pred = _segment_image_with_stardist_model(
                f, model=model, return_details=False
            )
        elif model_type == "cellpose":
            Y_pred = _segment_image_with_cellpose_model(
                f,
                model=model,
                diameter=diameter,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
            )

        if scale is not None:
            Y_pred = _rescale_labels(Y_pred, scale_model=scale_model)

        Y_pred = _check_label_dims(Y_pred, file)

        save_tiff_imagej_compatible(
            pos + os.sep.join([label_folder, f"{str(t).zfill(4)}.tif"]),
            Y_pred,
            axes="YX",
        )

        del f
        del Y_pred
        gc.collect()

    del model
    gc.collect()

    return


print(f"Starting the segmentation with {n_threads} thread(s) and GPU={use_gpu}...")

# Multithreading
indices = list(range(img_num_channels.shape[1]))
chunks = np.array_split(indices, n_threads)

with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(segment_index, chunks)
    try:
        for i, return_value in enumerate(results):
            print(f"Thread {i} output check: ", return_value)
    except Exception as e:
        print("Exception: ", e)

print("Done.")
gc.collect()
