"""
Copright © 2022 Laboratoire Adhesion et Inflammation, Authored by Remy Torro.
"""

import argparse
import os
import json
from celldetective.utils.image_loaders import (
    auto_load_number_of_frames,
    load_frames,
    _get_img_num_per_channel,
)
from celldetective.utils.experiment import (
    extract_position_name,
    extract_experiment_channels,
)
from celldetective.segmentation import segment_frame_from_thresholds
from celldetective.utils.parsing import (
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
    description="Segment a movie in position with a threshold pipeline",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("-p", "--position", required=True, help="Path to the position")
parser.add_argument("-c", "--config", required=True, help="Threshold instructions")
parser.add_argument(
    "--mode",
    default="target",
    choices=["target", "effector", "targets", "effectors"],
    help="Cell population of interest",
)
parser.add_argument("--threads", default="1", help="Number of parallel threads")

args = parser.parse_args()
process_arguments = vars(args)
pos = str(process_arguments["position"])
mode = str(process_arguments["mode"])
n_threads = int(process_arguments["threads"])

threshold_instructions = str(process_arguments["config"])
equalize = False

if os.path.exists(threshold_instructions):
    with open(threshold_instructions, "r") as f:
        threshold_instructions = json.load(f)
        required_channels = [threshold_instructions["target_channel"]]
        if "equalize_reference" in threshold_instructions:
            equalize_info = threshold_instructions["equalize_reference"]
            equalize = equalize_info[0]
            equalize_time = equalize_info[1]

else:
    print("The configuration path is not valid. Abort.")
    os.abort()

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

channel_indices = _extract_channel_indices_from_config(config, required_channels)
# need to abort if channel not found
print(
    f"Required channels: {required_channels} located at channel indices {channel_indices}..."
)

threshold_instructions.update({"target_channel": channel_indices[0]})

movie_prefix = config_section_to_dict(config, "MovieSettings")["movie_prefix"]
len_movie = float(config_section_to_dict(config, "MovieSettings")["len_movie"])
channel_names, channel_indices = extract_experiment_channels(expfolder)
threshold_instructions.update({"channel_names": channel_names})

# Try to find the file
try:
    file = glob(pos + f"movie/{movie_prefix}*.tif")[0]
except IndexError:
    print("Movie could not be found. Check the prefix.")
    os.abort()

len_movie_auto = auto_load_number_of_frames(file)
if len_movie_auto is not None:
    len_movie = len_movie_auto

nbr_channels = _extract_nbr_channels_from_config(config)
# print(f'Number of channels in the input movie: {nbr_channels}')
img_num_channels = _get_img_num_per_channel(
    np.arange(nbr_channels), len_movie, nbr_channels
)

# If everything OK, prepare output, load models
if os.path.exists(os.sep.join([pos, label_folder])):
    print("Erasing the previous labels folder...")
    rmtree(os.sep.join([pos, label_folder]))
os.mkdir(os.sep.join([pos, label_folder]))
print(f"Labels folder successfully generated...")

if equalize:
    f_reference = load_frames(
        img_num_channels[:, equalize_time], file, scale=None, normalize_input=False
    )
    f_reference = f_reference[:, :, threshold_instructions["target_channel"]]
else:
    f_reference = None

threshold_instructions.update({"equalize_reference": f_reference})
print(f"Instructions: {threshold_instructions}...")


# Loop over all frames and segment
def segment_index(indices):

    for t in tqdm(indices, desc="frame"):

        # Load channels at time t
        f = load_frames(img_num_channels[:, t], file, scale=None, normalize_input=False)
        mask = segment_frame_from_thresholds(f, **threshold_instructions)
        save_tiff_imagej_compatible(
            os.sep.join([pos, label_folder, f"{str(t).zfill(4)}.tif"]),
            mask.astype(np.uint16),
            axes="YX",
        )

        del f
        del mask
        gc.collect()

    return


print(f"Starting the segmentation with {n_threads} thread(s)...")

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
