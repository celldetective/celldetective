"""
Copright © 2022 Laboratoire Adhesion et Inflammation, Authored by Remy Torro.
"""

import argparse
import datetime
import os
import json
from celldetective.utils.data_loaders import interpret_tracking_configuration
from celldetective.utils.image_loaders import (
    locate_labels,
    auto_load_number_of_frames,
    _load_frames_to_measure,
    _get_img_num_per_channel,
)
from celldetective.utils.experiment import (
    extract_position_name,
    extract_experiment_channels,
)
from celldetective import (
    extract_experiment_channels,
)
from celldetective.utils.data_cleaning import _mask_intensity_measurements
from celldetective.utils.parsing import config_section_to_dict
from celldetective.measure import drop_tonal_features, measure_features
from celldetective.tracking import track
from pathlib import Path, PurePath
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import gc
import os
from natsort import natsorted
from art import tprint
import concurrent.futures


tprint("Track")

parser = argparse.ArgumentParser(
    description="Segment a movie in position with the selected model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("-p", "--position", required=True, help="Path to the position")
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

if not os.path.exists(pos + "output"):
    os.mkdir(pos + "output")

if not os.path.exists(pos + os.sep.join(["output", "tables"])):
    os.mkdir(pos + os.sep.join(["output", "tables"]))

if mode.lower() == "target" or mode.lower() == "targets":
    label_folder = "labels_targets"
    instruction_file = os.sep.join(["configs", "tracking_instructions_targets.json"])
    napari_name = "napari_target_trajectories.npy"
    table_name = "trajectories_targets.csv"

elif mode.lower() == "effector" or mode.lower() == "effectors":
    label_folder = "labels_effectors"
    instruction_file = os.sep.join(["configs", "tracking_instructions_effectors.json"])
    napari_name = "napari_effector_trajectories.npy"
    table_name = "trajectories_effectors.csv"

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

# from exp config fetch spatial calib, channel names
movie_prefix = config_section_to_dict(config, "MovieSettings")["movie_prefix"]
spatial_calibration = float(config_section_to_dict(config, "MovieSettings")["pxtoum"])
time_calibration = float(config_section_to_dict(config, "MovieSettings")["frametomin"])
len_movie = float(config_section_to_dict(config, "MovieSettings")["len_movie"])
shape_x = int(config_section_to_dict(config, "MovieSettings")["shape_x"])
shape_y = int(config_section_to_dict(config, "MovieSettings")["shape_y"])

channel_names, channel_indices = extract_experiment_channels(expfolder)
nbr_channels = len(channel_names)

# from tracking instructions, fetch btrack config, features, haralick, clean_traj, idea: fetch custom timeline?
print("Looking for tracking instruction file...")
instr_path = PurePath(expfolder, Path(f"{instruction_file}"))
if os.path.exists(instr_path):
    print(f"Tracking instruction file successfully loaded...")
    with open(instr_path, "r") as f:
        instructions = json.load(f)
    btrack_config = interpret_tracking_configuration(instructions["btrack_config_path"])

    if "features" in instructions:
        features = instructions["features"]
    else:
        features = None

    if "mask_channels" in instructions:
        mask_channels = instructions["mask_channels"]
    else:
        mask_channels = None

    if "haralick_options" in instructions:
        haralick_options = instructions["haralick_options"]
    else:
        haralick_options = None

    if "post_processing_options" in instructions:
        post_processing_options = instructions["post_processing_options"]
    else:
        post_processing_options = None

    btrack_option = True
    if "btrack_option" in instructions:
        btrack_option = instructions["btrack_option"]
    search_range = None
    if "search_range" in instructions:
        search_range = instructions["search_range"]
    memory = None
    if "memory" in instructions:
        memory = instructions["memory"]
else:
    print(
        "Tracking instructions could not be located... Using a standard bTrack motion model instead..."
    )
    btrack_config = interpret_tracking_configuration(None)
    features = None
    mask_channels = None
    haralick_options = None
    post_processing_options = None
    btrack_option = True
    memory = None
    search_range = None
if features is None:
    features = []

# from pos fetch labels
label_path = natsorted(glob(pos + f"{label_folder}" + os.sep + "*.tif"))
if len(label_path) > 0:
    print(f"Found {len(label_path)} segmented frames...")
else:
    print(
        f"No segmented frames have been found. Please run segmentation first. Abort..."
    )
    os.abort()

# Do this if features or Haralick is not None, else don't need stack
try:
    file = glob(pos + os.sep.join(["movie", f"{movie_prefix}*.tif"]))[0]
except IndexError:
    print(
        "Movie could not be found. Check the prefix. If you intended to measure texture or tone, this will not be performed."
    )
    file = None
    haralick_option = None
    features = drop_tonal_features(features)

len_movie_auto = auto_load_number_of_frames(file)
if len_movie_auto is not None:
    len_movie = len_movie_auto

img_num_channels = _get_img_num_per_channel(channel_indices, len_movie, nbr_channels)

#######################################
# Loop over all frames and find objects
#######################################

timestep_dataframes = []
features_log = f"features: {features}"
mask_channels_log = f"mask_channels: {mask_channels}"
haralick_option_log = f"haralick_options: {haralick_options}"
post_processing_option_log = f"post_processing_options: {post_processing_options}"
log_list = [
    features_log,
    mask_channels_log,
    haralick_option_log,
    post_processing_option_log,
]
log = "\n".join(log_list)

with open(pos + f"log_{mode}.json", "a") as f:
    f.write(f"{datetime.datetime.now()} TRACK \n")
    f.write(log + "\n")


if not btrack_option:
    features = []
    channel_names = None
    haralick_options = None


def measure_index(indices):

    props = []

    for t in tqdm(indices, desc="frame"):

        # Load channels at time t
        img = _load_frames_to_measure(file, indices=img_num_channels[:, t])
        lbl = locate_labels(pos, population=mode, frames=t)
        if lbl is None:
            continue

        df_props = measure_features(
            img,
            lbl,
            features=features + ["centroid"],
            border_dist=None,
            channels=channel_names,
            haralick_options=haralick_options,
            verbose=False,
        )
        df_props.rename(columns={"centroid-1": "x", "centroid-0": "y"}, inplace=True)
        df_props["t"] = int(t)

        props.append(df_props)

    return props


print(f"Measuring features with {n_threads} thread(s)...")

# Multithreading
indices = list(range(img_num_channels.shape[1]))
chunks = np.array_split(indices, n_threads)

timestep_dataframes = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(measure_index, chunks)
    try:
        for i, return_value in enumerate(results):
            print(f"Thread {i} completed...")
            timestep_dataframes.extend(return_value)
    except Exception as e:
        print("Exception: ", e)

print("Features successfully measured...")

df = pd.concat(timestep_dataframes)
df.reset_index(inplace=True, drop=True)

df = _mask_intensity_measurements(df, mask_channels)

# do tracking
if btrack_option:
    tracker = "bTrack"
else:
    tracker = "trackpy"

print(f"Start the tracking step using the {tracker} tracker...")

trajectories, napari_data = track(
    None,
    configuration=btrack_config,
    objects=df,
    spatial_calibration=spatial_calibration,
    channel_names=channel_names,
    return_napari_data=True,
    optimizer_options={"tm_lim": int(12e4)},
    track_kwargs={"step_size": 100},
    clean_trajectories_kwargs=post_processing_options,
    volume=(shape_x, shape_y),
    btrack_option=btrack_option,
    search_range=search_range,
    memory=memory,
)
print(f"Tracking successfully performed...")

# out trajectory table, create POSITION_X_um, POSITION_Y_um, TIME_min (new ones)
# Save napari data # deprecated, should disappear progressively
np.save(
    pos + os.sep.join(["output", "tables", napari_name]), napari_data, allow_pickle=True
)

trajectories.to_csv(pos + os.sep.join(["output", "tables", table_name]), index=False)
print(
    f"Trajectory table successfully exported in {os.sep.join(['output', 'tables'])}..."
)

if os.path.exists(
    pos + os.sep.join(["output", "tables", table_name.replace(".csv", ".pkl")])
):
    os.remove(
        pos + os.sep.join(["output", "tables", table_name.replace(".csv", ".pkl")])
    )

del trajectories
del napari_data
gc.collect()
