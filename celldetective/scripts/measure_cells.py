"""
Copright © 2022 Laboratoire Adhesion et Inflammation, Authored by Remy Torro.
"""

import argparse
import os
import json
from celldetective.utils.image_loaders import (
    locate_labels,
    auto_load_number_of_frames,
    load_frames,
    fix_missing_labels,
    _get_img_num_per_channel,
)
from celldetective.utils.experiment import (
    extract_position_name,
    extract_experiment_channels,
)
from celldetective import (
    extract_experiment_channels,
)
from celldetective.utils.parsing import config_section_to_dict
from celldetective.utils.data_cleaning import (
    _remove_invalid_cols,
    _extract_coordinates_from_features,
    remove_redundant_features,
    remove_trajectory_measurements,
)
from celldetective.measure import (
    drop_tonal_features,
    measure_features,
    measure_isotropic_intensity,
    center_of_mass_to_abs_coordinates,
    measure_radial_distance_to_center,
)
from pathlib import Path, PurePath
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from natsort import natsorted
from art import tprint
import datetime

tprint("Measure")

parser = argparse.ArgumentParser(
    description="Measure features and intensities in a multichannel timeseries.",
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

column_labels = {
    "track": "TRACK_ID",
    "time": "FRAME",
    "x": "POSITION_X",
    "y": "POSITION_Y",
}

if mode.lower() == "target" or mode.lower() == "targets":
    label_folder = "labels_targets"
    table_name = "trajectories_targets.csv"
    instruction_file = os.sep.join(["configs", "measurement_instructions_targets.json"])

elif mode.lower() == "effector" or mode.lower() == "effectors":
    label_folder = "labels_effectors"
    table_name = "trajectories_effectors.csv"
    instruction_file = os.sep.join(
        ["configs", "measurement_instructions_effectors.json"]
    )

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
channel_names, channel_indices = extract_experiment_channels(expfolder)
nbr_channels = len(channel_names)

# from tracking instructions, fetch btrack config, features, haralick, clean_traj, idea: fetch custom timeline?
instr_path = PurePath(expfolder, Path(f"{instruction_file}"))
print("Looking for measurement instruction file...")

if os.path.exists(instr_path):

    with open(instr_path, "r") as f:
        instructions = json.load(f)
        print(f"Measurement instruction file successfully loaded...")
        print(f"Instructions: {instructions}...")

    if "background_correction" in instructions:
        background_correction = instructions["background_correction"]
    else:
        background_correction = None

    if "features" in instructions:
        features = instructions["features"]
    else:
        features = None

    if "border_distances" in instructions:
        border_distances = instructions["border_distances"]
    else:
        border_distances = None

    if "spot_detection" in instructions:
        spot_detection = instructions["spot_detection"]
    else:
        spot_detection = None

    if "haralick_options" in instructions:
        haralick_options = instructions["haralick_options"]
    else:
        haralick_options = None

    if "intensity_measurement_radii" in instructions:
        intensity_measurement_radii = instructions["intensity_measurement_radii"]
    else:
        intensity_measurement_radii = None

    if "isotropic_operations" in instructions:
        isotropic_operations = instructions["isotropic_operations"]
    else:
        isotropic_operations = None

    if "clear_previous" in instructions:
        clear_previous = instructions["clear_previous"]
    else:
        clear_previous = True

else:
    print("No measurement instructions found. Use default measurements.")
    features = ["area", "intensity_mean"]
    border_distances = None
    haralick_options = None
    clear_previous = False
    background_correction = None
    spot_detection = None
    intensity_measurement_radii = 10
    isotropic_operations = ["mean"]

if features is None:
    features = []

# from pos fetch labels
label_path = natsorted(glob(os.sep.join([pos, label_folder, "*.tif"])))
if len(label_path) > 0:
    print(f"Found {len(label_path)} segmented frames...")
else:
    print(
        f"No segmented frames have been found. Please run segmentation first, skipping... Features cannot be computed."
    )
    features = None
    haralick_options = None
    border_distances = None
    label_path = None

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

# Load trajectories, add centroid if not in trajectory
trajectories = pos + os.sep.join(["output", "tables", table_name])
if os.path.exists(trajectories):
    print("A trajectory table was found...")
    trajectories = pd.read_csv(trajectories)
    if "TRACK_ID" not in list(trajectories.columns):
        do_iso_intensities = False
        intensity_measurement_radii = None
        if clear_previous:
            print("No TRACK_ID... Clear previous measurements...")
            trajectories = (
                None  # remove_trajectory_measurements(trajectories, column_labels)
            )
            do_features = True
            features += ["centroid"]
    else:
        if clear_previous:
            print("TRACK_ID found... Clear previous measurements...")
            trajectories = remove_trajectory_measurements(trajectories, column_labels)
else:
    trajectories = None
    do_features = True
    features += ["centroid"]
    do_iso_intensities = False


len_movie_auto = auto_load_number_of_frames(file)
if len_movie_auto is not None:
    len_movie = len_movie_auto

if label_path is not None and file is not None:
    test = len(label_path) == len_movie
    if not test:
        fix_missing_labels(pos, population=mode, prefix=movie_prefix)
        label_path = natsorted(glob(os.sep.join([pos, label_folder, "*.tif"])))

img_num_channels = _get_img_num_per_channel(channel_indices, len_movie, nbr_channels)


# Test what to do
if (file is None) or (intensity_measurement_radii is None):
    do_iso_intensities = False
    print(
        "Either no image, no positions or no radii were provided... Isotropic intensities will not be computed..."
    )
else:
    do_iso_intensities = True

if label_path is None:
    do_features = False
    print("No labels were provided... Features will not be computed...")
else:
    do_features = True


#######################################
# Loop over all frames and find objects
#######################################

timestep_dataframes = []
if trajectories is None:
    print("Use features as a substitute for the trajectory table.")
    if "label" not in features:
        features.append("label")

if label_path is not None:
    label_names = [os.path.split(lbl)[-1] for lbl in label_path]


features_log = f"features: {features}"
border_distances_log = f"border_distances: {border_distances}"
haralick_options_log = f"haralick_options: {haralick_options}"
background_correction_log = f"background_correction: {background_correction}"
spot_detection_log = f"spot_detection: {spot_detection}"
intensity_measurement_radii_log = (
    f"intensity_measurement_radii: {intensity_measurement_radii}"
)
isotropic_options_log = f"isotropic_operations: {isotropic_operations} \n"
log = "\n".join(
    [
        features_log,
        border_distances_log,
        haralick_options_log,
        background_correction_log,
        spot_detection_log,
        intensity_measurement_radii_log,
        isotropic_options_log,
    ]
)
with open(pos + f"log_{mode}.json", "a") as f:
    f.write(f"{datetime.datetime.now()} MEASURE \n")
    f.write(log + "\n")


def measure_index(indices):

    # global column_labels

    for t in tqdm(indices, desc="frame"):

        if file is not None:
            img = load_frames(
                img_num_channels[:, t], file, scale=None, normalize_input=False
            )

        if label_path is not None:

            lbl = locate_labels(pos, population=mode, frames=t)
            if lbl is None:
                continue

        if trajectories is not None:

            positions_at_t = trajectories.loc[
                trajectories[column_labels["time"]] == t
            ].copy()

        if do_features:
            feature_table = measure_features(
                img,
                lbl,
                features=features,
                border_dist=border_distances,
                channels=channel_names,
                haralick_options=haralick_options,
                verbose=False,
                normalisation_list=background_correction,
                spot_detection=spot_detection,
            )
            if trajectories is None:
                positions_at_t = _extract_coordinates_from_features(
                    feature_table, timepoint=t
                )
                column_labels = {
                    "track": "ID",
                    "time": column_labels["time"],
                    "x": column_labels["x"],
                    "y": column_labels["y"],
                }
            feature_table.rename(
                columns={"centroid-1": "POSITION_X", "centroid-0": "POSITION_Y"},
                inplace=True,
            )

        if do_iso_intensities and not trajectories is None:
            iso_table = measure_isotropic_intensity(
                positions_at_t,
                img,
                channels=channel_names,
                intensity_measurement_radii=intensity_measurement_radii,
                column_labels=column_labels,
                operations=isotropic_operations,
                verbose=False,
            )

        if do_iso_intensities and do_features and not trajectories is None:
            measurements_at_t = iso_table.merge(
                feature_table, how="outer", on="class_id", suffixes=("_delme", "")
            )
            measurements_at_t = measurements_at_t[
                [c for c in measurements_at_t.columns if not c.endswith("_delme")]
            ]
        elif do_iso_intensities * (not do_features) * (not trajectories is None):
            measurements_at_t = iso_table
        elif do_features:
            measurements_at_t = positions_at_t.merge(
                feature_table, how="outer", on="class_id", suffixes=("_delme", "")
            )
            measurements_at_t = measurements_at_t[
                [c for c in measurements_at_t.columns if not c.endswith("_delme")]
            ]

        measurements_at_t = center_of_mass_to_abs_coordinates(measurements_at_t)
        measurements_at_t = measure_radial_distance_to_center(
            measurements_at_t, volume=img.shape, column_labels=column_labels
        )

        if measurements_at_t is not None:
            measurements_at_t[column_labels["time"]] = t
            timestep_dataframes.append(measurements_at_t)

    return


print(f"Starting the measurements with {n_threads} thread(s)...")

import concurrent.futures

# Multithreading
indices = list(range(img_num_channels.shape[1]))
chunks = np.array_split(indices, n_threads)

with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(measure_index, chunks)
    try:
        for i, return_value in enumerate(results):
            print(f"Thread {i} output check: ", return_value)
    except Exception as e:
        print("Exception: ", e)

print("Done.")


if len(timestep_dataframes) > 0:

    df = pd.concat(timestep_dataframes)

    if trajectories is not None:
        df = df.sort_values(by=[column_labels["track"], column_labels["time"]])
        df = df.dropna(subset=[column_labels["track"]])
    else:
        df["ID"] = np.arange(len(df))
        df = df.sort_values(by=[column_labels["time"], "ID"])

    df = df.reset_index(drop=True)
    df = _remove_invalid_cols(df)

    df.to_csv(pos + os.sep.join(["output", "tables", table_name]), index=False)
    print(
        f'Measurement table successfully exported in  {os.sep.join(["output", "tables"])}...'
    )
    print("Done.")
else:
    print("No measurement could be performed. Check your inputs.")
    print("Done.")
