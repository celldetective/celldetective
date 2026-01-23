from multiprocessing import Process
import time
import datetime
import os
import json
from pathlib import Path, PurePath

from celldetective.utils.image_loaders import (
    auto_load_number_of_frames,
    load_frames,
    _get_img_num_per_channel,
)
from celldetective.utils.experiment import extract_experiment_channels
from celldetective.utils.parsing import config_section_to_dict
from celldetective.utils.data_cleaning import (
    _extract_coordinates_from_features,
    remove_trajectory_measurements,
)
from glob import glob
from tqdm import tqdm
import numpy as np
import concurrent.futures
from natsort import natsorted
from art import tprint
from typing import Optional, Union
import gc
from celldetective.measure import (
    measure_features,
    measure_isotropic_intensity,
    center_of_mass_to_abs_coordinates,
    measure_radial_distance_to_center,
    drop_tonal_features,
)
import pandas as pd
from celldetective.utils.image_loaders import locate_labels

from celldetective.log_manager import get_logger

logger = get_logger(__name__)


class MeasurementProcess(Process):

    pos: Optional[Union[str, Path]] = None
    mode: Optional[str] = None
    n_threads: int = 1

    def __init__(self, queue=None, process_args=None):

        super().__init__()

        self.queue = queue

        if process_args is not None:
            for key, value in process_args.items():
                setattr(self, key, value)

        self.column_labels = {
            "track": "TRACK_ID",
            "time": "FRAME",
            "x": "POSITION_X",
            "y": "POSITION_Y",
        }

        self.sum_done = 0
        self.t0 = time.time()

    def check_possible_measurements(self):

        if (self.file is None) or (self.intensity_measurement_radii is None):
            self.do_iso_intensities = False
            logger.warning(
                "Either no image, no positions or no radii were provided... Isotropic intensities will not be computed..."
            )
        else:
            self.do_iso_intensities = True

        if self.label_path is None:
            self.do_features = False
            logger.warning(
                "No labels were provided... Features will not be computed..."
            )
        else:
            self.do_features = True

        if self.trajectories is None:
            logger.info("Use features as a substitute for the trajectory table.")
            if "label" not in self.features:
                self.features.append("label")

    def read_measurement_instructions(self):

        logger.info("Looking for measurement instruction file...")
        instr_path = PurePath(self.exp_dir, Path(f"{self.instruction_file}"))
        if os.path.exists(instr_path):
            with open(instr_path, "r") as f:
                self.instructions = json.load(f)
                logger.info(f"Measurement instruction file successfully loaded...")
                logger.info(f"Instructions: {self.instructions}...")

            if "background_correction" in self.instructions:
                self.background_correction = self.instructions["background_correction"]
            else:
                self.background_correction = None

            if "features" in self.instructions:
                self.features = self.instructions["features"]
            else:
                self.features = None

            if "border_distances" in self.instructions:
                self.border_distances = self.instructions["border_distances"]
            else:
                self.border_distances = None

            if "spot_detection" in self.instructions:
                self.spot_detection = self.instructions["spot_detection"]
            else:
                self.spot_detection = None

            if "haralick_options" in self.instructions:
                self.haralick_options = self.instructions["haralick_options"]
            else:
                self.haralick_options = None

            if "intensity_measurement_radii" in self.instructions:
                self.intensity_measurement_radii = self.instructions[
                    "intensity_measurement_radii"
                ]
            else:
                self.intensity_measurement_radii = None

            if "isotropic_operations" in self.instructions:
                self.isotropic_operations = self.instructions["isotropic_operations"]
            else:
                self.isotropic_operations = None

            if "clear_previous" in self.instructions:
                self.clear_previous = self.instructions["clear_previous"]
            else:
                self.clear_previous = True

        else:
            logger.info("No measurement instructions found. Use default measurements.")
            self.features = ["area", "intensity_mean"]
            self.border_distances = None
            self.haralick_options = None
            self.clear_previous = False
            self.background_correction = None
            self.spot_detection = None
            self.intensity_measurement_radii = 10
            self.isotropic_operations = ["mean"]

        if self.features is None:
            self.features = []

    def detect_channels(self):
        self.img_num_channels = _get_img_num_per_channel(
            self.channel_indices, self.len_movie, self.nbr_channels
        )

    def write_log(self):

        features_log = f"features: {self.features}"
        border_distances_log = f"border_distances: {self.border_distances}"
        haralick_options_log = f"haralick_options: {self.haralick_options}"
        background_correction_log = (
            f"background_correction: {self.background_correction}"
        )
        spot_detection_log = f"spot_detection: {self.spot_detection}"
        intensity_measurement_radii_log = (
            f"intensity_measurement_radii: {self.intensity_measurement_radii}"
        )
        isotropic_options_log = f"isotropic_operations: {self.isotropic_operations} \n"
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
        with open(self.pos + f"log_{self.mode}.txt", "a") as f:
            f.write(f"{datetime.datetime.now()} MEASURE \n")
            f.write(log + "\n")

    def prepare_folders(self):

        if self.mode.lower() == "target" or self.mode.lower() == "targets":
            self.label_folder = "labels_targets"
            self.table_name = "trajectories_targets.csv"
            self.instruction_file = os.sep.join(
                ["configs", "measurement_instructions_targets.json"]
            )

        elif self.mode.lower() == "effector" or self.mode.lower() == "effectors":
            self.label_folder = "labels_effectors"
            self.table_name = "trajectories_effectors.csv"
            self.instruction_file = os.sep.join(
                ["configs", "measurement_instructions_effectors.json"]
            )

        else:
            self.label_folder = f"labels_{self.mode}"
            self.table_name = f"trajectories_{self.mode}.csv"
            self.instruction_file = os.sep.join(
                ["configs", f"measurement_instructions_{self.mode}.json"]
            )

    def extract_experiment_parameters(self):

        self.movie_prefix = config_section_to_dict(self.config, "MovieSettings")[
            "movie_prefix"
        ]
        self.spatial_calibration = float(
            config_section_to_dict(self.config, "MovieSettings")["pxtoum"]
        )
        self.time_calibration = float(
            config_section_to_dict(self.config, "MovieSettings")["frametomin"]
        )
        self.len_movie = float(
            config_section_to_dict(self.config, "MovieSettings")["len_movie"]
        )
        self.shape_x = int(
            config_section_to_dict(self.config, "MovieSettings")["shape_x"]
        )
        self.shape_y = int(
            config_section_to_dict(self.config, "MovieSettings")["shape_y"]
        )

        self.channel_names, self.channel_indices = extract_experiment_channels(
            self.exp_dir
        )
        self.nbr_channels = len(self.channel_names)

    def locate_experiment_config(self):

        parent1 = Path(self.pos).parent
        self.exp_dir = parent1.parent
        self.config = PurePath(self.exp_dir, Path("config.ini"))

        if not os.path.exists(self.config):
            logger.error("The configuration file for the experiment was not found...")
            self.abort_process()

    def detect_tracks(self):

        # Load trajectories, add centroid if not in trajectory
        self.trajectories = self.pos + os.sep.join(
            ["output", "tables", self.table_name]
        )
        if os.path.exists(self.trajectories):
            logger.info("Previous table detected...")
            self.trajectories = pd.read_csv(self.trajectories)
            if "TRACK_ID" not in list(self.trajectories.columns):
                logger.info("Static measurements detected...")
                self.do_iso_intensities = False
                self.intensity_measurement_radii = None
                if self.clear_previous:
                    logger.info("Clear previous measurements...")
                    self.trajectories = None  # remove_trajectory_measurements(trajectories, column_labels)
                    self.do_features = True
                    self.features += ["centroid"]
                self.column_labels.update({"track": "ID"})
            else:
                logger.info("Time series detected...")
                if self.clear_previous:
                    logger.info("TRACK_ID found... Clear previous measurements...")
                    self.trajectories = remove_trajectory_measurements(
                        self.trajectories, self.column_labels
                    )
        else:
            self.trajectories = None
            self.do_features = True
            self.features += ["centroid"]
            self.do_iso_intensities = False

    def detect_movie_and_labels(self):

        self.label_path = natsorted(
            glob(os.sep.join([self.pos, self.label_folder, "*.tif"]))
        )
        if len(self.label_path) > 0:
            logger.info(f"Found {len(self.label_path)} segmented frames...")
        else:
            self.features = None
            self.haralick_options = None
            self.border_distances = None
            self.label_path = None

        try:
            self.file = glob(
                self.pos + os.sep.join(["movie", f"{self.movie_prefix}*.tif"])
            )[0]
        except IndexError:
            self.file = None
            self.haralick_option = None
            self.features = drop_tonal_features(self.features)

        len_movie_auto = auto_load_number_of_frames(self.file)
        if len_movie_auto is not None:
            self.len_movie = len_movie_auto

    def parallel_job(self, indices):

        measurements = []

        for t in tqdm(indices, desc="frame"):

            measurements_at_t = None
            perform_measurement = True

            if self.file is not None:
                img = load_frames(
                    self.img_num_channels[:, t],
                    self.file,
                    scale=None,
                    normalize_input=False,
                )

            if self.label_path is not None:
                lbl = locate_labels(self.pos, population=self.mode, frames=t)
                if lbl is None:
                    perform_measurement = False

            if perform_measurement:

                if self.trajectories is not None:
                    # Optimized access
                    if self.frame_slices is not None:
                        # Check if frame t is in our precomputed slices
                        if t in self.frame_slices:
                            start, end = self.frame_slices[t]
                            positions_at_t = self.trajectories.iloc[start:end].copy()
                        else:
                            # Empty frame for trajectories
                            positions_at_t = pd.DataFrame(
                                columns=self.trajectories.columns
                            )
                    else:
                        # Fallback or original method (should not be reached if optimized)
                        positions_at_t = self.trajectories.loc[
                            self.trajectories[self.column_labels["time"]] == t
                        ].copy()

                if self.do_features:
                    feature_table = measure_features(
                        img,
                        lbl,
                        features=self.features,
                        border_dist=self.border_distances,
                        channels=self.channel_names,
                        haralick_options=self.haralick_options,
                        verbose=False,
                        normalisation_list=self.background_correction,
                        spot_detection=self.spot_detection,
                    )
                    if self.trajectories is None:
                        positions_at_t = _extract_coordinates_from_features(
                            feature_table, timepoint=t
                        )
                        column_labels = {
                            "track": "ID",
                            "time": self.column_labels["time"],
                            "x": self.column_labels["x"],
                            "y": self.column_labels["y"],
                        }
                    feature_table.rename(
                        columns={
                            "centroid-1": "POSITION_X",
                            "centroid-0": "POSITION_Y",
                        },
                        inplace=True,
                    )

                if self.do_iso_intensities and not self.trajectories is None:
                    iso_table = measure_isotropic_intensity(
                        positions_at_t,
                        img,
                        channels=self.channel_names,
                        intensity_measurement_radii=self.intensity_measurement_radii,
                        column_labels=self.column_labels,
                        operations=self.isotropic_operations,
                        verbose=False,
                    )

                if (
                    self.do_iso_intensities
                    and self.do_features
                    and not self.trajectories is None
                ):
                    measurements_at_t = iso_table.merge(
                        feature_table,
                        how="outer",
                        on="class_id",
                        suffixes=("_delme", ""),
                    )
                    measurements_at_t = measurements_at_t[
                        [
                            c
                            for c in measurements_at_t.columns
                            if not c.endswith("_delme")
                        ]
                    ]
                elif (
                    self.do_iso_intensities
                    * (not self.do_features)
                    * (not self.trajectories is None)
                ):
                    measurements_at_t = iso_table
                elif self.do_features:
                    measurements_at_t = positions_at_t.merge(
                        feature_table,
                        how="outer",
                        on="class_id",
                        suffixes=("_delme", ""),
                    )
                    measurements_at_t = measurements_at_t[
                        [
                            c
                            for c in measurements_at_t.columns
                            if not c.endswith("_delme")
                        ]
                    ]

                measurements_at_t = center_of_mass_to_abs_coordinates(measurements_at_t)

                measurements_at_t = measure_radial_distance_to_center(
                    measurements_at_t,
                    volume=img.shape,
                    column_labels=self.column_labels,
                )

            self.sum_done += 1
            data = {}

            # Frame Progress
            frame_progress = (self.sum_done / self.len_movie) * 100
            if frame_progress > 100:
                frame_progress = 100
            data["frame_progress"] = frame_progress

            # Frame Time Estimation
            elapsed = time.time() - getattr(self, "t0_frame", time.time())
            if self.sum_done > 0:
                avg = elapsed / self.sum_done
                rem = self.len_movie - self.sum_done
                rem_t = rem * avg
                mins = int(rem_t // 60)
                secs = int(rem_t % 60)
                data["frame_time"] = f"Measurement: {mins} m {secs} s"
            else:
                data["frame_time"] = "Measuring..."

            self.queue.put(data)

            if measurements_at_t is not None:
                measurements_at_t[self.column_labels["time"]] = t
            else:
                measurements_at_t = pd.DataFrame()

            measurements.append(measurements_at_t)

        return measurements

    def setup_for_position(self, pos):

        self.pos = pos
        # Experiment
        self.prepare_folders()
        self.locate_experiment_config()
        self.extract_experiment_parameters()
        self.read_measurement_instructions()
        self.detect_movie_and_labels()
        self.detect_tracks()
        self.detect_channels()
        self.check_possible_measurements()
        self.write_log()

    def process_position(self):
        tprint("Measure")

        self.indices = list(range(self.img_num_channels.shape[1]))
        chunks = np.array_split(self.indices, self.n_threads)

        self.timestep_dataframes = []
        self.t0_frame = time.time()
        self.sum_done = 0

        # Optimize: Group trajectories by frame for O(1) access inside the loop
        self.frame_slices = None
        if self.trajectories is not None:
            # Sort by FRAME to enable searchsorted
            self.trajectories = self.trajectories.sort_values(
                self.column_labels["time"]
            )
            frames = self.trajectories[self.column_labels["time"]].values

            # Find unique frames and their indices
            unique_frames = np.unique(frames)

            # searchsorted returns the indices where elements should be inserted to maintain order
            # 'left' gives the start index, 'right' gives the end index
            start_indices = np.searchsorted(frames, unique_frames, side="left")
            end_indices = np.searchsorted(frames, unique_frames, side="right")

            self.frame_slices = {
                frame: (start, end)
                for frame, start, end in zip(unique_frames, start_indices, end_indices)
            }

        if self.n_threads > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.n_threads
            ) as executor:
                results = executor.map(
                    self.parallel_job, chunks
                )  # list(map(lambda x: executor.submit(self.parallel_job, x), chunks))
                try:
                    for i, return_value in enumerate(results):
                        logger.info(f"Thread {i} completed...")
                        self.timestep_dataframes.extend(return_value)
                except Exception as e:
                    logger.error("Exception: ", e)
                    raise e
        else:
            try:
                # Avoid thread pool overhead for single thread
                results = [self.parallel_job(chunks[0])]
                for i, return_value in enumerate(results):
                    logger.info(f"Job {i} completed...")
                    self.timestep_dataframes.extend(return_value)
            except Exception as e:
                logger.error("Exception: ", e)
                raise e

        logger.info("Measurements successfully performed...")

        if len(self.timestep_dataframes) > 0:

            df = pd.concat(self.timestep_dataframes)

            if self.trajectories is not None:
                df = df.sort_values(
                    by=[self.column_labels["track"], self.column_labels["time"]]
                )
                df = df.dropna(subset=[self.column_labels["track"]])
            else:
                df["ID"] = np.arange(len(df))
                df = df.sort_values(by=[self.column_labels["time"], "ID"])

            df = df.reset_index(drop=True)
            # df = _remove_invalid_cols(df)
            logger.info(f"Final columns before export: {df.columns.tolist()}")
            df = df.replace([np.inf, -np.inf], np.nan)

            df.to_csv(
                self.pos + os.sep.join(["output", "tables", self.table_name]),
                index=False,
            )
            logger.info(
                f'Measurement table successfully exported in  {os.sep.join(["output", "tables"])}...'
            )
            logger.info("Done.")
        else:
            logger.error("No measurement could be performed. Check your inputs.")

        logger.info("Done.")
        gc.collect()

    def run(self):

        self.setup_for_position(self.pos)
        self.process_position()

        # Send end signal
        self.queue.put("finished")
        self.queue.close()

    def end_process(self):

        self.terminate()
        self.queue.put("finished")

    def abort_process(self):

        self.terminate()
        self.queue.put("error")
