from multiprocessing import Process
import time
import os
from pathlib import PurePath, Path
import numpy as np
from tifffile import imwrite
from celldetective.preprocessing import (
    correct_background_model,
    correct_background_model_free,
    auto_load_number_of_frames,
    _get_img_num_per_channel,
    _extract_channel_indices_from_config,
)
from celldetective.utils import config_section_to_dict, extract_experiment_channels


class BackgroundCorrectionProcess(Process):

    def __init__(self, queue=None, process_args=None):

        super().__init__()

        self.queue = queue

        if process_args is not None:
            for key, value in process_args.items():
                setattr(self, key, value)

        self.sum_done = 0
        self.t0 = time.time()

    def run(self):

        print("Start background correction process...")

        try:
            # Load config to get movie length for progress estimation
            self.config = PurePath(self.exp_dir, Path("config.ini"))
            self.len_movie = float(
                config_section_to_dict(self.config, "MovieSettings")["len_movie"]
            )
            self.nbr_channels = len(extract_experiment_channels(self.exp_dir)[0])
            channel_indices = _extract_channel_indices_from_config(
                self.config, [self.target_channel]
            )
            self.img_num_channels = _get_img_num_per_channel(
                channel_indices, self.len_movie, self.nbr_channels
            )

            # Safer calculation of total frames
            n_pos = (
                len(self.position_option)
                if hasattr(self.position_option, "__len__")
                else 1
            )
            n_wells = (
                len(self.well_option) if hasattr(self.well_option, "__len__") else 1
            )
            self.total_frames = self.len_movie * n_pos * n_wells
            print(f"Process initialized. Total frames: {self.total_frames}")

        except Exception as e:
            print(f"Error initializing process: {e}")
            self.queue.put("error")
            return

        self.count = 0
        self.t0 = time.time()

        export = getattr(self, "export", False)
        return_stacks = getattr(self, "return_stacks", True)
        movie_prefix = getattr(self, "movie_prefix", None)
        export_prefix = getattr(self, "export_prefix", "Corrected")
        correction_type = getattr(self, "correction_type", "model")

        # Timestamps for estimation
        self.t0_well = time.time()
        self.t0_pos = time.time()  # resets per well
        self.count_pos = 0  # pos processed in current well

        def progress_callback(**kwargs):

            level = kwargs.get("level", None)
            iteration = kwargs.get("iter", 0)
            total = kwargs.get("total", 1)
            stage = kwargs.get("stage", "")

            current_time = time.time()
            data = {}

            if level == "well":
                if iteration == 0:
                    self.t0_well = current_time

                well_progress = ((iteration) / total) * 100
                if well_progress > 100:
                    well_progress = 100
                data["well_progress"] = well_progress

                elapsed = current_time - self.t0_well
                if iteration > 0:
                    avg = elapsed / iteration
                    rem = total - iteration
                    rem_t = rem * avg
                    mins = int(rem_t // 60)
                    secs = int(rem_t % 60)
                    data["well_time"] = f"Estimated: {mins} m {secs} s"
                else:
                    data["well_time"] = "Estimating..."

                # Reset pos timer for new well
                self.t0_pos = current_time
                self.count_pos = 0

            elif level == "position":
                # Overall progress based on positions inside well
                # iteration is pidx within well (0-indexed)
                self.count_pos = iteration

                # Reset timer if stage changes or if it's the first item
                current_stage = getattr(self, "current_stage", None)
                reset_timer = False
                if stage != current_stage:
                    self.current_stage = stage
                    reset_timer = True
                if iteration == 0:
                    reset_timer = True

                if reset_timer:
                    self.t0_pos = current_time

                pos_progress = ((iteration + 1) / total) * 100
                if pos_progress > 100:
                    pos_progress = 100
                data["pos_progress"] = pos_progress

                # We calculate average speed based on items completed SINCE reset
                # If we reset at iteration 0 (end of item 0), then at iteration 1 (end of item 1):
                # elapsed = time(item 1). count = 1.
                # So we use 'iteration' as the count of items measured by 'elapsed' (if reset at 0).

                elapsed = current_time - self.t0_pos

                measured_count = iteration
                if measured_count > 0:
                    avg = elapsed / measured_count
                    rem = total - (iteration + 1)
                    rem_t = rem * avg
                    mins = int(rem_t // 60)
                    secs = int(rem_t % 60)
                    data["pos_time"] = f"{stage}: {mins} m {secs} s"
                else:
                    data["pos_time"] = f"{stage}..."

            elif level == "frame":
                # Sub-progress for frames
                if iteration == 0:
                    self.t0_frame = current_time

                frame_progress = ((iteration + 1) / total) * 100
                if frame_progress > 100:
                    frame_progress = 100
                data["frame_progress"] = frame_progress

                elapsed = current_time - getattr(self, "t0_frame", current_time)
                # iteration 0: t0 set.
                # iteration 1: elapsed = frame 1 dur. measured_count = 1.
                measured_count = iteration

                if measured_count > 0:
                    avg = elapsed / measured_count
                    rem = total - (iteration + 1)
                    rem_t = rem * avg
                    mins = int(rem_t // 60)
                    secs = int(rem_t % 60)
                    data["frame_time"] = f"{mins} m {secs} s"
                else:
                    data["frame_time"] = f"{iteration + 1}/{total} frames"

            elif level is None:
                # Fallback for old style logic - maintain compatibility just in case
                self.count += 1
                if self.count > self.total_frames:
                    self.count = self.total_frames

                self.sum_done = (self.count / self.total_frames) * 100

                # We map this to position progress as main
                data["pos_progress"] = self.sum_done
                if self.count > 0:
                    elapsed = current_time - self.t0
                    avg = elapsed / self.count
                    rem = (self.total_frames - self.count) * avg
                    mins = int(rem // 60)
                    secs = int(rem % 60)
                    data["pos_time"] = f"{mins} m {secs} s"

            if data:
                self.queue.put(data)

        try:
            if correction_type == "model-free":
                corrected_stacks = correct_background_model_free(
                    self.exp_dir,
                    well_option=self.well_option,
                    position_option=self.position_option,
                    target_channel=self.target_channel,
                    mode=getattr(self, "mode", "timeseries"),
                    threshold_on_std=self.threshold_on_std,
                    frame_range=getattr(self, "frame_range", [0, 5]),
                    optimize_option=getattr(self, "optimize_option", False),
                    opt_coef_range=getattr(self, "opt_coef_range", [0.95, 1.05]),
                    opt_coef_nbr=getattr(self, "opt_coef_nbr", 100),
                    operation=self.operation,
                    clip=self.clip,
                    offset=getattr(self, "offset", None),
                    export=export,
                    return_stacks=return_stacks,
                    fix_nan=getattr(self, "fix_nan", False),
                    activation_protocol=self.activation_protocol,
                    show_progress_per_well=False,
                    show_progress_per_pos=False,
                    movie_prefix=movie_prefix,
                    export_prefix=export_prefix,
                    progress_callback=progress_callback,
                )
            else:
                corrected_stacks = correct_background_model(
                    self.exp_dir,
                    well_option=self.well_option,
                    position_option=self.position_option,
                    target_channel=self.target_channel,
                    model=self.model,
                    threshold_on_std=self.threshold_on_std,
                    operation=self.operation,
                    clip=self.clip,
                    export=export,
                    return_stacks=return_stacks,
                    activation_protocol=self.activation_protocol,
                    show_progress_per_well=False,
                    show_progress_per_pos=False,
                    movie_prefix=movie_prefix,
                    export_prefix=export_prefix,
                    progress_callback=progress_callback,
                    downsample=getattr(self, "downsample", 10),
                )

            if return_stacks and corrected_stacks and len(corrected_stacks) > 0:
                # Save to temp file for the main process to pick up
                # We assume single position for preview
                temp_path = os.path.join(self.exp_dir, "temp_corrected_stack.tif")
                try:
                    imwrite(temp_path, corrected_stacks[0])
                    print(f"Saved temp stack to {temp_path}")
                except Exception as temp_e:
                    print(f"Failed to save temp stack: {temp_e}")

            self.queue.put(
                {
                    "well_progress": 100,
                    "pos_progress": 100,
                    "frame_progress": 100,
                    "status": "finished",
                }
            )

        except Exception as e:
            print(f"Error in background correction process: {e}")
            self.queue.put("error")
            return

        self.queue.put("finished")
        self.queue.close()

    def end_process(self):
        self.terminate()
        self.queue.put("finished")
