from multiprocessing import Process
import time
import os
from pathlib import PurePath, Path
from tifffile import imwrite
from celldetective import get_logger
from celldetective.utils.experiment import extract_experiment_channels
from celldetective.utils.image_loaders import _get_img_num_per_channel
from celldetective.utils.parsing import config_section_to_dict, _extract_channel_indices_from_config

logger = get_logger(__name__)


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

        logger.info("Start background correction process...")

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

            logger.info("Process initialized.")

        except Exception as e:
            logger.error(f"Error initializing process: {e}")
            self.queue.put("error")
            return

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
            status = kwargs.get("status", None)
            image_preview = kwargs.get("image_preview", None)
            # Legacy support
            if image_preview is None:
                image_preview = kwargs.get("bg_image", None)

            data = {}

            if status:
                data["status"] = status

            if image_preview is not None:
                data["image_preview"] = image_preview

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

                self.count_pos = iteration

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
                if iteration == 0:
                    self.t0_frame = current_time

                frame_progress = ((iteration + 1) / total) * 100
                if frame_progress > 100:
                    frame_progress = 100
                data["frame_progress"] = frame_progress

                elapsed = current_time - getattr(self, "t0_frame", current_time)
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

            if data:
                self.queue.put(data)

        try:
            if correction_type == "model-free":
                from celldetective.preprocessing import correct_background_model_free

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
            elif correction_type == "offset":
                from celldetective.preprocessing import correct_channel_offset

                corrected_stacks = correct_channel_offset(
                    self.exp_dir,
                    well_option=self.well_option,
                    position_option=self.position_option,
                    target_channel=self.target_channel,
                    export=export,
                    return_stacks=return_stacks,
                    show_progress_per_well=False,
                    show_progress_per_pos=False,
                    movie_prefix=movie_prefix,
                    export_prefix=export_prefix,
                    progress_callback=progress_callback,
                    correction_horizontal=getattr(self, "correction_horizontal", 0),
                    correction_vertical=getattr(self, "correction_vertical", 0),
                    **self.kwargs if hasattr(self, "kwargs") else {},
                )
            else:
                from celldetective.preprocessing import correct_background_model

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
                    subset_indices=getattr(self, "subset_indices", None),
                )

            if return_stacks and corrected_stacks and len(corrected_stacks) > 0:
                # If doing a preview (subset_indices is set), return via queue instead of disk
                if getattr(self, "subset_indices", None) is not None:
                    self.queue.put({"status": "result", "data": corrected_stacks[0]})
                else:
                    temp_path = os.path.join(self.exp_dir, "temp_corrected_stack.tif")
                    try:
                        imwrite(temp_path, corrected_stacks[0])
                        logger.info(f"Saved temp stack to {temp_path}")
                    except Exception as temp_e:
                        logger.error(f"Failed to save temp stack: {temp_e}")

            self.queue.put(
                {
                    "well_progress": 100,
                    "pos_progress": 100,
                    "frame_progress": 100,
                    "status": "finished",
                }
            )

        except Exception as e:
            logger.error(f"Error in background correction process: {e}")
            self.queue.put({"status": "error", "message": str(e)})
            return

        self.queue.put("finished")
        self.queue.close()

    def end_process(self):
        self.terminate()
        self.queue.put("finished")
