from multiprocessing import Process, Queue
from typing import Optional, Dict, Any
import time
import os
from tifffile import imwrite
from celldetective import get_logger

logger = get_logger(__name__)


class _CorrectionStep:
    """
    One correction of a list, with its own options: those of a step cannot reach the next.

    Parameters
    ----------
    queue : Queue
        The queue to communicate with the main process.
    options : dict
        Options of the correction.
    step_index : int, optional
        Position of the correction in the list. Default is 0.
    step_count : int, optional
        Length of the list. Default is 1.
    """

    def __init__(
        self,
        queue: Queue,
        options: Dict[str, Any],
        step_index: int = 0,
        step_count: int = 1,
    ) -> None:
        self.queue = queue
        for key, value in options.items():
            setattr(self, key, value)
        self.step_index = step_index
        self.step_count = step_count

    def _label(self, text: str) -> str:
        """Prefix a message with the step, in a list of corrections."""
        if self.step_count == 1:
            return text
        return f"[Step {self.step_index + 1}/{self.step_count}] {text}"

    def _overall_progress(self, fraction: float) -> float:
        """Percentage of the whole list, given the fraction of this step that is done."""
        return (self.step_index + min(fraction, 1)) / self.step_count * 100

    def run(self) -> bool:
        """
        Run the correction.

        Returns
        -------
        bool
            False if it failed, in which case the progress window was told.
        """
        logger.info("Start background correction process...")

        if self.step_count > 1:
            self.queue.put(
                {
                    "status": self._label(getattr(self, "step_title", "Correcting...")),
                    "well_progress": self._overall_progress(0),
                    # Those of the previous step are full.
                    "pos_progress": 0,
                    "frame_progress": 0,
                }
            )

        export = getattr(self, "export", False)
        return_stacks = getattr(self, "return_stacks", True)
        movie_prefix = getattr(self, "movie_prefix", None)
        export_prefix = getattr(self, "export_prefix", "Corrected")
        correction_type = getattr(self, "correction_type", "model")

        # Timestamps for estimation
        self.t0_well = time.time()
        self.t0_pos = time.time()  # resets per well

        def progress_callback(**kwargs):
            """
            Callback for reporting progress.

            Parameters
            ----------
            **kwargs
                Arbitrary keyword arguments.
            """

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
                data["status"] = self._label(status)

            if image_preview is not None:
                data["image_preview"] = image_preview

            if level == "well":
                if iteration == 0:
                    self.t0_well = current_time

                data["well_progress"] = self._overall_progress(iteration / total)

                elapsed = current_time - self.t0_well
                if iteration > 0:
                    avg = elapsed / iteration
                    rem = total - iteration
                    rem_t = rem * avg
                    mins = int(rem_t // 60)
                    secs = int(rem_t % 60)
                    data["well_time"] = self._label(f"Estimated: {mins} m {secs} s")
                else:
                    data["well_time"] = self._label("Estimating...")

                # Reset pos timer for new well
                self.t0_pos = current_time

            elif level == "position":

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
            common = dict(
                well_option=self.well_option,
                position_option=self.position_option,
                target_channel=self.target_channel,
                export=export,
                return_stacks=return_stacks,
                show_progress_per_well=False,
                show_progress_per_pos=False,
                movie_prefix=movie_prefix,
                export_prefix=export_prefix,
                source_stacks=getattr(self, "source_stacks", None),
                progress_callback=progress_callback,
            )
            if correction_type == "model-free":
                from celldetective.preprocessing import correct_background_model_free

                corrected_stacks = correct_background_model_free(
                    self.exp_dir,
                    mode=getattr(self, "mode", "timeseries"),
                    threshold_on_std=self.threshold_on_std,
                    frame_range=getattr(self, "frame_range", [0, 5]),
                    optimize_option=getattr(self, "optimize_option", False),
                    opt_coef_range=getattr(self, "opt_coef_range", [0.95, 1.05]),
                    opt_radius=getattr(self, "opt_radius", None),
                    operation=self.operation,
                    clip=self.clip,
                    offset=getattr(self, "offset", None),
                    fix_nan=getattr(self, "fix_nan", False),
                    activation_protocol=self.activation_protocol,
                    **common,
                    subset_indices=getattr(self, "subset_indices", None),
                    preview_frames=getattr(self, "preview_frames", None),
                )
            elif correction_type == "offset":
                from celldetective.preprocessing import correct_channel_offset

                corrected_stacks = correct_channel_offset(
                    self.exp_dir,
                    **common,
                    correction_horizontal=getattr(self, "correction_horizontal", 0),
                    correction_vertical=getattr(self, "correction_vertical", 0),
                    **self.kwargs if hasattr(self, "kwargs") else {},
                )
            elif correction_type == "registration":
                from celldetective.preprocessing import register_stacks

                corrected_stacks = register_stacks(
                    self.exp_dir,
                    **common,
                    # register_stacks owns the defaults of the options left out.
                    **{
                        key: getattr(self, key)
                        for key in (
                            "radius",
                            "tukey_alpha",
                            "upsample_factor",
                            "reference",
                            "downscale",
                            "highpass_sigma",
                        )
                        if hasattr(self, key)
                    },
                )
            else:
                from celldetective.preprocessing import correct_background_model

                corrected_stacks = correct_background_model(
                    self.exp_dir,
                    model=self.model,
                    threshold_on_std=self.threshold_on_std,
                    operation=self.operation,
                    clip=self.clip,
                    activation_protocol=self.activation_protocol,
                    **common,
                    downsample=getattr(self, "downsample", 10),
                    subset_indices=getattr(self, "subset_indices", None),
                    offset=getattr(self, "offset", None),
                )

            if return_stacks and corrected_stacks and len(corrected_stacks) > 0:
                # Handed to the preview through a file: the progress window reads no result.
                temp_path = os.path.join(self.exp_dir, "temp_corrected_stack.tif")
                try:
                    imwrite(temp_path, corrected_stacks[0])
                    logger.info(f"Saved temp stack to {temp_path}")
                except Exception as temp_e:
                    logger.error(f"Failed to save temp stack: {temp_e}")

            self.queue.put({"well_progress": self._overall_progress(1)})

        except Exception as e:
            logger.error(f"Error in background correction process: {e}")
            self.queue.put({"status": "error", "message": self._label(str(e))})
            return False

        return True


class BackgroundCorrectionProcess(Process):
    """
    Run a list of corrections one after the other, in one progress window.

    Parameters
    ----------
    queue : Queue
        The queue to communicate with the main process.
    process_args : dict
        The options of a correction, or under ``steps`` the list of those of each correction.
    """

    def __init__(
        self,
        queue: Optional[Queue] = None,
        process_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.queue = queue
        process_args = process_args or {}
        self.steps = process_args.get("steps") or [process_args]

    def run(self):
        """Run the corrections. The window closes on the last message: only once all are done."""
        for k, options in enumerate(self.steps):
            if not _CorrectionStep(self.queue, options, k, len(self.steps)).run():
                return

        self.queue.put(
            {
                "well_progress": 100,
                "pos_progress": 100,
                "frame_progress": 100,
                "status": "finished",
            }
        )
        self.queue.put("finished")
        self.queue.close()

    def end_process(self):
        """End the process."""
        self.terminate()
        self.queue.put("finished")
