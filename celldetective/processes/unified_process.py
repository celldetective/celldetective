import time
import os
import gc
import importlib
import threading
from multiprocessing import Process, Queue
from typing import List, Optional, Dict, Any
from pathlib import Path

from celldetective.log_manager import (
    get_logger,
    setup_global_logging,
    forward_logs_to_queue,
    capture_library_logs,
)

logger = get_logger(__name__)


def _warm_up_imports(module_names: List[str]) -> None:
    """
    Import modules ahead of time, ignoring failures.

    Only ``sys.modules`` is populated: the caller still performs its own import
    later on and will surface any error then. Running this in a background
    thread lets the (largely GIL-free) import of the deep learning backends
    absorb the import cost of the stages that run after the first one.

    Parameters
    ----------
    module_names : list of str
        Dotted module names to import.
    """

    for name in module_names:
        try:
            importlib.import_module(name)
        except Exception as e:
            # The blocking import on the main thread will raise for real.
            logger.debug(f"Warm-up import of {name} failed ({e}); ignoring.")


class UnifiedBatchProcess(Process):
    """
    A unified process that handles Segmentation, Tracking, Measurement, and Signal Analysis
    in a sequential manner for each position, updating a multi-bar progress window.
    """

    def __init__(
        self, queue: Queue, process_args: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the unified batch process.

        Parameters
        ----------
        queue : multiprocessing.Queue
            Queue for progress updates.
        process_args : dict, optional
            Arguments for the process.
        """
        super(UnifiedBatchProcess, self).__init__()
        self.queue = queue
        if process_args is None:
            process_args = {}
        self.process_args = process_args
        self.batch_structure = process_args.get("batch_structure", {})

        self.run_segmentation = process_args.get("run_segmentation", False)
        self.run_tracking = process_args.get("run_tracking", False)
        self.run_measurement = process_args.get("run_measurement", False)
        self.run_signals = process_args.get("run_signals", False)

        self.seg_args = process_args.get("seg_args", {})
        self.track_args = process_args.get("track_args", {})
        self.measure_args = process_args.get("measure_args", {})
        self.signal_args = process_args.get("signal_args", {})
        self.log_file = process_args.get("log_file", None)

    def run(self):
        """
        Run the unified batch process.

        SEGMENT/TRACK/MEASURE execute in this spawned child, whose stdout the GUI does not
        capture. ``forward_logs_to_queue`` ships this process's logs — and those of the
        libraries it calls (cellpose, stardist, btrack, trackpy) — back to the parent so
        they surface where the user is watching.
        """

        if self.log_file is not None:
            setup_global_logging(log_file=self.log_file)

        with forward_logs_to_queue(self.queue):
            self._run_batch()

    def _run_batch(self):
        """Run the segmentation/tracking/measurement/signal batch for every position."""

        logger.info("Starting Unified Batch Process...")

        # The stages are initialized in order, but only the first one blocks the
        # start of the run. Import the modules of the later stages on a
        # background thread so that their (multi-second) import cost overlaps
        # with the model loading of the first stage instead of following it.
        deferred_modules = []
        stage_modules = [
            (self.run_segmentation, ["celldetective.processes.segment_cells"]),
            (self.run_tracking, ["celldetective.processes.track_cells"]),
            (self.run_measurement, ["celldetective.processes.measure_cells"]),
            (
                self.run_signals,
                [
                    "celldetective.utils.event_detection",
                    "celldetective.processes.detect_events",
                ],
            ),
        ]
        for enabled, modules in stage_modules:
            if enabled:
                deferred_modules.extend(modules)
        # The first enabled stage is imported by the main thread right away.
        deferred_modules = deferred_modules[1:]

        warm_up_thread = None
        if deferred_modules:
            warm_up_thread = threading.Thread(
                target=_warm_up_imports,
                args=(deferred_modules,),
                name="celldetective-warmup",
                daemon=True,
            )
            warm_up_thread.start()

        # Initialize Workers
        # Propagate batch structure to sub-processes so they can locate experiment config
        for args_dict in [
            self.seg_args,
            self.track_args,
            self.measure_args,
            self.signal_args,
        ]:
            if isinstance(args_dict, dict):
                args_dict["batch_structure"] = self.batch_structure

        seg_worker = None
        model = None
        scale_model = None

        if self.run_segmentation:
            logger.info("Initializing the segmentation worker...")
            self.queue.put({"status": "Initializing segmentation..."})

            if "threshold_instructions" in self.seg_args:
                from celldetective.processes.segment_cells import (
                    SegmentCellThresholdProcess,
                )

                seg_worker = SegmentCellThresholdProcess(
                    queue=self.queue, process_args=self.seg_args
                )
            else:
                from celldetective.processes.segment_cells import SegmentCellDLProcess

                seg_worker = SegmentCellDLProcess(
                    queue=self.queue, process_args=self.seg_args
                )

                if seg_worker.model_type == "stardist":
                    logger.info("Loading the StarDist library...")
                    from celldetective.utils.stardist_utils import _prep_stardist_model

                    model, scale_model = _prep_stardist_model(
                        seg_worker.model_name,
                        Path(seg_worker.model_complete_path).parent,
                        use_gpu=seg_worker.use_gpu,
                        scale=seg_worker.scale,
                    )
                elif seg_worker.model_type == "cellpose":
                    logger.info("Loading the cellpose_utils library...")
                    from celldetective.utils.cellpose_utils import _prep_cellpose_model

                    model, scale_model = _prep_cellpose_model(
                        seg_worker.model_name,
                        seg_worker.model_complete_path,
                        use_gpu=seg_worker.use_gpu,
                        n_channels=len(seg_worker.required_channels),
                        scale=seg_worker.scale,
                    )

        track_worker = None
        if self.run_tracking:
            from celldetective.processes.track_cells import TrackingProcess

            logger.info("Initializing the tracking worker...")
            self.queue.put({"status": "Initializing tracking..."})
            track_worker = TrackingProcess(
                queue=self.queue, process_args=self.track_args
            )

        measure_worker = None
        if self.run_measurement:
            logger.info("Loading the measurement libraries...")
            from celldetective.processes.measure_cells import MeasurementProcess

            logger.info("Initializing the measurement worker...")
            self.queue.put({"status": "Initializing measurements..."})
            measure_worker = MeasurementProcess(
                queue=self.queue, process_args=self.measure_args
            )

        signal_worker = None
        signal_model = None

        if self.run_signals:
            try:
                from celldetective.utils.event_detection import (
                    _prep_event_detection_model,
                )

                logger.info("Loading the event detection model...")
                self.queue.put({"status": "Loading event detection model..."})
                model_name = self.signal_args["model_name"]
                signal_model = _prep_event_detection_model(
                    model_name, use_gpu=self.signal_args.get("gpu", True)
                )
            except (ImportError, ModuleNotFoundError) as e:
                if "tensorflow" in str(e):
                    logger.warning(
                        "Tensorflow not found/installed. Signal analysis will be skipped."
                    )
                    self.run_signals = False
                else:
                    logger.error(
                        f"Failed to initialize event detection model: {e}",
                        exc_info=True,
                    )
                    self.run_signals = False
            except Exception as e:
                logger.error(
                    f"Failed to initialize event detection model: {e}", exc_info=True
                )
                self.run_signals = False  # Disable signal analysis if model fails

        if self.run_signals:
            from celldetective.processes.detect_events import SignalAnalysisProcess

            logger.info("Initializing the event detection worker...")
            signal_worker = SignalAnalysisProcess(
                queue=self.queue, process_args=self.signal_args
            )
            signal_worker.signal_model_instance = signal_model

        self.t0_well = time.time()

        for w_i, (w_idx, well_data) in enumerate(self.batch_structure.items()):

            positions = well_data["positions"]

            # Well Progress Update
            elapsed = time.time() - self.t0_well
            if w_i > 0:
                avg = elapsed / w_i
                rem = (len(self.batch_structure) - w_i) * avg
                mins = int(rem // 60)
                secs = int(rem % 60)
                well_str = f"Well {w_i + 1}/{len(self.batch_structure)} - {mins} m {secs} s left"
            else:
                well_str = f"Processing well {w_i + 1}/{len(self.batch_structure)}..."

            self.queue.put(
                {
                    "well_progress": (w_i / len(self.batch_structure)) * 100,
                    "well_time": well_str,
                }
            )

            self.t0_pos = time.time()

            for pos_idx, pos_path in enumerate(positions):

                # Position Progress Update
                elapsed_pos = time.time() - self.t0_pos
                if pos_idx > 0:
                    avg_p = elapsed_pos / pos_idx
                    rem_p = (len(positions) - pos_idx) * avg_p
                    mins_p = int(rem_p // 60)
                    secs_p = int(rem_p % 60)
                    pos_str = f"Pos {pos_idx + 1}/{len(positions)} - {mins_p} m {secs_p} s left"
                else:
                    pos_str = f"Processing position {pos_idx + 1}/{len(positions)}..."

                self.queue.put(
                    {
                        "pos_progress": (pos_idx / len(positions)) * 100,
                        "pos_time": pos_str,
                    }
                )

                # Calculate active steps for this run
                active_steps = []
                if self.run_segmentation:
                    active_steps.append("Segmentation")
                if self.run_tracking:
                    active_steps.append("Tracking")
                if self.run_measurement:
                    active_steps.append("Measurement")
                if self.run_signals:
                    active_steps.append("Event detection")

                total_steps = len(active_steps)
                current_step = 0

                try:
                    # --- SEGMENTATION ---
                    if self.run_segmentation and seg_worker:
                        current_step += 1
                        step_info = f"[Step {current_step}/{total_steps}]"
                        msg = f"{step_info} Segmenting {os.path.basename(pos_path)}..."
                        logger.info(msg)
                        self.queue.put({"status": msg})

                        seg_worker.setup_for_position(pos_path)

                        with capture_library_logs(
                            seg_worker.pos, f"log_{seg_worker.mode}.txt"
                        ):
                            if not "threshold_instructions" in self.seg_args:
                                seg_worker.process_position(
                                    model=model, scale_model=scale_model
                                )
                            else:
                                seg_worker.process_position()

                    # --- TRACKING ---
                    if self.run_tracking and track_worker:
                        current_step += 1
                        step_info = f"[Step {current_step}/{total_steps}]"
                        msg = f"{step_info} Tracking {os.path.basename(pos_path)}..."
                        logger.info(msg)
                        self.queue.put({"status": msg})

                        track_worker.setup_for_position(pos_path)
                        with capture_library_logs(
                            track_worker.pos, f"log_{track_worker.mode}.txt"
                        ):
                            track_worker.process_position()

                    # --- MEASUREMENT ---
                    if self.run_measurement and measure_worker:
                        current_step += 1
                        step_info = f"[Step {current_step}/{total_steps}]"
                        msg = f"{step_info} Measuring {os.path.basename(pos_path)}..."
                        logger.info(msg)
                        self.queue.put({"status": msg})

                        measure_worker.setup_for_position(pos_path)
                        with capture_library_logs(
                            measure_worker.pos, f"log_{measure_worker.mode}.txt"
                        ):
                            measure_worker.process_position()

                    # --- SIGNAL ANALYSIS ---
                    if self.run_signals and signal_worker:
                        current_step += 1
                        step_info = f"[Step {current_step}/{total_steps}]"
                        msg = f"{step_info} Detecting events in position {os.path.basename(pos_path)}..."
                        logger.info(msg)
                        self.queue.put({"status": msg})

                        signal_worker.setup_for_position(pos_path)
                        signal_worker.process_position(model=signal_model)

                except Exception as e:
                    logger.error(f"Error processing position {pos_path}: {e}")
                    self.queue.put(
                        {
                            "status": f"Error at {os.path.basename(pos_path)}. Skipping..."
                        }
                    )
                    logger.error(
                        f"Skipping position {os.path.basename(pos_path)} due to error: {e}",
                        exc_info=True,
                    )
                    continue

                gc.collect()

                # Update Position Progress (Complete)
                self.queue.put({"pos_progress": ((pos_idx + 1) / len(positions)) * 100})

            # Update Well Progress (Complete)
            self.queue.put(
                {"well_progress": ((w_i + 1) / len(self.batch_structure)) * 100}
            )

        self.queue.put("finished")
        self.queue.close()

    def end_process(self):
        """
        Terminate the process.
        """
        try:
            if self.is_alive():
                self.terminate()
                self.join()
        except Exception as e:
            logger.error(f"Error terminating process: {e}")
