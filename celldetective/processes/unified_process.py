from multiprocessing import Process
import time
import os
import gc
from celldetective.log_manager import get_logger

logger = get_logger(__name__)


class UnifiedBatchProcess(Process):

    def __init__(self, queue=None, process_args=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = queue
        if process_args is not None:
            for key, value in process_args.items():
                setattr(self, key, value)
        logger.info("Process initialized...")

    def end_process(self):
        self.terminate()
        self.queue.put("finished")

    def abort_process(self):
        self.terminate()
        self.queue.put("error")

    def run(self):
        logger.info("Starting Process...")
        try:
            self._run_unsafe()
        except Exception:
            logger.error(f"Critical error in child process", exc_info=True)
        finally:
            logger.info("Exiting Process...")

    def _run_unsafe(self):
        self.queue.put({"status": "Process started. Loading libraries..."})

        if hasattr(self, "log_file") and self.log_file:
            from celldetective.log_manager import setup_global_logging

            setup_global_logging(log_file=self.log_file)

        from celldetective.log_manager import get_logger

        logger = get_logger(__name__)

        # Lazy imports to speed up process start on Windows (spawn)
        from celldetective.processes.segment_cells import (
            SegmentCellDLProcess,
            SegmentCellThresholdProcess,
        )
        from celldetective.utils import (
            _prep_stardist_model,
            _prep_cellpose_model,
            _prep_event_detection_model,
        )
        from celldetective.processes.track_cells import TrackingProcess
        from celldetective.processes.measure_cells import MeasurementProcess
        from celldetective.processes.detect_events import SignalAnalysisProcess

        seg_worker = None
        model = None
        scale_model = None

        if self.run_segmentation:

            logger.info("Initializing the segmentation worker...")
            self.queue.put({"status": "Initializing segmentation models..."})
            self.seg_args["batch_structure"] = self.batch_structure

            if self.seg_args.get("threshold_instructions", None) is not None:
                seg_worker = SegmentCellThresholdProcess(
                    queue=self.queue, process_args=self.seg_args
                )
            else:
                seg_worker = SegmentCellDLProcess(
                    queue=self.queue, process_args=self.seg_args
                )

                if seg_worker.model_type == "stardist":
                    model, scale_model = _prep_stardist_model(
                        seg_worker.model_name,
                        Path(seg_worker.model_complete_path).parent,
                        use_gpu=seg_worker.use_gpu,
                        scale=seg_worker.scale,
                    )
                elif seg_worker.model_type == "cellpose":
                    model, scale_model = _prep_cellpose_model(
                        seg_worker.model_name,
                        seg_worker.model_complete_path,
                        use_gpu=seg_worker.use_gpu,
                        n_channels=len(seg_worker.required_channels),
                        scale=seg_worker.scale,
                    )

        track_worker = None
        if self.run_tracking:
            logger.info("Initializing the tracking worker...")
            self.queue.put({"status": "Initializing tracking..."})
            track_worker = TrackingProcess(
                queue=self.queue, process_args=self.track_args
            )

        measure_worker = None
        if self.run_measurement:
            logger.info("Initializing the measurement worker...")
            self.queue.put({"status": "Initializing measurements..."})
            measure_worker = MeasurementProcess(
                queue=self.queue, process_args=self.measure_args
            )

        signal_worker = None
        signal_model = None

        if self.run_signals:

            try:
                logger.info("Loading the event detection model...")
                self.queue.put({"status": "Loading event detection model..."})
                model_name = self.signal_args["model_name"]
                signal_model = _prep_event_detection_model(
                    model_name, use_gpu=self.signal_args.get("gpu", True)
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize event detection model: {e}", exc_info=True
                )
                self.run_signals = False  # Disable signal analysis if model fails

        if self.run_signals:

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
                        if isinstance(seg_worker, SegmentCellDLProcess):
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
                        track_worker.process_position()

                    # --- MEASUREMENT ---
                    if self.run_measurement and measure_worker:
                        current_step += 1
                        step_info = f"[Step {current_step}/{total_steps}]"
                        msg = f"{step_info} Measuring {os.path.basename(pos_path)}..."
                        logger.info(msg)
                        self.queue.put({"status": msg})

                        measure_worker.setup_for_position(pos_path)
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
