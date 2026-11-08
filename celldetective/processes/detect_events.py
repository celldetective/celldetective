from multiprocessing import Process, Queue
from typing import Optional, Dict, Any
import os
import pandas as pd

from celldetective.log_manager import get_logger, positionlogger
from celldetective.utils.event_detection import _prep_event_detection_model
from celldetective.utils import COLUMN_LABELS

logger = get_logger(__name__)


class SignalAnalysisProcess(Process):

    pos = None
    mode = None
    model_name = None
    use_gpu = True

    def __init__(
        self,
        queue: Optional[Queue] = None,
        process_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the process.

        Parameters
        ----------
        queue : Queue
            The queue to communicate with the main process.
        process_args : dict
            Arguments for the process.
        """
        super().__init__()
        self.queue = queue
        if process_args is not None:
            for key, value in process_args.items():
                setattr(self, key, value)

        self.column_labels = COLUMN_LABELS.copy()

    def setup_for_position(self, pos: str) -> None:
        """
        Setup the process for a specific position.

        Parameters
        ----------
        pos : str
            The position path.
        """
        self.pos = pos
        self.pos_path = rf"{pos}"

    def process_position(self, model: Optional[Any] = None) -> None:
        """
        Process the position for signal analysis.

        Parameters
        ----------
        model : object
            The signal detection model.
        """
        logger.info(
            f"Analyzing signals for position {self.pos} with model {self.model_name}"
        )

        try:
            # Determine table name based on mode
            if self.mode.lower() in ["target", "targets"]:
                table_name = "trajectories_targets.csv"
            elif self.mode.lower() in ["effector", "effectors"]:
                table_name = "trajectories_effectors.csv"
            else:
                table_name = f"trajectories_{self.mode}.csv"

            trajectories_path = os.path.join(self.pos, "output", "tables", table_name)

            if not os.path.exists(trajectories_path):
                logger.warning(f"No trajectories table found at {trajectories_path}")
                return

            trajectories = pd.read_csv(trajectories_path)

            if self.column_labels["track"] not in trajectories.columns:
                logger.warning(
                    f"Column {self.column_labels['track']} not found in {trajectories_path}. Skipping position."
                )
                return

            # Imported here: celldetective.signals loads matplotlib and scipy.
            from celldetective.signals import (
                detect_events_with_model,
                experiment_frame_interval,
            )

            if model is None:
                # Fallback, though unified_process passes the model
                if hasattr(self, "signal_model_instance"):
                    model = self.signal_model_instance
                else:
                    model = _prep_event_detection_model(
                        self.model_name, use_gpu=self.use_gpu
                    )

            def report(i: int, n: int) -> None:
                self.queue.put(
                    {
                        "frame_progress": i / n * 100,  # frame_progress key for UI compatibility
                        "frame_time": f"Extracting signals: {i}/{n}",
                    }
                )
                if i == n:
                    self.queue.put({"frame_time": "Predicting events..."})

            trajectories, selected_signals, (class_col, time_col, status_col) = (
                detect_events_with_model(
                    trajectories,
                    model,
                    column_labels=self.column_labels,
                    interpolate_na=True,
                    progress=report,
                    frame_to_min=experiment_frame_interval(self.pos),
                )
            )

            self.queue.put({"frame_time": "Saving results..."})
            trajectories = trajectories.sort_values(
                by=[self.column_labels["track"], self.column_labels["time"]]
            )
            trajectories.to_csv(trajectories_path, index=False)

            with positionlogger(self.pos, filename=f"log_{self.mode}.txt"):
                logger.info("SIGNAL ANALYSIS")
                logger.info(f"signal model: {self.model_name}")
                logger.info(f"selected_signals: {selected_signals}")
                logger.info(f"columns_written: {[class_col, time_col, status_col]}")

            logger.info(f"Signal analysis completed for {self.pos}")

        except Exception as e:
            logger.error(f"Error in SignalAnalysisProcess: {e}", exc_info=True)
            raise

    def run(self):
        """Run the signal analysis process."""
        # This run method is for independent execution, but UnifiedBatchProcess calls methods directly.
        # However, keeping it robust.
        self.setup_for_position(self.pos)
        model = _prep_event_detection_model(
            self.model_name, use_gpu=self.use_gpu
        )  # Load local if running standalone
        self.process_position(model)
        self.queue.put("finished")
