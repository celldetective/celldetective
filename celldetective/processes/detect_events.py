from multiprocessing import Process
import os
import numpy as np
import pandas as pd

from celldetective.log_manager import get_logger
from celldetective.tracking import clean_trajectories
from celldetective.utils.color_mappings import (
    color_from_status,
    color_from_class,
)
from celldetective.utils.event_detection import _prep_event_detection_model

logger = get_logger(__name__)


class SignalAnalysisProcess(Process):

    pos = None
    mode = None
    model_name = None
    use_gpu = True

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

    def setup_for_position(self, pos):
        self.pos = pos
        self.pos_path = rf"{pos}"

    def process_position(self, model=None):
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

            # --- Logic adapted from analyze_signals to include progress ---

            # Configuration checks (similar to analyze_signals)
            if model is None:
                # This path handles if model instance wasn't passed (fallback, though unified_process should pass it)
                if hasattr(self, "signal_model_instance"):
                    model = self.signal_model_instance
                else:
                    # Lazy load if needed
                    model = _prep_event_detection_model(
                        self.model_name, use_gpu=self.use_gpu
                    )

            config = model.config
            required_signals = config["channels"]
            model_signal_length = config["model_signal_length"]

            # Channel selection logic
            available_signals = list(trajectories.columns)
            selected_signals = config.get("selected_channels", None)

            if selected_signals is None:
                selected_signals = []
                for s in required_signals:
                    priority_cols = [a for a in available_signals if a == s]
                    second_priority_cols = [
                        a for a in available_signals if a.startswith(s) and a != s
                    ]
                    third_priority_cols = [
                        a for a in available_signals if s in a and not a.startswith(s)
                    ]
                    candidates = (
                        priority_cols + second_priority_cols + third_priority_cols
                    )

                    if len(candidates) > 0:
                        selected_signals.append(candidates[0])
                    else:
                        logger.error(f"No match for signal {s} in {available_signals}")
                        raise ValueError(f"Missing required channel: {s}")

            # Preprocessing
            trajectories_clean = clean_trajectories(
                trajectories,
                interpolate_na=True,
                interpolate_position_gaps=True,
                column_labels=self.column_labels,
            )

            max_signal_size = (
                int(trajectories_clean[self.column_labels["time"]].max()) + 2
            )
            if max_signal_size > model_signal_length:
                logger.warning(
                    f"Signals longer than model input ({max_signal_size} > {model_signal_length}). Truncating may occur."
                )

            tracks = trajectories_clean[self.column_labels["track"]].unique()
            signals = np.zeros((len(tracks), max_signal_size, len(selected_signals)))

            # Progress loop for signal extraction
            total_tracks = len(tracks)

            for i, (tid, group) in enumerate(
                trajectories_clean.groupby(self.column_labels["track"])
            ):

                # Report progress
                progress = ((i + 1) / total_tracks) * 100
                self.queue.put(
                    {
                        "frame_progress": progress,  # Reusing frame_progress key for UI compatibility
                        "frame_time": f"Extracting signals: {i+1}/{total_tracks}",
                    }
                )

                frames = group[self.column_labels["time"]].to_numpy().astype(int)
                for j, col in enumerate(selected_signals):
                    signal = group[col].to_numpy()
                    signals[i, frames, j] = signal
                    signals[i, max(frames) :, j] = signal[-1]

            # Prediction
            self.queue.put({"frame_time": "Predicting events..."})
            classes = model.predict_class(signals)
            times_recast = model.predict_time_of_interest(signals)

            # Assign results
            try:
                label = config.get("label", "")
                if label == "":
                    label = None
            except:
                label = None

            if label is None:
                class_col = "class"
                time_col = "t0"
                status_col = "status"
            else:
                class_col = "class_" + label
                time_col = "t_" + label
                status_col = "status_" + label

            self.queue.put({"frame_time": "Saving results..."})

            # Vectorized assignment is faster than loop, but let's stick to safe logic
            # We need to map track_id to result index. 'tracks' array indices align with 'signals' indices
            track_to_idx = {t: i for i, t in enumerate(tracks)}

            # Map predictions to original dataframe
            # Using map is much faster than iterating if possible, but let's do safe iteration for now or efficient mapping
            # Actually, let's use the track ID map
            trajectories[class_col] = trajectories[self.column_labels["track"]].map(
                lambda x: classes[track_to_idx[x]] if x in track_to_idx else 0
            )
            trajectories[time_col] = trajectories[self.column_labels["track"]].map(
                lambda x: times_recast[track_to_idx[x]] if x in track_to_idx else 0
            )

            # Generate Status/Color columns
            # This is complex to vectorize due to time dependency (t >= t0).
            # We can iterate group-wise again or use vectorized pandas ops

            # For status generation, we stick to the loop as in original code, but maybe optimize?
            # Original code iterates groupby. Let's do that for safety and correctness.

            for tid, group in trajectories.groupby(self.column_labels["track"]):
                indices = group.index
                t0 = group[time_col].iloc[0]
                cclass = group[class_col].iloc[0]
                timeline = group[self.column_labels["time"]].to_numpy()
                status = np.zeros_like(timeline)

                if t0 > 0:
                    status[timeline >= t0] = 1.0
                if cclass == 2:
                    status[:] = 2
                if cclass > 2:
                    status[:] = 42

                # Color mapping is slow if done element-wise.
                # But color_from_status returns list/string.
                # Let's just assign status first.
                trajectories.loc[indices, status_col] = status

            # Status colors
            # Optimization: define color map and map values
            # status_color = [color_from_status(s) for s in status]
            # applying function on column is faster
            trajectories["status_color"] = trajectories[status_col].apply(
                color_from_status
            )
            trajectories["class_color"] = trajectories[class_col].apply(
                color_from_class
            )

            trajectories = trajectories.sort_values(
                by=[self.column_labels["track"], self.column_labels["time"]]
            )
            trajectories.to_csv(trajectories_path, index=False)

            logger.info(f"Signal analysis completed for {self.pos}")

        except Exception as e:
            logger.error(f"Error in SignalAnalysisProcess: {e}", exc_info=True)
            raise e

    def run(self):
        # This run method is for independent execution, but UnifiedBatchProcess calls methods directly.
        # However, keeping it robust.
        self.setup_for_position(self.pos)
        model = _prep_event_detection_model(
            self.model_name, use_gpu=self.use_gpu
        )  # Load local if running standalone
        self.process_position(model)
        self.queue.put("finished")
