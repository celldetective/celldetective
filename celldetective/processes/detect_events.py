from multiprocessing import Process, Queue
from typing import Optional, Dict, Any
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
from celldetective.utils import COLUMN_LABELS
from celldetective.utils.dataset_helpers import resolve_signal_channels
from celldetective.utils.event_schema import event_column_names, status_from_event
from celldetective.utils.schema import trajectory_table_name, trajectory_table_path

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
        # Threshold/query classification path: no DL model involved.
        if getattr(self, "threshold_config", None) is not None:
            self._process_position_threshold()
            return

        logger.info(
            f"Analyzing signals for position {self.pos} with model {self.model_name}"
        )

        try:
            trajectories_path = trajectory_table_path(self.pos, self.mode)

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

            # Channel selection logic (shared resolver — identical to training)
            available_signals = list(trajectories.columns)
            selected_signals = config.get("selected_channels", None)

            if selected_signals is None:
                selected_signals = resolve_signal_channels(
                    required_signals, available_signals
                )
                if selected_signals is None:
                    logger.error(
                        f"No match for required signals {required_signals} in {available_signals}"
                    )
                    raise ValueError(f"Missing required channel(s): {required_signals}")

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
                logger.error(
                    f"Signals are longer than the model input ({max_signal_size} > "
                    f"{model_signal_length}); this model cannot process this position."
                )
                raise ValueError(
                    f"Signals longer ({max_signal_size}) than model input "
                    f"({model_signal_length}) for position {self.pos}."
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

            # Prediction. When a wider normalization scope (well/experiment) was
            # requested, a pooled range was pre-computed for this worker and is
            # applied here instead of re-fitting per position.
            self.queue.put({"frame_time": "Predicting events..."})
            norm_override = getattr(self, "normalization_stats", None)
            classes = model.predict_class(
                signals, normalization_values_override=norm_override
            )
            times_recast = model.predict_time_of_interest(
                signals, normalization_values_override=norm_override
            )

            # Assign results
            try:
                label = config.get("label", "")
                if label == "":
                    label = None
            except (KeyError, AttributeError):
                label = None

            class_col, time_col, status_col = event_column_names(label)

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
                status = status_from_event(timeline, cclass, t0)
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
            raise

    def _process_position_threshold(self) -> None:
        """Apply a saved threshold/query classification config to this position."""
        from celldetective.signals import classify_position_from_config

        name = self.threshold_config.get("name", "")
        self.queue.put(
            {"frame_time": f"Applying threshold classification '{name}'..."}
        )
        try:
            classify_position_from_config(
                self.pos, self.threshold_config, mode=self.mode
            )
            logger.info(f"Threshold classification completed for {self.pos}")
        except FileNotFoundError as e:
            logger.warning(str(e))
        except Exception as e:
            logger.error(
                f"Threshold classification failed for {self.pos}: {e}", exc_info=True
            )
            raise

    def _table_name(self) -> str:
        """Return the trajectories table filename for the current mode."""
        return trajectory_table_name(self.mode)

    def compute_well_normalization_stats(self, positions, model):
        """Pool the per-channel normalization range over every position of a well.

        Used when the model's ``normalization_scope`` is ``"well"`` or
        ``"experiment"``: the percentile range is fitted once over all the
        positions in scope and then applied unchanged to each position, instead
        of being re-fitted from a single (possibly small or atypical) position.

        Parameters
        ----------
        positions : list of str
            Position paths to pool over.
        model : SignalDetectionModel
            The loaded model, used for its config (channels, normalization).

        Returns
        -------
        list of [float, float] or None
            One ``[min, max]`` per channel, or None if normalization is disabled
            or no signals could be gathered (caller falls back to per-position).
        """
        from celldetective.event_detection_models import (
            compute_normalization_stats,
            pad_to_model_length,
        )

        config = model.config
        if not config.get("normalize", True):
            return None

        required_signals = config["channels"]
        model_signal_length = config["model_signal_length"]
        table_name = self._table_name()

        # Only pool when every position in scope already has a measurement table.
        # During an interleaved full run the sibling tables are produced
        # just-in-time, so the pool would be incomplete; fall back to
        # per-position normalization in that case.
        table_paths = [
            os.path.join(pos, "output", "tables", table_name) for pos in positions
        ]
        if not table_paths or not all(os.path.exists(p) for p in table_paths):
            logger.info(
                "Pooled normalization skipped (not all position tables are present "
                "yet); falling back to per-position normalization."
            )
            return None

        per_pos_signals = []
        for pos in positions:
            trajectories_path = os.path.join(pos, "output", "tables", table_name)
            if not os.path.exists(trajectories_path):
                continue
            trajectories = pd.read_csv(trajectories_path)
            if self.column_labels["track"] not in trajectories.columns:
                continue

            available_signals = list(trajectories.columns)
            selected_signals = config.get("selected_channels", None)
            if selected_signals is None:
                selected_signals = resolve_signal_channels(
                    required_signals, available_signals
                )
            if selected_signals is None:
                continue

            trajectories_clean = clean_trajectories(
                trajectories,
                interpolate_na=True,
                interpolate_position_gaps=True,
                column_labels=self.column_labels,
            )
            max_signal_size = (
                int(trajectories_clean[self.column_labels["time"]].max()) + 2
            )
            # Cap so we can pad to a common length; out-of-range frames are dropped
            # (they would error at prediction time anyway).
            max_signal_size = min(max_signal_size, model_signal_length)
            tracks = trajectories_clean[self.column_labels["track"]].unique()
            signals = np.zeros(
                (len(tracks), max_signal_size, len(selected_signals))
            )
            for i, (tid, group) in enumerate(
                trajectories_clean.groupby(self.column_labels["track"])
            ):
                frames = group[self.column_labels["time"]].to_numpy().astype(int)
                keep = frames < max_signal_size
                frames_k = frames[keep]
                if len(frames_k) == 0:
                    continue
                for j, col in enumerate(selected_signals):
                    signal = group[col].to_numpy()[keep]
                    signals[i, frames_k, j] = signal
                    signals[i, max(frames_k):, j] = signal[-1]

            per_pos_signals.append(pad_to_model_length(signals, model_signal_length))

        if not per_pos_signals:
            return None

        pooled = np.concatenate(per_pos_signals, axis=0)
        return compute_normalization_stats(
            pooled,
            required_signals,
            normalization_percentile=config.get("normalization_percentile"),
            normalization_values=config.get("normalization_values"),
            normalization_clip=config.get("normalization_clip"),
        )

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
