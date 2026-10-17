from multiprocessing import Process, Queue
from typing import Optional, Dict, Any
import logging
import time
import os

logger = logging.getLogger("celldetective")

from celldetective.log_manager import positionlogger

import pandas as pd
from art import tprint

from celldetective.neighborhood import (
    _neigh_col,
    _population_column,
    compute_contact_neighborhood_at_position,
    compute_neighborhood_at_position,
)
from celldetective.relative_measurements import measure_pair_signals_at_position


class NeighborhoodProcess(Process):
    """
    Compute neighborhoods, and optionally measure the pairs, over a batch of
    positions in one process, so that a single progress window follows the
    wells, the positions and the frames.

    ``process_args`` holds ``batch_structure`` (``{well index: {"positions":
    [position folders]}}``), ``protocols`` (the neighborhoods to compute at
    each position, in order), ``img_shape``, and optionally
    ``measure_pairs`` and ``log_file``.
    """

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

        tprint("Neighborhood")

        if not hasattr(self, "measure_pairs"):
            self.measure_pairs = False

    def report_frame(self, done: int, total: int) -> None:
        """Sends the progress of the frames of the position, and an estimate of
        the time left, to the main process."""
        now = time.time()
        if done == 1 and self.frames_done > 0:
            self.sweeps_left = max(self.sweeps_left - 1, 0)
        self.frames_done += 1
        if self.t0 is None:
            # The time per frame is measured from the end of the first frame,
            # so that the loading of the tables, labels and movie, done
            # before it, is not counted as frame time.
            self.t0 = now
            frame_time = "Estimating the time left..."
        else:
            time_per_frame = (now - self.t0) / (self.frames_done - 1)
            frames_left = total - done + self.sweeps_left * total
            frame_time = f"Time left: {round(frames_left * time_per_frame, 1)}s"
        self.queue.put(
            {
                "frame_progress": done / total * 100,
                "frame_time": frame_time,
                "well_progress": self.well_progress,
                "pos_progress": self.pos_progress,
            }
        )

    @staticmethod
    def _time_left(done: int, total: int, elapsed: float, unit: str) -> str:
        """The progress of a loop over wells or positions, with an estimate
        of the time left from the mean time per item so far."""
        if done == 0:
            return f"Processing {unit} 1/{total}..."
        left = (total - done) * elapsed / done
        return f"{unit.capitalize()} {done + 1}/{total} - {int(left // 60)} m {int(left % 60)} s left"

    def run(self):
        """Compute every protocol at every position of the batch, then tell
        the main process that the batch is finished, or why it failed."""
        try:
            self._run_batch()
        except Exception as e:
            logger.exception("The neighborhood computation failed.")
            where = getattr(self, "pos", None)
            message = f"{os.path.basename(os.path.normpath(where))}: {e}" if where else str(e)
            self.queue.put({"status": "error", "message": message})
            self.queue.close()
            return

        # Send end signal
        self.queue.put("finished")
        self.queue.close()

    def _run_batch(self):
        """Compute every protocol at every position of the batch."""
        wells = list(self.batch_structure.values())
        t0_wells = time.time()
        for w, well in enumerate(wells):
            self.well_progress = w / len(wells) * 100
            positions = well["positions"]
            t0_positions = time.time()
            self.queue.put({
                "well_progress": self.well_progress,
                "well_time": self._time_left(w, len(wells), t0_positions - t0_wells, "well"),
            })
            for p, pos in enumerate(positions):
                self.pos = pos
                self.pos_progress = p / len(positions) * 100
                self.queue.put({
                    "pos_progress": self.pos_progress,
                    "pos_time": self._time_left(p, len(positions), time.time() - t0_positions, "position"),
                })
                for protocol in self.protocols:
                    self.protocol = protocol
                    # The frame bar and its estimate follow one protocol;
                    # sweeps_left counts the sweeps over the frames still to
                    # come after the current one (one per distance for the
                    # circle neighborhoods), set in `_run_protocol`.
                    self.frames_done, self.t0, self.sweeps_left = 0, None, 0
                    self._run_protocol()
            self.queue.put({"pos_progress": 100})
        self.queue.put({"well_progress": 100})

    def _run_protocol(self):
        """Compute the neighborhood of ``self.protocol`` at ``self.pos``, and
        measure its pairs if asked."""
        self.queue.put({"status": f"Computing neighborhood: {os.path.basename(os.path.normpath(self.pos))}..."})
        logger.info("Launching the neighborhood computation...")
        options = dict(
            population=self.protocol["population"],
            theta_dist=None,
            img_shape=self.img_shape,
            return_tables=False,
            clear_neigh=self.protocol["clear_neigh"],
            event_time_col=self.protocol["event_time_col"],
            neighborhood_kwargs=self.protocol["neighborhood_kwargs"],
            progress_callback=self.report_frame,
        )
        if self.protocol["neighborhood_type"] == "distance_threshold":
            distances = self.protocol["distance"]
            self.sweeps_left = len(distances) - 1 if isinstance(distances, list) else 0
            compute_neighborhood_at_position(
                self.pos, self.protocol["distance"], **options
            )
            logger.info("Computation done.")
        elif self.protocol["neighborhood_type"] == "mask_contact":
            logger.info("Computing contact neighborhood...")
            compute_contact_neighborhood_at_position(
                self.pos,
                self.protocol["distance"],
                channel_names=self.protocol.get("channel_names"),
                **options,
            )
            logger.info("Computation done.")

        if self.measure_pairs:
            self.queue.put({"status": "Measuring pairs..."})
            logger.info("Measuring pairs...")

            distances = self.protocol["distance"]
            if not isinstance(distances, list):
                distances = [distances]

            population = self.protocol["population"]
            mode = "self" if population[0] == population[1] else "two-pop"
            neigh_type = (
                "circle"
                if self.protocol["neighborhood_type"] == "distance_threshold"
                else "contact"
            )
            for d in distances:
                neigh_col = _population_column(_neigh_col(mode, neigh_type, d), population)

                pair_protocol = {
                    "reference": self.protocol["population"][0],
                    "neighbor": self.protocol["population"][1],
                    "type": neigh_type,
                    "distance": d,
                    "description": neigh_col,
                }
                if self.protocol["neighborhood_type"] == "mask_contact":
                    ch = self.protocol.get("channel_names")
                    if ch:
                        pair_protocol["channel_names"] = ch
                        pair_protocol["contact_border"] = d

                logger.info(f"Processing pairs for {neigh_col}...")
                df_pairs = measure_pair_signals_at_position(self.pos, pair_protocol)

                if df_pairs is not None:
                    if "REFERENCE_ID" in list(df_pairs.columns):
                        previous_pair_table_path = self.pos + os.sep.join(
                            ["output", "tables", "trajectories_pairs.csv"]
                        )

                        if os.path.exists(previous_pair_table_path):
                            df_prev = pd.read_csv(previous_pair_table_path)
                            cols = [
                                c
                                for c in list(df_prev.columns)
                                if c in list(df_pairs.columns)
                            ]
                            df_pairs = pd.merge(df_prev, df_pairs, how="outer", on=cols)

                        try:
                            df_pairs = df_pairs.sort_values(
                                by=[
                                    "reference_population",
                                    "neighbor_population",
                                    "REFERENCE_ID",
                                    "NEIGHBOR_ID",
                                    "FRAME",
                                ]
                            )
                        except KeyError as e:
                            logger.debug(f"Could not sort pair table (missing column): {e}")

                        df_pairs.to_csv(previous_pair_table_path, index=False)
                        logger.info(f"Pair measurements saved to {previous_pair_table_path}")

        with positionlogger(self.pos, filename="log_neighborhood.txt"):
            logger.info("NEIGHBORHOOD")
            logger.info(f"neighborhood_type: {self.protocol.get('neighborhood_type')}")
            logger.info(f"population: {self.protocol.get('population')}")
            logger.info(f"distance: {self.protocol.get('distance')}")
            logger.info(f"clear_neigh: {self.protocol.get('clear_neigh')}")
            logger.info(f"event_time_col: {self.protocol.get('event_time_col')}")
            logger.info(f"neighborhood_kwargs: {self.protocol.get('neighborhood_kwargs')}")
            logger.info(f"measure_pairs: {self.measure_pairs}")

    def end_process(self):
        """End the process."""

        self.terminate()
        self.queue.put("finished")

    def abort_process(self):
        """Abort the process."""

        self.terminate()
        self.queue.put("error")
