"""Pair event detection with a model, in its own process as for single cells."""

from multiprocessing import Process, Queue
from typing import Any, Dict, List, Optional

from celldetective.log_manager import get_logger, positionlogger

logger = get_logger(__name__)


class PairSignalAnalysisProcess(Process):
    """Detect pair events at one position, reporting to a `ProgressWindow`."""

    pos: Optional[str] = None
    model_name: Optional[str] = None
    use_gpu: bool = True
    populations: List[str] = ["targets", "effectors"]
    well_progress: float = 0
    pos_progress: float = 0

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
            ``pos``, ``model_name``, and optionally ``use_gpu``, ``populations``,
            ``well_progress`` and ``pos_progress``.
        """

        super().__init__()
        self.queue = queue
        if process_args is not None:
            for key, value in process_args.items():
                setattr(self, key, value)

    def run(self) -> None:
        """Detect the pair events of the position, then report the end."""

        from celldetective.signals import analyze_pair_signals_at_position

        self.queue.put(
            {
                "status": "Detecting pair events...",
                "well_progress": self.well_progress,
                "pos_progress": self.pos_progress,
            }
        )
        try:
            table = analyze_pair_signals_at_position(
                self.pos,
                self.model_name,
                use_gpu=self.use_gpu,
                populations=self.populations,
            )
            if table is None:
                # A table it needs is missing: skip the position, as before, and
                # let a batch go on with the next one.
                logger.warning(f"Pair event detection skipped for {self.pos}: a table is missing.")
                self.queue.put({"status": "Skipped: a table is missing"})
            else:
                with positionlogger(self.pos, filename="log_pairs.txt"):
                    logger.info("PAIR SIGNAL ANALYSIS")
                    logger.info(f"signal model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error in PairSignalAnalysisProcess: {e}", exc_info=True)
            self.queue.put({"status": "error", "message": str(e)})
            return
        self.queue.put({"status": "finished"})

    def end_process(self) -> None:
        """End the process."""

        self.terminate()
        self.queue.put("finished")

    def abort_process(self) -> None:
        """Abort the process."""

        self.terminate()
        self.queue.put("error")
