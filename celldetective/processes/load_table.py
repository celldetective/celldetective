import time
from multiprocessing import Process, Queue
from typing import Optional, Dict, Any
from celldetective.utils.data_loaders import load_experiment_tables
from celldetective import get_logger

logger = get_logger()


class TableLoaderProcess(Process):

    def __init__(
        self,
        queue: Optional[Queue] = None,
        process_args: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the process.

        Parameters
        ----------
        queue : Queue
            The queue to communicate with the main process.
        process_args : dict
            Arguments for the process.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """

        super().__init__(*args, **kwargs)

        if process_args is not None:
            for key, value in process_args.items():
                setattr(self, key, value)

        self.queue = queue

    def run(self):
        """Run the table loading process."""

        def progress(well_progress: float, pos_progress: float) -> bool:
            """
            Report progress.

            Parameters
            ----------
            well_progress : float
                Progress for the well.
            pos_progress : float
                Progress for the position.
            """
            # Check for cancellation if needed?
            # The runner checks queue for instructions? No, runner closes queue.
            # But here we can just push updates.
            self.queue.put(
                {
                    "well_progress": well_progress,
                    "pos_progress": pos_progress,
                    "status": f"Loading tables... Well {well_progress}%, Position {pos_progress}%",
                }
            )
            return True  # continue

        try:
            self.queue.put({"status": "Started loading..."})

            df = load_experiment_tables(
                experiment=self.experiment,
                population=self.population,
                well_option=self.well_option,
                position_option=self.position_option,
                return_pos_info=False,
                progress_callback=progress,
            )

            self.queue.put({"status": "finished", "result": df})

        except Exception as e:
            logger.error(f"Table loading failed: {e}")
            self.queue.put({"status": "error", "message": str(e)})

    def end_process(self):
        """End the process."""
        self.terminate()
