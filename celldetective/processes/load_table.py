import time
from multiprocessing import Process
from celldetective.utils.data_loaders import load_experiment_tables
from celldetective import get_logger

logger = get_logger()


class TableLoaderProcess(Process):

    def __init__(self, queue=None, process_args=None, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if process_args is not None:
            for key, value in process_args.items():
                setattr(self, key, value)

        self.queue = queue

    def run(self):

        def progress(well_progress, pos_progress):
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
        self.terminate()
