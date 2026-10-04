import logging
import os
from multiprocessing import Process, Queue

logger = logging.getLogger("celldetective")
from typing import Optional, Dict, Any
import time
import json

from celldetective.utils.io import remove_file_if_exists


class DownloadProcess(Process):

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

        # Get celldetective package root
        current_dir = os.path.dirname(os.path.realpath(__file__))
        package_root = os.path.dirname(current_dir)
        zenodo_json = os.path.join(package_root, "links", "zenodo.json")
        with open(zenodo_json, "r") as f:
            zenodo_json = json.load(f)
        all_files = list(zenodo_json["files"]["entries"].keys())
        all_files_short = [f.replace(".zip", "") for f in all_files]
        zenodo_url = zenodo_json["links"]["files"].replace("api/", "")
        full_links = ["/".join([zenodo_url, f]) for f in all_files]
        index = all_files_short.index(self.file)

        self.zip_url = full_links[index]
        self.path_to_zip_file = os.sep.join([self.output_dir, "temp.zip"])

        self.t0 = time.time()

    def download_url_to_file(self, url: str, dst: str) -> None:
        """
        Download a file from a URL, reporting the progress to the queue.

        Parameters
        ----------
        url : str
            The URL to download from.
        dst : str
            The destination file path.

        Raises
        ------
        Exception
            If the download fails once transient errors have been retried.
        """
        from celldetective.utils.downloaders import stream_url_to_file

        self.queue.put({"status": "Contacting Zenodo..."})

        # The last tenth of a percent reported: one message per 8 KiB chunk would
        # flood the queue the progress window reads (some 130,000 for a 1 GB file).
        last_step = -1

        def on_chunk(downloaded: int, file_size: Optional[int]) -> None:
            nonlocal last_step
            if downloaded == 0:
                self.queue.put({"status": "Downloading..."})
            elif file_size:
                pct = downloaded / file_size * 100
                step = int(pct * 10)
                if step == last_step:
                    return
                last_step = step
                mean_exec_per_step = (time.time() - self.t0) / (downloaded + 1)
                pred_time = (file_size - (downloaded + 1)) * mean_exec_per_step
                self.queue.put([pct, pred_time])

        stream_url_to_file(url, dst, on_chunk)

    def run(self):
        """Run the download process."""

        try:
            self._download_and_extract()
        except Exception as e:
            logger.error(f"Download of {self.file} failed: {e}")
            remove_file_if_exists(self.path_to_zip_file)
            self.queue.put(
                {
                    "status": "error",
                    "message": f"Could not download {self.file} from Zenodo: {e}",
                }
            )
            self.queue.close()
            return

        # Send end signal
        self.queue.put("finished")
        self.queue.close()

    def _download_and_extract(self):
        """Download the zip archive, extract it and tidy up the model folder."""

        from celldetective.utils.downloaders import extract_zenodo_archive

        self.download_url_to_file(rf"{self.zip_url}", self.path_to_zip_file)
        extract_zenodo_archive(self.path_to_zip_file, self.output_dir, self.file)
        os.remove(self.path_to_zip_file)
        self.queue.put([100, 0])
        time.sleep(0.5)

    def end_process(self):
        """End the process."""

        self.terminate()
        self.queue.put("finished")

    def abort_process(self):
        """Abort the process."""

        self.terminate()
        self.queue.put("error")
