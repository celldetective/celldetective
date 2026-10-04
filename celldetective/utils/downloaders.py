import json
import os
import shutil
import tempfile
import time
import zipfile
from glob import glob
from urllib.request import urlopen

import numpy as np
from tqdm import tqdm

from celldetective.utils.io import remove_file_if_exists
from celldetective import get_logger
from typing import Callable, Optional, List, Union, Tuple

logger = get_logger()


def get_zenodo_files(
    cat: Optional[str] = None,
) -> Union[List[str], Tuple[List[str], List[str]]]:
    """
    Get list of files available on Zenodo.

    Parameters
    ----------
    cat : str, optional
        Category of files to retrieve. Default is None.

    Returns
    -------
    list or tuple
        List of files, or (list of files, list of categories).
    """

    zenodo_json = os.sep.join(
        [
            os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],
            # "celldetective",
            "links",
            "zenodo.json",
        ]
    )
    with open(zenodo_json, "r") as f:
        zenodo_json = json.load(f)
    all_files = list(zenodo_json["files"]["entries"].keys())
    all_files_short = [f.replace(".zip", "") for f in all_files]

    categories = []
    for f in all_files_short:
        if f.startswith("CP") or f.startswith("SD"):
            category = os.sep.join(["models", "segmentation_generic"])
        elif f.startswith("MCF7") or f.startswith("mcf7"):
            category = os.sep.join(["models", "segmentation_targets"])
        elif f.startswith("primNK") or f.startswith("lymphocytes"):
            category = os.sep.join(["models", "segmentation_effectors"])
        elif f.startswith("demo"):
            category = "demos"
        elif f.startswith("db-si"):
            category = os.sep.join(["datasets", "signal_annotations"])
        elif f.startswith("db"):
            category = os.sep.join(["datasets", "segmentation_annotations"])
        else:
            category = os.sep.join(["models", "signal_detection"])
        categories.append(category)

    if cat is not None:
        if cat in [
            os.sep.join(["models", "segmentation_generic"]),
            os.sep.join(["models", "segmentation_targets"]),
            os.sep.join(["models", "segmentation_effectors"]),
            "demos",
            os.sep.join(["datasets", "signal_annotations"]),
            os.sep.join(["datasets", "segmentation_annotations"]),
            os.sep.join(["models", "signal_detection"]),
        ]:
            categories = np.array(categories)
            all_files_short = np.array(all_files_short)
            return list(all_files_short[np.where(categories == cat)[0]])
        else:
            return []
    else:
        return all_files_short, categories


# Transient server- and network-side conditions: the file is expected to be
# there, the host just could not serve it this second.
RETRYABLE_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def _is_retryable(error: Exception) -> bool:
    """
    Return whether a failed download attempt is worth repeating.

    Parameters
    ----------
    error : Exception
        The exception raised by the attempt.

    Returns
    -------
    bool
        True for transient conditions (timeouts, dropped connections, 5xx and
        friends), False for answers that will not change on a retry, such as a
        404 for a file that is simply not published.
    """

    import socket
    from urllib.error import HTTPError, URLError

    if isinstance(error, HTTPError):
        return error.code in RETRYABLE_HTTP_STATUS
    if isinstance(error, URLError):
        return True
    return isinstance(error, (socket.timeout, ConnectionError, TimeoutError))


def open_url_with_retries(url: str):
    """
    Open a URL, retrying transient failures with a jittered backoff.

    Parameters
    ----------
    url : str
        URL of the object to download.

    Returns
    -------
    tuple
        (response, file_size), where file_size is None when the server sends
        no Content-Length.

    Raises
    ------
    Exception
        The last error, once retries are exhausted or the error is not
        retryable (e.g. a 404).
    """
    import random
    import socket
    import ssl
    from urllib.error import HTTPError, URLError

    ssl._create_default_https_context = ssl._create_unverified_context

    # Retry configuration
    max_retries = 7
    retry_delay = 5  # Initial delay in seconds
    max_retry_delay = 60  # ~4 min of retries in total, jitter included

    for attempt in range(max_retries):
        try:
            u = urlopen(url, timeout=60)
            file_size = None
            meta = u.info()
            if hasattr(meta, "getheaders"):
                content_length = meta.getheaders("Content-Length")
            else:
                content_length = meta.get_all("Content-Length")
            if content_length is not None and len(content_length) > 0:
                file_size = int(content_length[0])
            return u, file_size
        except (HTTPError, URLError, socket.timeout, OSError) as e:
            last_attempt = attempt == max_retries - 1
            if last_attempt or not _is_retryable(e):
                # A 404 is the host telling us the file is not there; sleeping
                # through the whole backoff schedule before saying so only
                # delays the error by minutes.
                logger.error(
                    f"Download of {url} failed after {attempt + 1} attempt(s): {e}"
                )
                raise

            # Zenodo answers 502/504 while it is staging an archive, and the
            # window can outlast a short schedule. Jitter keeps the parallel CI
            # jobs from retrying in lockstep and re-timing-out together.
            delay = min(retry_delay, max_retry_delay)
            delay += random.uniform(0, delay / 2)
            logger.warning(
                f"Download check failed ({e}). "
                f"Retry {attempt + 2}/{max_retries} in {delay:.0f}s..."
            )
            time.sleep(delay)
            retry_delay = min(retry_delay * 2, max_retry_delay)


class DownloadCancelled(Exception):
    """The user cancelled a download from its progress dialog."""


class IncompleteDownloadError(OSError):
    """The connection closed before the whole file was received."""


def check_download_complete(path: str, file_size: Optional[int], url: str) -> None:
    """
    Check that a downloaded file has the size the server announced.

    A connection dropped mid-transfer can end the read loop as a plain end of
    stream, so a truncated file would otherwise be kept as if it were complete.

    Parameters
    ----------
    path : str
        The downloaded file.
    file_size : int or None
        The Content-Length the server sent, None if it sent none (nothing to
        check against then).
    url : str
        Where the file came from, for the error message.

    Raises
    ------
    IncompleteDownloadError
        If the file is not the announced size.
    """

    if file_size is None:
        return
    received = os.path.getsize(path)
    if received != file_size:
        raise IncompleteDownloadError(
            f"Download of {url} incomplete: received {received} of {file_size} bytes."
        )


def _merge_into(src: str, dst: str) -> None:
    """Move the content of directory `src` into the existing directory `dst`."""

    for entry in os.listdir(src):
        s, d = os.path.join(src, entry), os.path.join(dst, entry)
        if os.path.isdir(s) and os.path.isdir(d):
            _merge_into(s, d)
        else:
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.replace(s, d)


# Age [s] past which a staging folder is taken for that of an extraction killed
# half-way: a younger one may be that of an extraction still running.
STALE_EXTRACTION_AGE = 3600


def extract_zenodo_archive(path_to_zip_file: str, output_dir: str, file: str) -> None:
    """
    Extract a Zenodo archive into `output_dir`, all or nothing.

    The archive is extracted into a hidden staging folder of `output_dir`, and
    its folders are moved into place only once complete. Extracting straight
    into `output_dir` left a partial model folder behind when the extraction
    was interrupted -- a download cancelled from its progress window terminates
    the process doing it -- and that folder was then taken for the installed
    model and never downloaded again, failing at every load instead.

    Parameters
    ----------
    path_to_zip_file : str
        The downloaded archive.
    output_dir : str
        The folder the archive's content goes into.
    file : str
        The name of the Zenodo entry (archive name without ``.zip``); the
        weights file of a model folder of that name is renamed after it.
    """

    # Staging folders of an extraction that was killed half-way, leaving alone
    # those of another download extracting into the same folder right now.
    for staged in glob(os.path.join(output_dir, ".extract-*")):
        try:
            age = time.time() - os.path.getmtime(staged)
        except OSError:
            continue
        if age > STALE_EXTRACTION_AGE:
            shutil.rmtree(staged, ignore_errors=True)

    staging = tempfile.mkdtemp(prefix=".extract-", dir=output_dir)
    try:
        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(staging)

        file_to_rename = glob(
            os.sep.join(
                [staging, file, "*[!.json][!.png][!.h5][!.csv][!.npy][!.tif][!.ini]"]
            )
        )
        if (
            len(file_to_rename) > 0
            and not file_to_rename[0].endswith(os.sep)
            and not file.startswith("demo")
        ):
            os.rename(file_to_rename[0], os.sep.join([staging, file, file]))

        # Extracting over an existing folder (a demo downloaded again) overwrites
        # what the archive holds and keeps the rest, as extracting in place did.
        _merge_into(staging, output_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def stream_url_to_file(
    url: str,
    dst: str,
    on_chunk: Optional[Callable[[int, Optional[int]], None]] = None,
) -> None:
    """
    Download the object at a URL to a local path, all or nothing.

    The object is written to a temporary file next to `dst`, moved into place
    only once the size the server announced is checked.

    Parameters
    ----------
    url : str
        URL of the object to download.
    dst : str
        Full path where the object is saved.
    on_chunk : callable, optional
        Called as ``on_chunk(downloaded, file_size)`` once the URL is open, then
        after each chunk; ``file_size`` is None if the server announced none. An
        exception raised from it aborts the download.
    """

    dst = os.path.expanduser(dst)
    u, file_size = open_url_with_retries(url)
    try:
        f = tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(dst))
    except BaseException:
        # No file to write to (e.g. a missing or read-only folder): the response is
        # closed all the same.
        u.close()
        raise
    try:
        downloaded = 0
        if on_chunk is not None:
            on_chunk(downloaded, file_size)
        while True:
            buffer = u.read(8192)
            if len(buffer) == 0:
                break
            f.write(buffer)
            downloaded += len(buffer)
            if on_chunk is not None:
                on_chunk(downloaded, file_size)
        f.close()
        check_download_complete(f.name, file_size, url)
        shutil.move(f.name, dst)
    finally:
        u.close()
        f.close()
        remove_file_if_exists(f.name)


def download_url_to_file(url: str, dst: str, progress: bool = True) -> None:
    r"""
    Download object at the given URL to a local path.
    Thanks to torch, slightly modified, from Cellpose

    Parameters
    ----------
    url : str
        URL of the object to download.
    dst : str
        Full path where object will be saved, e.g. `/tmp/temporary_file`.
    progress : bool, optional
        Whether to display a progress bar to stderr. Default is True.
    """

    # GUI Check
    try:
        from PyQt5.QtWidgets import QApplication, QProgressDialog
        from PyQt5.QtCore import Qt, QThread

        app = QApplication.instance()
        # Widgets may only be built, and the event loop only pumped, from the
        # thread the application lives on. A download started from a worker --
        # the napari single-frame panel fetches a model that way -- would
        # otherwise put a QProgressDialog and a `processEvents` on a non-GUI
        # thread, which hangs the whole interface rather than raising. Off the
        # GUI thread we fall through to the console bar instead.
        use_gui = app is not None and QThread.currentThread() is app.thread()
    except ImportError:
        use_gui = False

    if use_gui and progress:
        pd = QProgressDialog("Downloading...", "Cancel", 0, 100)
        pd.setWindowTitle("Downloading content")
        pd.setWindowModality(Qt.WindowModal)
        pd.setMinimumDuration(0)
        pd.setValue(0)

        def on_chunk(downloaded: int, file_size: Optional[int]) -> None:
            if file_size:
                pd.setValue(int(downloaded * 100 / file_size))
                pd.setLabelText(
                    f"Downloading... {downloaded/1024/1024:.1f}/{file_size/1024/1024:.1f} MB"
                )
            QApplication.processEvents()
            if pd.wasCanceled():
                raise DownloadCancelled(f"Download of {url} cancelled.")

        try:
            stream_url_to_file(url, dst, on_chunk)
        finally:
            pd.close()
    else:
        with tqdm(
            disable=not progress, unit="B", unit_scale=True, unit_divisor=1024
        ) as pbar:

            def on_chunk(downloaded: int, file_size: Optional[int]) -> None:
                if downloaded == 0 and file_size:
                    pbar.reset(total=file_size)
                pbar.update(downloaded - pbar.n)

            stream_url_to_file(url, dst, on_chunk)


def download_zenodo_file(file: str, output_dir: str) -> None:
    """
    Download a file from Zenodo.

    Parameters
    ----------
    file : str
        Name of the file to download.
    output_dir : str
        Directory to save the downloaded file.
    """

    logger.info(f"{file=} {output_dir=}")

    # GUI Check
    try:
        from PyQt5.QtWidgets import QApplication, QDialog
        from PyQt5.QtCore import QThread

        app = QApplication.instance()
        # Only from the thread the application lives on. The progress window is
        # a widget run with a modal `exec_()`, and both are GUI-thread-only: a
        # download started from a worker -- the napari single-frame panel
        # fetches a model that way -- would hang the interface rather than
        # raise. Off the GUI thread the console implementation below runs
        # instead, and the caller reports progress its own way.
        use_gui = app is not None and QThread.currentThread() is app.thread()
    except ImportError:
        use_gui = False

    if use_gui:
        try:
            from celldetective.gui.workers import GenericProgressWindow
            from celldetective.processes.downloader import DownloadProcess

            # Find parent window if possible, else None is fine for a dialog
            parent = app.activeWindow()

            process_args = {"output_dir": output_dir, "file": file}
            job = GenericProgressWindow(
                DownloadProcess,
                parent_window=parent,
                title="Download",
                process_args=process_args,
                label_text=f"Downloading {file}...",
            )
            result = job.exec_()
            if result == QDialog.Accepted:
                return  # DownloadProcess handles the file operations
            else:
                logger.info("Download cancelled or failed.")
                return

        except Exception as e:
            logger.error(f"Failed to use GUI downloader: {e}. Falling back to console.")
            # Fallback to console implementation below if GUI fails

    # Console Implementation
    zenodo_json = os.sep.join(
        [
            os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],
            # "celldetective",
            "links",
            "zenodo.json",
        ]
    )
    logger.info(f"{zenodo_json=}")
    with open(zenodo_json, "r") as f:
        zenodo_json = json.load(f)
    all_files = list(zenodo_json["files"]["entries"].keys())
    all_files_short = [f.replace(".zip", "") for f in all_files]
    zenodo_url = zenodo_json["links"]["files"].replace("api/", "")
    full_links = ["/".join([zenodo_url, f]) for f in all_files]
    index = all_files_short.index(file)
    zip_url = full_links[index]

    path_to_zip_file = os.sep.join([output_dir, "temp.zip"])
    try:
        download_url_to_file(rf"{zip_url}", path_to_zip_file)
        extract_zenodo_archive(path_to_zip_file, output_dir, file)
    except DownloadCancelled:
        logger.info("Download cancelled or failed.")
    finally:
        remove_file_if_exists(path_to_zip_file)
