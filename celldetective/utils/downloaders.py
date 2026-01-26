import json
import os
import shutil
import tempfile
import zipfile
from glob import glob
from urllib.request import urlopen

import numpy as np
from tqdm import tqdm

from celldetective.utils.io import remove_file_if_exists
from celldetective import get_logger

logger = get_logger()


def get_zenodo_files(cat=None):

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


def download_url_to_file(url, dst, progress=True):
    r"""Download object at the given URL to a local path.
                    Thanks to torch, slightly modified, from Cellpose
    Args:
            url (string): URL of the object to download
            dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
            progress (bool, optional): whether to display a progress bar to stderr
                    Default: True

    """
    import ssl
    import time
    from urllib.error import HTTPError, URLError

    file_size = None
    ssl._create_default_https_context = ssl._create_unverified_context

    # Retry configuration
    max_retries = 5
    retry_delay = 10  # Initial delay in seconds

    for attempt in range(max_retries):
        try:
            u = urlopen(url)
            meta = u.info()
            if hasattr(meta, "getheaders"):
                content_length = meta.getheaders("Content-Length")
            else:
                content_length = meta.get_all("Content-Length")
            if content_length is not None and len(content_length) > 0:
                file_size = int(content_length[0])
            break  # Success
        except (HTTPError, URLError) as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Download check failed: {e}. Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"Download check failed after {max_retries} attempts: {e}")
                raise e

    # We deliberately save it in a temp file and move it after
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    # GUI Check
    try:
        from PyQt5.QtWidgets import QApplication, QProgressDialog
        from PyQt5.QtCore import Qt

        app = QApplication.instance()
        use_gui = app is not None
    except ImportError:
        use_gui = False

    try:
        if use_gui and progress:
            # Setup QProgressDialog
            pd = QProgressDialog("Downloading...", "Cancel", 0, 100)
            pd.setWindowTitle("Downloading content")
            pd.setWindowModality(Qt.WindowModal)
            pd.setMinimumDuration(0)
            pd.setValue(0)

            downloaded = 0
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                downloaded += len(buffer)
                if file_size:
                    perc = int(downloaded * 100 / file_size)
                    pd.setValue(perc)
                    pd.setLabelText(
                        f"Downloading... {downloaded/1024/1024:.1f}/{file_size/1024/1024:.1f} MB"
                    )

                QApplication.processEvents()
                if pd.wasCanceled():
                    print("Download cancelled by user.")
                    break
            pd.close()

        else:
            # Console / TQDM fallback
            with tqdm(
                total=file_size,
                disable=not progress,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                while True:
                    try:
                        buffer = u.read(8192)  # 8192
                        if len(buffer) == 0:
                            break
                        f.write(buffer)
                        pbar.update(len(buffer))
                    except (HTTPError, URLError) as e:
                        # Attempt rudimentary resume-like behavior or just fail?
                        # Simple retry of read is hard without Range headers on a stream.
                        # Best to just fail the whole download and rely on outer retry if we wrapped the whole thing.
                        # For now, let's just let it raise, but really we should wrap the whole download block.
                        raise e

        f.close()
        shutil.move(f.name, dst)
    except Exception as e:
        f.close()
        remove_file_if_exists(f.name)
        # If we failed during download reading (after open), we should probably retry the whole function from start
        # but that requires significant refactoring. Given the error was 504 on open, the retry block above handles it.
        raise e
    finally:
        f.close()
        remove_file_if_exists(f.name)


def download_zenodo_file(file, output_dir):

    logger.info(f"{file=} {output_dir=}")

    # GUI Check
    try:
        from PyQt5.QtWidgets import QApplication, QDialog

        app = QApplication.instance()
        use_gui = app is not None
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
    download_url_to_file(rf"{zip_url}", path_to_zip_file)
    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    file_to_rename = glob(
        os.sep.join(
            [output_dir, file, "*[!.json][!.png][!.h5][!.csv][!.npy][!.tif][!.ini]"]
        )
    )
    if (
        len(file_to_rename) > 0
        and not file_to_rename[0].endswith(os.sep)
        and not file.startswith("demo")
    ):
        os.rename(file_to_rename[0], os.sep.join([output_dir, file, file]))

    os.remove(path_to_zip_file)
