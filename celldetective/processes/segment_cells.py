from multiprocessing import Process, Queue
from typing import Optional, Dict, Any, List
import time
import datetime
import os
import json
from pathlib import Path, PurePath
from glob import glob
from shutil import rmtree
from tqdm import tqdm
import numpy as np
import gc
from art import tprint
import concurrent.futures

from celldetective.log_manager import get_logger
from celldetective.utils.experiment import (
    extract_position_name,
    extract_experiment_channels,
)
from celldetective.utils.image_loaders import (
    auto_load_number_of_frames,
    _get_img_num_per_channel,
    _load_frames_to_segment,
    load_frames,
)
from celldetective.utils.image_transforms import _estimate_scale_factor
from celldetective.utils.mask_cleaning import _check_label_dims
from celldetective.utils.mask_transforms import _rescale_labels
from celldetective.utils.model_loaders import locate_segmentation_model
from celldetective.utils.parsing import (
    config_section_to_dict,
    _extract_nbr_channels_from_config,
    _get_normalize_kwargs_from_config,
    _extract_channel_indices_from_config,
)

logger = get_logger(__name__)


def _blas_thread_limit(limits):
    """
    Context manager that caps native BLAS/OpenMP threads to ``limits``.

    Used to avoid CPU oversubscription when running our own thread pool on top
    of numpy/skimage. Degrades to a no-op context manager if ``threadpoolctl``
    is not installed (it normally is, via scikit-image/scikit-learn).
    """
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=limits)
    except Exception:
        from contextlib import nullcontext

        return nullcontext()


def _create_preview_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Create a preview overlay of the segmentation mask on the image.

    Parameters
    ----------
    image : ndarray
        The input image.
    mask : ndarray
        The segmentation mask.

    Returns
    -------
    img : ndarray
        The overlay image.
    """
    # If image has channels (C, Y, X), take max projection or first channel
    if image.ndim == 3:
        image = np.max(image, axis=0)

    # Robust handling for shape mismatch (e.g. transpose or scale issues)
    if image.shape != mask.shape:
        # Try transpose match
        if image.T.shape == mask.shape:
            image = image.T
        else:
            # Resize image to fit mask (mask is the truth for segmentation result)
            from skimage.transform import resize

            # normalize to float 0-1 before resize to avoid artifacts
            image = image.astype(float)
            image = resize(image, mask.shape, preserve_range=True)

    # Normalize image to dim range 0-150 (uint8)
    img = image.copy().astype(float)
    img = np.nan_to_num(img)
    min_v, max_v = np.min(img), np.max(img)
    if max_v > min_v:
        img = (img - min_v) / (max_v - min_v) * 150  # Darker context
    else:
        img = np.zeros_like(img)
    img = img.astype(np.uint8)

    # Overlay: Set mask region to 255 (Bright White)
    img[mask > 0] = 255

    return img  # Returns 2D uint8, handled robustly by workers.py


class BaseSegmentProcess(Process):

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

        self.queue = queue

        if process_args is not None:
            for key, value in process_args.items():
                setattr(self, key, value)

        # Handle batch of positions or single pos
        if hasattr(self, "batch_structure"):
            # Flatten positions from structure for compatibility
            self.positions = []
            for w_idx, data in self.batch_structure.items():
                self.positions.extend(data["positions"])
        elif not hasattr(self, "positions"):
            if hasattr(self, "pos"):
                self.positions = [self.pos]
            else:
                self.positions = []
                logger.error("No positions provided to segmentation process.")

        # Experiment
        self.locate_experiment_config()

        logger.info(f"Configuration file: {self.config}")
        logger.info(f"Population: {self.mode}...")
        self.instruction_file = os.sep.join(
            ["configs", f"segmentation_instructions_{self.mode}.json"]
        )
        self.read_instructions()
        self.extract_experiment_parameters()

    def setup_for_position(self, pos_path: str) -> None:
        """
        Setup the process for a specific position.

        Parameters
        ----------
        pos_path : str
            The path to the position.
        """
        self.pos = pos_path
        logger.info(f"Position: {extract_position_name(self.pos)}...")
        logger.info(f"Population: {self.mode}...")

        self.detect_movie_length()
        self.write_folders()

    def read_instructions(self):
        """Read the segmentation instructions from the configuration file."""
        logger.info("Looking for instruction file...")
        instr_path = PurePath(self.exp_dir, Path(f"{self.instruction_file}"))
        if os.path.exists(instr_path):
            with open(instr_path, "r") as f:
                _instructions = json.load(f)
                logger.info(f"Measurement instruction file successfully loaded...")
                logger.info(f"Instructions: {_instructions}...")
            self.flip = _instructions.get("flip", False)
        else:
            self.flip = False

    def write_folders(self):
        """
        Prepare the labels folder, preserving the previous masks as a backup.

        New masks are written *directly* into ``labels_<mode>`` so they remain
        viewable (e.g. in napari) live and even after an abort, instead of being
        hidden in a temporary folder. The previous masks are not deleted up
        front: they are renamed to ``labels_<mode>.bak`` and only removed by
        :meth:`finalize_position` once the new run has produced every frame. If
        the run fails or is cancelled, the partial new masks stay in
        ``labels_<mode>`` (inspectable) and the previous masks remain recoverable
        in the backup folder.
        """

        self.mode = self.mode.lower()
        self.label_folder = f"labels_{self.mode}"
        self.backup_label_folder = f"labels_{self.mode}.bak"

        final_path = self.pos + self.label_folder
        backup_path = self.pos + self.backup_label_folder

        # Drop any stale backup left behind by a previously aborted run.
        if os.path.exists(backup_path):
            rmtree(backup_path)
        # Rename (not delete) the previous masks so they can be recovered.
        if os.path.exists(final_path):
            os.rename(final_path, backup_path)
        os.mkdir(final_path)
        logger.info("Labels folder successfully generated...")

    def finalize_position(self):
        """
        Verify the run is complete, then drop the backup of the previous masks.

        On success the previous masks (``labels_<mode>.bak``) are removed. On an
        incomplete run this raises, leaving the partial new masks in
        ``labels_<mode>`` (so they can be inspected) and the previous masks in
        the backup folder (so they can be recovered).

        Raises
        ------
        RuntimeError
            If the number of mask files written does not match the expected
            number of frames (i.e. some frame failed to segment).
        """

        final_path = self.pos + self.label_folder
        backup_path = self.pos + self.backup_label_folder

        n_written = len(glob(os.sep.join([final_path, "*.tif"])))
        expected = int(self.len_movie)
        if n_written != expected:
            raise RuntimeError(
                f"Segmentation incomplete for {extract_position_name(self.pos)}: "
                f"{n_written}/{expected} frame masks were written. The partial "
                f"masks are kept in '{self.label_folder}' (viewable); the previous "
                f"masks remain in '{self.backup_label_folder}'."
            )

        # Success: the new masks are already in place; discard the backup.
        if os.path.exists(backup_path):
            rmtree(backup_path)
        logger.info(f"Labels folder successfully generated ({n_written} frames)...")

    def extract_experiment_parameters(self):
        """Extract experiment parameters from the configuration file."""

        self.spatial_calibration = float(
            config_section_to_dict(self.config, "MovieSettings")["pxtoum"]
        )
        self.len_movie = float(
            config_section_to_dict(self.config, "MovieSettings")["len_movie"]
        )
        self.movie_prefix = config_section_to_dict(self.config, "MovieSettings")[
            "movie_prefix"
        ]
        self.nbr_channels = _extract_nbr_channels_from_config(self.config)
        self.channel_names, self.channel_indices = extract_experiment_channels(
            self.exp_dir
        )

    def locate_experiment_config(self):
        """Locate the experiment configuration file."""

        if hasattr(self, "pos"):
            p = self.pos
        elif hasattr(self, "positions") and len(self.positions) > 0:
            p = self.positions[0]
        else:
            logger.error("No position available to locate experiment config.")
            return

        parent1 = Path(p).parent
        self.exp_dir = parent1.parent
        self.config = PurePath(self.exp_dir, Path("config.ini"))

        if not os.path.exists(self.config):
            logger.error(
                "The configuration file for the experiment could not be located. Abort."
            )
            self.abort_process()

    def detect_movie_length(self):
        """Detect the length of the movie."""

        try:
            self.file = glob(self.pos + f"movie/{self.movie_prefix}*.tif")[0]
        except Exception as e:
            logger.error(f"Error {e}.\nMovie could not be found. Check the prefix.")
            self.abort_process()

        len_movie_auto = auto_load_number_of_frames(self.file)
        if len_movie_auto is not None:
            self.len_movie = len_movie_auto

    def end_process(self):
        """End the process."""

        self.terminate()
        self.queue.put("finished")

    def abort_process(self):
        """Abort the process."""

        self.terminate()
        self.queue.put("error")


class SegmentCellDLProcess(BaseSegmentProcess):

    def __init__(self, *args, **kwargs):
        """
        Initialize the process.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """

        super().__init__(*args, **kwargs)

        # Model (must come before check_gpu so model_type is known).
        # NOTE: check_gpu() is intentionally NOT called here — it imports torch,
        # which is slow, and __init__ runs on the GUI thread (the Process object
        # is built by the Runner before start()). It is deferred to run(), which
        # executes in the spawned child, so the progress window stays responsive.
        self.locate_model_path()
        self.extract_model_input_parameters()
        self.detect_rescaling()

        self.sum_done = 0
        self.t0 = time.time()

    def setup_for_position(self, pos_path: str) -> None:
        """
        Setup the process for a specific position.

        Parameters
        ----------
        pos_path : str
            The path to the position.
        """
        super().setup_for_position(pos_path)
        self.detect_channels()
        self.write_log()

    def extract_model_input_parameters(self):
        """Extract the model input parameters from the configuration file."""

        self.required_channels = self.input_config["channels"]
        if "selected_channels" in self.input_config:
            self.required_channels = self.input_config["selected_channels"]

        self.target_cell_size = None
        if (
            "target_cell_size_um" in self.input_config
            and "cell_size_um" in self.input_config
        ):
            self.target_cell_size = self.input_config["target_cell_size_um"]
            self.cell_size = self.input_config["cell_size_um"]

        self.normalize_kwargs = _get_normalize_kwargs_from_config(self.input_config)

        self.model_type = self.input_config["model_type"]
        self.required_spatial_calibration = self.input_config["spatial_calibration"]
        logger.info(
            f"Spatial calibration expected by the model: {self.required_spatial_calibration}..."
        )

        if self.model_type == "cellpose":
            self.diameter = self.input_config["diameter"]
            self.cellprob_threshold = self.input_config["cellprob_threshold"]
            self.flow_threshold = self.input_config["flow_threshold"]

    def write_log(self):
        """Write the logo to the log file."""

        log = f"segmentation model: {self.model_name}\n"
        with open(self.pos + f"log_{self.mode}.txt", "a") as f:
            f.write(f"{datetime.datetime.now()} SEGMENT \n")
            f.write(log)

    def detect_channels(self):
        """Detect the channels required for the model."""

        self.channel_indices = _extract_channel_indices_from_config(
            self.config, self.required_channels
        )
        logger.info(
            f"Required channels: {self.required_channels} located at channel indices {self.channel_indices}."
        )
        if self.channel_indices is None:
            raise ValueError(
                f"Could not resolve channel indices for required channels {self.required_channels}. "
                "Check that the model's required channels match the channels configured for this position."
            )
        self.img_num_channels = _get_img_num_per_channel(
            self.channel_indices, int(self.len_movie), self.nbr_channels
        )

    def detect_rescaling(self):
        """Detect the rescheduling factor for the images."""

        self.scale = _estimate_scale_factor(
            self.spatial_calibration, self.required_spatial_calibration
        )
        logger.info(f"Scale: {self.scale} [None = 1]...")

        if self.target_cell_size is not None and self.scale is not None:
            self.scale *= self.cell_size / self.target_cell_size
        elif self.target_cell_size is not None:
            if self.target_cell_size != self.cell_size:
                self.scale = self.cell_size / self.target_cell_size

        logger.info(
            f"Scale accounting for expected cell size: {self.scale} [None = 1]..."
        )

    def locate_model_path(self):
        """Locate the path to the model."""

        self.model_complete_path = locate_segmentation_model(self.model_name)
        if self.model_complete_path is None:
            logger.error("Model could not be found. Abort.")
            self.abort_process()
        else:
            logger.info(f"Model path: {self.model_complete_path}...")

        if not os.path.exists(self.model_complete_path + "config_input.json"):
            logger.error(
                "The configuration for the inputs to the model could not be located. Abort."
            )
            self.abort_process()

        with open(self.model_complete_path + "config_input.json") as config_file:
            self.input_config = json.load(config_file)

    def check_gpu(self):
        """Check if GPU is available and compatible, falling back to CPU if not."""

        if not self.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            return

        # Pin the requested device (default "0") before any GPU context is
        # created, so multi-GPU machines can target a specific device via the
        # CELLDETECTIVE_GPU_DEVICE environment variable.
        from celldetective.utils.resources import resolve_gpu_device

        os.environ["CUDA_VISIBLE_DEVICES"] = resolve_gpu_device()

        if getattr(self, "model_type", None) == "cellpose":
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available.")
                # Verify the GPU is actually usable with this PyTorch build
                torch.zeros(1, device="cuda")
            except Exception as e:
                logger.warning(
                    f"GPU requested but could not be used ({e}). Falling back to CPU."
                )
                self.use_gpu = False
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    def process_position(
        self, model: Optional[Any] = None, scale_model: Optional[Any] = None
    ) -> None:
        """
        Process a single position.

        Parameters
        ----------
        model : object
            The segmentation model.
        scale_model : object
            The scaling model.
        """

        tprint("Segment")

        list_indices = range(self.len_movie)
        if self.flip:
            list_indices = reversed(list_indices)

        # Reset counter for this position
        self.loop_count = 0
        self.t0_frame = time.time()

        for t in tqdm(list_indices, desc="frame"):

            f = _load_frames_to_segment(
                self.file,
                self.img_num_channels[:, t],
                scale_model=scale_model,
                normalize_kwargs=self.normalize_kwargs,
            )

            # Shared per-frame inference + label rescaling, identical to the
            # in-memory segment() API and the CLI script.
            from celldetective.segmentation import _run_dl_model_on_frame

            Y_pred = _run_dl_model_on_frame(
                f,
                model=model,
                model_type=self.model_type,
                scale_model=scale_model,
                file=self.file,
                diameter=getattr(self, "diameter", None),
                cellprob_threshold=getattr(self, "cellprob_threshold", None),
                flow_threshold=getattr(self, "flow_threshold", None),
            )

            from celldetective.utils.io import save_tiff_imagej_compatible

            save_tiff_imagej_compatible(
                self.pos + os.sep.join([self.label_folder, f"{str(t).zfill(4)}.tif"]),
                Y_pred,
                axes="YX",
            )

            # del f
            # del Y_pred
            # gc.collect()

            # Send signal for progress bar
            # Triple progress bar logic

            if self.loop_count == 0:
                self.t0_frame = time.time()

            frame_progress = ((self.loop_count + 1) / self.len_movie) * 100
            if frame_progress > 100:
                frame_progress = 100

            data = {}
            data["frame_progress"] = frame_progress

            # Frame time estimation (skip first)
            elapsed = time.time() - getattr(self, "t0_frame", time.time())
            measured_count = self.loop_count

            if measured_count > 0:
                avg = elapsed / measured_count
                rem = self.len_movie - (self.loop_count + 1)
                rem_t = rem * avg
                mins = int(rem_t // 60)
                secs = int(rem_t % 60)
                data["frame_time"] = f"Segmentation: {mins} m {secs} s"
            else:
                data["frame_time"] = (
                    f"Segmentation: {self.loop_count + 1}/{int(self.len_movie)} frames"
                )

            # Saturate preview: Convert labels to binary (0/1) so all cells are visible
            # data["image_preview"] = Y_pred > 0
            # Saturate preview: Convert labels to binary (0/1) so all cells are visible
            data["image_preview"] = (Y_pred > 0).astype(np.uint8)
            self.queue.put(data)
            self.loop_count += 1

            del f
            del Y_pred
            gc.collect()

    def run(self):
        """Run the segmentation process."""

        try:

            # GPU check is done here (in the child) rather than __init__ so its
            # torch import never blocks the GUI thread.
            self.check_gpu()

            # Loading the DL backend (TensorFlow/PyTorch) and the model weights
            # is the dominant startup cost and emits no progress, so tell the
            # user what is happening instead of leaving the bars at 0%.
            self.queue.put({"status": f"Loading {self.model_type} model…"})

            if self.model_type == "stardist":
                from celldetective.utils.stardist_utils import _prep_stardist_model

                # Enable TF memory growth before StarDist initializes the GPU,
                # so it does not pre-allocate all VRAM.
                if self.use_gpu:
                    from celldetective.utils.resources import configure_memory_growth

                    configure_memory_growth()

                model, scale_model = _prep_stardist_model(
                    self.model_name,
                    Path(self.model_complete_path).parent,
                    use_gpu=self.use_gpu,
                    scale=self.scale,
                )

            elif self.model_type == "cellpose":
                from celldetective.utils.cellpose_utils import _prep_cellpose_model

                model, scale_model = _prep_cellpose_model(
                    self.model_name,
                    self.model_complete_path,
                    use_gpu=self.use_gpu,
                    n_channels=len(self.required_channels),
                    scale=self.scale,
                )

            self.queue.put({"status": "Segmenting…"})

            # Wrapper for single-position compatibility if batch_structure is missing
            if not hasattr(self, "batch_structure"):
                self.batch_structure = {
                    0: {"well_name": "Batch", "positions": self.positions}
                }

            self.t0_well = time.time()
            # Loop over Wells
            for w_i, (w_idx, well_data) in enumerate(self.batch_structure.items()):
                positions = well_data["positions"]

                # Well Time Estimation
                elapsed = time.time() - self.t0_well
                if w_i > 0:
                    avg_well = elapsed / w_i
                    rem_well = (len(self.batch_structure) - w_i) * avg_well
                    mins_w = int(rem_well // 60)
                    secs_w = int(rem_well % 60)
                    well_str = f"Well {w_i + 1}/{len(self.batch_structure)} - {mins_w} m {secs_w} s left"
                else:
                    well_str = (
                        f"Processing well {w_i + 1}/{len(self.batch_structure)}..."
                    )

                # Update Well Progress
                self.queue.put(
                    {
                        "well_progress": (w_i / len(self.batch_structure)) * 100,
                        "well_time": well_str,
                    }
                )

                self.t0_pos = time.time()
                # Loop over positions in this well
                for pos_idx, pos_path in enumerate(positions):

                    # Setup specific variables for this position (folders, length, etc.)
                    self.setup_for_position(pos_path)

                    list_indices = range(self.len_movie)
                    if self.flip:
                        list_indices = reversed(list_indices)

                    # Position Time Estimation relative to current well
                    elapsed_pos = time.time() - self.t0_pos
                    if pos_idx > 0:
                        avg_pos = elapsed_pos / pos_idx
                        rem_pos = (len(positions) - pos_idx) * avg_pos
                        mins_p = int(rem_pos // 60)
                        secs_p = int(rem_pos % 60)
                        pos_str = f"Pos {pos_idx + 1}/{len(positions)} - {mins_p} m {secs_p} s left"
                    else:
                        pos_str = (
                            f"Processing position {pos_idx + 1}/{len(positions)}..."
                        )

                    self.process_position(model=model, scale_model=scale_model)

                    # Verify every frame was written and atomically swap the
                    # temporary masks onto the final folder. Raises if any frame
                    # is missing, which aborts the run (caught below).
                    self.finalize_position()

                    # End of position loop
                    self.queue.put(
                        {"pos_progress": ((pos_idx + 1) / len(positions)) * 100}
                    )

                # End of Well loop
                self.queue.put(
                    {"well_progress": ((w_i + 1) / len(self.batch_structure)) * 100}
                )

        except Exception as e:
            # Fail loudly: report the error to the GUI/orchestrator and do NOT
            # emit "finished" (which would let tracking run on partial masks).
            logger.error(f"Segmentation failed: {e}", exc_info=True)
            try:
                del model
            except NameError:
                pass
            gc.collect()
            self.queue.put({"status": "error", "message": str(e)})
            self.queue.close()
            return

        try:
            del model
        except NameError:
            pass

        gc.collect()
        logger.info("Segmentation task is done.")

        # Send end signal
        self.queue.put("finished")
        self.queue.close()


class SegmentCellThresholdProcess(BaseSegmentProcess):

    def __init__(self, *args, **kwargs):
        """
        Initialize the process.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """

        super().__init__(*args, **kwargs)

        self.equalize = False

        self.sum_done = 0
        self.t0 = time.time()

    def prepare_equalize(self):
        """Prepare the equalization reference."""

        for i in range(len(self.instructions)):

            if self.equalize[i]:
                f_reference = load_frames(
                    self.img_num_channels[:, self.equalize_time[i]],
                    self.file,
                    scale=None,
                    normalize_input=False,
                )
                f_reference = f_reference[:, :, self.instructions[i]["target_channel"]]
            else:
                f_reference = None

            self.instructions[i].update({"equalize_reference": f_reference})

    def load_threshold_config(self):
        """Load the threshold configuration."""

        self.instructions = []
        for inst in self.threshold_instructions:
            if os.path.exists(inst):
                with open(inst, "r") as f:
                    self.instructions.append(json.load(f))
            else:
                logger.error("The configuration path is not valid. Abort.")
                self.abort_process()

    def extract_threshold_parameters(self):
        """Extract the threshold parameters."""

        self.required_channels = []
        self.equalize = []
        self.equalize_time = []

        for i in range(len(self.instructions)):
            ch = [self.instructions[i]["target_channel"]]
            self.required_channels.append(ch)

            if "equalize_reference" in self.instructions[i]:
                equalize, equalize_time = self.instructions[i]["equalize_reference"]
                self.equalize.append(equalize)
                self.equalize_time.append(equalize_time)

    def write_log(self):
        """Write the logo to the log file."""

        log = f"Threshold segmentation: {self.threshold_instructions}\n"
        with open(self.pos + f"log_{self.mode}.txt", "a") as f:
            f.write(f"{datetime.datetime.now()} SEGMENT \n")
            f.write(log)

    def detect_channels(self):
        """Detect the channels required for the thresholding."""

        for i in range(len(self.instructions)):

            self.channel_indices = _extract_channel_indices_from_config(
                self.config, self.required_channels[i]
            )
            logger.info(
                f"Required channels: {self.required_channels[i]} located at channel indices {self.channel_indices}."
            )
            self.instructions[i].update({"target_channel": self.channel_indices[0]})
            self.instructions[i].update({"channel_names": self.channel_names})

        self.img_num_channels = _get_img_num_per_channel(
            np.arange(self.nbr_channels), self.len_movie, self.nbr_channels
        )

    def parallel_job(self, indices: List[int]) -> None:
        """
        Run the parallel segmentation job.

        Parameters
        ----------
        indices : list
            The list of indices to process.
        """

        # Exceptions are intentionally NOT swallowed here: they propagate to the
        # ThreadPoolExecutor result iteration in process_position, which re-raises
        # so the whole run can fail loudly instead of silently writing fewer
        # masks than there are frames.
        from celldetective.segmentation import (
            segment_frame_from_thresholds,
            merge_instance_segmentation,
        )

        for t in tqdm(
            indices, desc="frame"
        ):  # for t in tqdm(range(self.len_movie),desc="frame"):

            # Load channels at time t
            masks = []
            for i in range(len(self.instructions)):
                f = load_frames(
                    self.img_num_channels[:, t],
                    self.file,
                    scale=None,
                    normalize_input=False,
                )

                mask = segment_frame_from_thresholds(f, **self.instructions[i])
                # print(f'Frame {t}; segment with {self.instructions[i]=}...')
                masks.append(mask)

            if len(self.instructions) > 1:
                mask = merge_instance_segmentation(masks, mode="OR")

            from celldetective.utils.io import save_tiff_imagej_compatible

            save_tiff_imagej_compatible(
                os.sep.join(
                    [self.pos, self.label_folder, f"{str(t).zfill(4)}.tif"]
                ),
                mask.astype(np.uint16),
                axes="YX",
            )

            # Send signal for progress bar
            self.sum_done += 1 / self.len_movie * 100

            # Triple progress bar logic
            data = {}
            data["frame_progress"] = self.sum_done

            # Frame time estimation
            elapsed = time.time() - getattr(self, "t0_frame", time.time())
            measured_count = int((self.sum_done / 100) * self.len_movie)

            if measured_count > 0:
                avg = elapsed / measured_count
                rem = self.len_movie - measured_count
                if rem < 0:
                    rem = 0
                rem_t = rem * avg
                mins = int(rem_t // 60)
                secs = int(rem_t % 60)
                data["frame_time"] = f"Segmentation: {mins} m {secs} s"
            else:
                data["frame_time"] = f"Segmentation..."

            # Saturate preview: Convert labels to binary (0/1)
            data["image_preview"] = (mask > 0).astype(np.uint8)
            self.queue.put(data)

            del f
            del mask
            gc.collect()

        return

    def process_position(self):
        """Process a single position."""

        tprint("Segment")

        # Re-initialize threshold specific stuff (depends on channel indices which depend on metadata)
        self.load_threshold_config()
        self.extract_threshold_parameters()
        self.detect_channels()
        self.prepare_equalize()
        self.write_log()  # Log start of segmentation for this pos

        self.indices = list(range(self.img_num_channels.shape[1]))
        if self.flip:
            self.indices = np.array(list(reversed(self.indices)))

        chunks = np.array_split(self.indices, self.n_threads)

        self.t0_frame = time.time()  # Reset timer for accurate frame timing
        self.sum_done = 0  # Reset progress for this pos

        # Cap the BLAS/OpenMP thread pool to one thread per worker for the
        # duration of the parallel region. Each of the n_threads workers runs
        # numpy/skimage on its own frame chunk; left uncapped, every worker
        # would also spawn a full BLAS pool, oversubscribing the CPU (n_threads
        # x n_cores threads) and slowing things down. _blas_thread_limit() is a
        # no-op context manager when threadpoolctl is unavailable.
        with _blas_thread_limit(1):
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.n_threads
            ) as executor:
                results = executor.map(
                    self.parallel_job, chunks
                )  # list(map(lambda x: executor.submit(self.parallel_job, x), chunks))
                try:
                    for i, return_value in enumerate(results):
                        pass
                except Exception as e:
                    logger.error(f"Exception: {e}")
                    raise

    def run(self):
        """Run the segmentation process."""

        # The child must re-import the celldetective module tree (Windows spawn)
        # before the first frame is ready; signal that work has begun so the
        # bars don't look frozen at 0%.
        self.queue.put({"status": "Segmenting…"})

        # Wrapper for single-position compatibility if batch_structure is missing
        if not hasattr(self, "batch_structure"):
            self.batch_structure = {
                0: {"well_name": "Batch", "positions": self.positions}
            }

        try:
            self.t0_well = time.time()
            # Loop over Wells
            for w_i, (w_idx, well_data) in enumerate(self.batch_structure.items()):
                positions = well_data["positions"]

                # Well Time Estimation
                elapsed = time.time() - self.t0_well
                if w_i > 0:
                    avg_well = elapsed / w_i
                    rem_well = (len(self.batch_structure) - w_i) * avg_well
                    mins_w = int(rem_well // 60)
                    secs_w = int(rem_well % 60)
                    well_str = f"Well {w_i + 1}/{len(self.batch_structure)} - {mins_w} m {secs_w} s left"
                else:
                    well_str = f"Processing well {w_i + 1}/{len(self.batch_structure)}..."

                # Update Well Progress
                self.queue.put(
                    {
                        "well_progress": (w_i / len(self.batch_structure)) * 100,
                        "well_time": well_str,
                    }
                )

                self.t0_pos = time.time()
                # Loop over positions in this well
                for pos_idx, pos_path in enumerate(positions):

                    # Setup specific variables for this position
                    self.setup_for_position(pos_path)

                    self.process_position()

                    # Verify completeness and atomically swap the temp masks in.
                    self.finalize_position()

                    # End of position loop
                    self.queue.put(
                        {"pos_progress": ((pos_idx + 1) / len(positions)) * 100}
                    )

                # End of Well loop
                self.queue.put(
                    {"well_progress": ((w_i + 1) / len(self.batch_structure)) * 100}
                )

        except Exception as e:
            # Fail loudly rather than emitting "finished" on a partial result.
            logger.error(f"Threshold segmentation failed: {e}", exc_info=True)
            self.queue.put({"status": "error", "message": str(e)})
            self.queue.close()
            return

        logger.info("Done.")
        # Send end signal
        self.queue.put("finished")
        self.queue.close()
