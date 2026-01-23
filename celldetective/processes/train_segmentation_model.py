from distutils.dir_util import copy_tree
from multiprocessing import Process
import time
import os
import shutil
from glob import glob
import json
import logging
import re

from tensorflow.python.keras.callbacks import Callback
from tqdm import tqdm
import numpy as np
import random

from celldetective.utils.image_augmenters import augmenter
from celldetective.utils.image_loaders import load_image_dataset
from celldetective.utils.image_cleaning import interpolate_nan
from celldetective.utils.normalization import normalize_multichannel
from celldetective.utils.mask_cleaning import fill_label_holes
from art import tprint
from csbdeep.utils import save_json
from celldetective import get_logger

logger = get_logger()


class ProgressCallback(Callback):

    def __init__(self, queue=None, epochs=100, stop_event=None):
        super().__init__()
        self.queue = queue
        self.epochs = epochs
        self.stop_event = stop_event
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):

        if self.stop_event and self.stop_event.is_set():
            self.model.stop_training = True
            return

        if logs is None:
            logs = {}

        # Send signal for progress bar
        sum_done = (epoch + 1) / self.epochs * 100
        mean_exec_per_step = (time.time() - self.t0) / (epoch + 1)
        pred_time = (self.epochs - (epoch + 1)) * mean_exec_per_step
        if self.queue is not None:
            self.queue.put([sum_done, pred_time])

            # Plot update
            metrics = {k: v for k, v in logs.items() if not k.startswith("val_")}
            val_metrics = {k: v for k, v in logs.items() if k.startswith("val_")}

            plot_data = {
                "epoch": epoch,
                "metrics": metrics,
                "val_metrics": val_metrics,
                "model_name": "StarDist",
                "total_epochs": self.epochs,
            }
            self.queue.put({"plot_data": plot_data})


class QueueLoggingHandler(logging.Handler):
    def __init__(self, queue, total_epochs, stop_event=None):
        super().__init__()
        self.queue = queue
        self.total_epochs = total_epochs
        self.stop_event = stop_event
        self.epoch_pattern = re.compile(
            r"Epoch (\d+), Time .*, Loss ([\d\.eE\-\+naninf]+)(?:, Loss Test ([\d\.eE\-\+naninf]+))?",
            re.IGNORECASE,
        )
        self.t0 = time.time()

    def emit(self, record):
        if self.stop_event and self.stop_event.is_set():
            # Can't easily stop cellpose_utils loop from here without raising exception or hacking
            # raising exception might be safest to exit training loop
            raise InterruptedError("Training interrupted")

        log_entry = self.format(record)
        match = self.epoch_pattern.search(log_entry)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            val_loss = float(match.group(3)) if match.group(3) else None

            sum_done = (epoch + 1) / self.total_epochs * 100
            mean_exec_per_step = (time.time() - self.t0) / (epoch + 1)
            pred_time = (self.total_epochs - (epoch + 1)) * mean_exec_per_step

            self.queue.put([sum_done, pred_time])

            metrics = {"loss": loss}
            val_metrics = {}
            if val_loss is not None:
                val_metrics["val_loss"] = val_loss

            plot_data = {
                "epoch": epoch,
                "metrics": metrics,
                "val_metrics": val_metrics,
                "model_name": "Cellpose",
                "total_epochs": self.total_epochs,
            }
            self.queue.put({"plot_data": plot_data})


class TrainSegModelProcess(Process):

    def __init__(self, queue=None, process_args=None, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.queue = queue

        if process_args is not None:
            for key, value in process_args.items():
                setattr(self, key, value)

        tprint("Train segmentation")
        self.read_instructions()
        self.extract_training_params()
        self.load_dataset()
        self.split_test_train()

        self.sum_done = 0
        self.t0 = time.time()

    def read_instructions(self):

        if os.path.exists(self.instructions):
            with open(self.instructions, "r") as f:
                self.training_instructions = json.load(f)
        else:
            logger.error("Training instructions could not be found. Abort.")
            self.abort_process()

    def run(self):

        self.queue.put("Loading dataset...")

        if self.model_type == "cellpose":
            self.train_cellpose_model()
        elif self.model_type == "stardist":
            self.train_stardist_model()

        self.queue.put("finished")
        self.queue.close()

    def train_stardist_model(self):

        from stardist import calculate_extents, gputools_available
        from stardist.models import Config2D, StarDist2D

        n_rays = 32
        logger.info(gputools_available())

        n_channel = self.X_trn[0].shape[-1]

        # Predict on subsampled grid for increased efficiency and larger field of view
        grid = (2, 2)
        conf = Config2D(
            n_rays=n_rays,
            grid=grid,
            use_gpu=self.use_gpu,
            n_channel_in=n_channel,
            train_learning_rate=self.learning_rate,
            train_patch_size=(256, 256),
            train_epochs=self.epochs,
            train_reduce_lr={"factor": 0.1, "patience": 30, "min_delta": 0},
            train_batch_size=self.batch_size,
            train_steps_per_epoch=int(self.augmentation_factor * len(self.X_trn)),
        )

        if self.use_gpu:
            from csbdeep.utils.tf import limit_gpu_memory

            limit_gpu_memory(None, allow_growth=True)

        if self.pretrained is None:
            model = StarDist2D(
                conf, name=self.model_name, basedir=self.target_directory
            )
        else:
            os.rename(
                self.instructions,
                os.sep.join([self.target_directory, self.model_name, "temp.json"]),
            )
            copy_tree(
                self.pretrained, os.sep.join([self.target_directory, self.model_name])
            )

            if os.path.exists(
                os.sep.join(
                    [
                        self.target_directory,
                        self.model_name,
                        "training_instructions.json",
                    ]
                )
            ):
                os.remove(
                    os.sep.join(
                        [
                            self.target_directory,
                            self.model_name,
                            "training_instructions.json",
                        ]
                    )
                )
            if os.path.exists(
                os.sep.join(
                    [self.target_directory, self.model_name, "config_input.json"]
                )
            ):
                os.remove(
                    os.sep.join(
                        [self.target_directory, self.model_name, "config_input.json"]
                    )
                )
            if os.path.exists(
                os.sep.join([self.target_directory, self.model_name, "logs" + os.sep])
            ):
                shutil.rmtree(
                    os.sep.join([self.target_directory, self.model_name, "logs"])
                )
            os.rename(
                os.sep.join([self.target_directory, self.model_name, "temp.json"]),
                os.sep.join(
                    [
                        self.target_directory,
                        self.model_name,
                        "training_instructions.json",
                    ]
                ),
            )

            # shutil.copytree(pretrained, os.sep.join([target_directory, model_name]))
            model = StarDist2D(
                None, name=self.model_name, basedir=self.target_directory
            )
            model.config.train_epochs = self.epochs
            model.config.train_batch_size = min(len(self.X_trn), self.batch_size)
            model.config.train_learning_rate = (
                self.learning_rate
            )  # perf seems bad if lr is changed in transfer
            model.config.use_gpu = self.use_gpu
            model.config.train_reduce_lr = {
                "factor": 0.1,
                "patience": 10,
                "min_delta": 0,
            }
            logger.info(f"{model.config=}")

            save_json(
                vars(model.config),
                os.sep.join([self.target_directory, self.model_name, "config.json"]),
            )

        if self.pretrained is not None:
            logger.info("Freezing encoder layers for StarDist model...")
            mod = model.keras_model
            encoder_depth = len(mod.layers) // 2

            for layer in mod.layers[:encoder_depth]:
                layer.trainable = False

            # Keep decoder trainable
            for layer in mod.layers[encoder_depth:]:
                layer.trainable = True

        median_size = calculate_extents(list(self.Y_trn), np.mean)
        fov = np.array(model._axes_tile_overlap("YX"))
        logger.info(f"median object size:      {median_size}")
        logger.info(f"network field of view :  {fov}")
        if any(median_size > fov):
            logger.warning(
                "WARNING: median object size larger than field of view of the neural network."
            )

        import sys

        class StreamToQueue:
            def __init__(self, queue, total_epochs, original_stream, stop_event=None):
                self.queue = queue
                self.total_epochs = total_epochs
                self.original_stream = original_stream
                self.stop_event = stop_event
                self.epoch_pattern = re.compile(r"Epoch (\d+)/(\d+)")
                # Generic pattern to capture "key: value" pairs
                self.metric_pattern = re.compile(
                    r"([\w_]+)\s*:\s*([\d\.eE\-\+naninf]+)"
                )
                self.current_epoch = 0
                self.t0 = time.time()
                self.buffer = ""

            def write(self, message):
                if self.stop_event and self.stop_event.is_set():
                    raise InterruptedError("Training interrupted by user")

                self.original_stream.write(message)
                self.original_stream.flush()  # Ensure immediate display
                self.buffer += message
                if "\n" in message or "\r" in message:
                    self._parse_buffer()

            def flush(self):
                self.original_stream.flush()

            def _parse_buffer(self):
                lines = re.split(r"[\r\n]+", self.buffer)
                # Keep the last incomplete part in buffer
                if not (self.buffer.endswith("\n") or self.buffer.endswith("\r")):
                    self.buffer = lines[-1]
                    lines = lines[:-1]
                else:
                    self.buffer = ""

                for line in lines:
                    if not line.strip():
                        continue

                    # Check for Epoch
                    m_epoch = self.epoch_pattern.search(line)
                    if m_epoch:
                        self.current_epoch = int(m_epoch.group(1))
                        # Put progress?
                        sum_done = (self.current_epoch - 1) / self.total_epochs * 100
                        self.queue.put(
                            [sum_done, 0]
                        )  # Time estimation handled by GUI or ignored
                        continue

                    # Capture all metrics in the line
                    found_metrics = self.metric_pattern.findall(line)
                    if found_metrics:
                        metrics = {}
                        val_metrics = {}

                        for key, val_str in found_metrics:
                            try:
                                val = float(val_str)
                                if key.startswith("val_"):
                                    val_metrics[key] = val
                                else:
                                    metrics[key] = val
                            except ValueError:
                                pass

                        # Only send plot data if we have validation metrics (indicates end of epoch)
                        if metrics and val_metrics:
                            plot_data = {
                                "epoch": self.current_epoch,
                                "metrics": metrics,
                                "val_metrics": val_metrics,
                                "model_name": "StarDist",
                                "total_epochs": self.total_epochs,
                            }
                            self.queue.put({"plot_data": plot_data})

        # Redirect stdout/stderr to capture Keras output
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        stream_parser = StreamToQueue(
            self.queue,
            self.epochs,
            original_stdout,
            stop_event=self.stop_event if hasattr(self, "stop_event") else None,
        )
        sys.stdout = stream_parser
        sys.stderr = stream_parser  # Keras often prints to stderr

        try:
            if self.augmentation_factor == 1.0:
                model.train(
                    self.X_trn,
                    self.Y_trn,
                    validation_data=(self.X_val, self.Y_val),
                    epochs=self.epochs,
                )
            else:
                model.train(
                    self.X_trn,
                    self.Y_trn,
                    validation_data=(self.X_val, self.Y_val),
                    augmenter=augmenter,
                    epochs=self.epochs,
                )
        except Exception as e:
            logger.error(f"Error in StarDist training: {e}")
            raise e
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

        model.optimize_thresholds(self.X_val, self.Y_val)

        if isinstance(median_size, (list, np.ndarray)):
            median_size_scalar = np.mean(median_size)
        else:
            median_size_scalar = median_size

        config_inputs = {
            "channels": self.target_channels,
            "normalization_percentile": self.normalization_percentile,
            "normalization_clip": self.normalization_clip,
            "normalization_values": self.normalization_values,
            "model_type": "stardist",
            "spatial_calibration": self.spatial_calibration,
            "cell_size_um": float(median_size_scalar * self.spatial_calibration),
            "dataset": {"train": self.files_train, "validation": self.files_val},
        }

        def make_json_safe(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            return str(obj)

        json_input_config = json.dumps(config_inputs, indent=4, default=make_json_safe)
        with open(
            os.sep.join([self.target_directory, self.model_name, "config_input.json"]),
            "w",
        ) as outfile:
            outfile.write(json_input_config)

    def train_cellpose_model(self):

        # do augmentation in place
        X_aug = []
        Y_aug = []
        n_val = max(1, int(round(self.augmentation_factor * len(self.X_trn))))
        indices = random.choices(list(np.arange(len(self.X_trn))), k=n_val)
        logger.info("Performing image augmentation pre-training...")
        for i in tqdm(indices):
            x_aug, y_aug = augmenter(self.X_trn[i], self.Y_trn[i])
            X_aug.append(x_aug)
            Y_aug.append(y_aug)

        # Channel axis in front for cellpose_utils
        X_aug = [np.moveaxis(x, -1, 0) for x in X_aug]
        self.X_val = [np.moveaxis(x, -1, 0) for x in self.X_val]
        logger.info("number of augmented images: %3d" % len(X_aug))

        from cellpose.models import CellposeModel
        from cellpose.io import logger_setup
        import torch

        if not self.use_gpu:
            logger.info("Using CPU for training...")
            device = torch.device("cpu")
        else:
            logger.info("Using GPU for training...")

        # logger_setup configures console and file handlers for cellpose_utils
        _, log_file = logger_setup()

        # Get cellpose_utils logger explicitly to ensure we catch all cellpose_utils logs (e.g. from models)
        logger_cellpose = logging.getLogger("cellpose")

        # Add custom handler
        handler = QueueLoggingHandler(
            self.queue,
            self.epochs,
            stop_event=self.stop_event if hasattr(self, "stop_event") else None,
        )
        handler.setLevel(logging.INFO)
        logger_cellpose.addHandler(handler)

        try:
            logger.info(f"Pretrained model: {self.pretrained}")
            if self.pretrained is not None:
                pretrained_path = os.sep.join(
                    [self.pretrained, os.path.split(self.pretrained)[-1]]
                )
            else:
                pretrained_path = self.pretrained

            model = CellposeModel(
                gpu=self.use_gpu,
                model_type=None,
                pretrained_model=pretrained_path,
                diam_mean=30.0,
                nchan=X_aug[0].shape[0],
            )

            if self.pretrained is not None:
                logger.info("Freezing encoder layers for Cellpose model...")
                for param in model.net.downsample.parameters():
                    param.requires_grad = False

                # Optional: freeze style branch
                for param in model.net.make_style.parameters():
                    param.requires_grad = False

                # Keep decoder trainable
                for param in model.net.upsample.parameters():
                    param.requires_grad = True

                # Keep output head trainable
                for param in model.net.output.parameters():
                    param.requires_grad = True

                # Unfreeze all output heads (version-safe)
                output_heads = ["output", "output_conv", "flow", "prob"]
                for head_name in output_heads:
                    if hasattr(model.net, head_name):
                        for param in getattr(model.net, head_name).parameters():
                            param.requires_grad = True

            model.train(
                train_data=X_aug,
                train_labels=Y_aug,
                normalize=False,
                channels=None,
                batch_size=self.batch_size,
                min_train_masks=1,
                save_path=self.target_directory + os.sep + self.model_name,
                n_epochs=self.epochs,
                model_name=self.model_name,
                learning_rate=self.learning_rate,
                test_data=self.X_val,
                test_labels=self.Y_val,
            )
        except InterruptedError:
            logger.info("Training interrupted.")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise e
        finally:
            logger_cellpose.removeHandler(handler)

        file_to_move = glob(
            os.sep.join([self.target_directory, self.model_name, "models", "*"])
        )[0]
        shutil.move(
            file_to_move,
            os.sep.join([self.target_directory, self.model_name, ""])
            + os.path.split(file_to_move)[-1],
        )
        os.rmdir(os.sep.join([self.target_directory, self.model_name, "models"]))

        diameter = model.diam_labels

        if (
            self.pretrained is not None
            and os.path.split(self.pretrained)[-1] == "CP_nuclei"
        ):
            standard_diameter = 17.0
        else:
            standard_diameter = 30.0

        input_spatial_calibration = (
            self.spatial_calibration
        )  # *diameter / standard_diameter

        config_inputs = {
            "channels": self.target_channels,
            "diameter": standard_diameter,
            "cellprob_threshold": 0.0,
            "flow_threshold": 0.4,
            "normalization_percentile": self.normalization_percentile,
            "normalization_clip": self.normalization_clip,
            "normalization_values": self.normalization_values,
            "model_type": "cellpose",
            "spatial_calibration": input_spatial_calibration,
            "cell_size_um": round(diameter * input_spatial_calibration, 4),
            "dataset": {"train": self.files_train, "validation": self.files_val},
        }

        def make_json_safe(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            return str(obj)

        json_input_config = json.dumps(config_inputs, indent=4, default=make_json_safe)
        with open(
            os.sep.join([self.target_directory, self.model_name, "config_input.json"]),
            "w",
        ) as outfile:
            outfile.write(json_input_config)

    def split_test_train(self):

        if not len(self.X) > 1:
            logger.error("Not enough training data")
            self.abort_process()

        rng = np.random.RandomState()
        ind = rng.permutation(len(self.X))
        n_val = max(1, int(round(self.validation_split * len(ind))))
        ind_train, ind_val = ind[:-n_val], ind[-n_val:]
        self.X_val, self.Y_val = [self.X[i] for i in ind_val], [
            self.Y[i] for i in ind_val
        ]
        self.X_trn, self.Y_trn = [self.X[i] for i in ind_train], [
            self.Y[i] for i in ind_train
        ]

        self.files_train = [self.filenames[i] for i in ind_train]
        self.files_val = [self.filenames[i] for i in ind_val]

        logger.info("number of images: %3d" % len(self.X))
        logger.info("- training:       %3d" % len(self.X_trn))
        logger.info("- validation:     %3d" % len(self.X_val))

    def extract_training_params(self):

        self.model_name = self.training_instructions["model_name"]
        self.target_directory = self.training_instructions["target_directory"]
        self.model_type = self.training_instructions["model_type"]
        self.pretrained = self.training_instructions["pretrained"]

        self.datasets = self.training_instructions["ds"]

        self.target_channels = self.training_instructions["channel_option"]
        self.normalization_percentile = self.training_instructions[
            "normalization_percentile"
        ]
        self.normalization_clip = self.training_instructions["normalization_clip"]
        self.normalization_values = self.training_instructions["normalization_values"]
        self.spatial_calibration = self.training_instructions["spatial_calibration"]

        self.validation_split = self.training_instructions["validation_split"]
        self.augmentation_factor = self.training_instructions["augmentation_factor"]

        self.learning_rate = self.training_instructions["learning_rate"]
        self.epochs = self.training_instructions["epochs"]
        self.batch_size = self.training_instructions["batch_size"]

    def load_dataset(self):

        logger.info(f"Datasets: {self.datasets}")
        self.X, self.Y, self.filenames = load_image_dataset(
            self.datasets,
            self.target_channels,
            train_spatial_calibration=self.spatial_calibration,
            mask_suffix="labelled",
        )
        logger.info("Dataset loaded...")

        self.values = []
        self.percentiles = []
        for k in range(len(self.normalization_percentile)):
            if self.normalization_percentile[k]:
                self.percentiles.append(self.normalization_values[k])
                self.values.append(None)
            else:
                self.percentiles.append(None)
                self.values.append(self.normalization_values[k])

        self.X = [
            normalize_multichannel(
                x,
                **{
                    "percentiles": self.percentiles,
                    "values": self.values,
                    "clip": self.normalization_clip,
                },
            )
            for x in self.X
        ]

        for k in range(len(self.X)):
            x = self.X[k].copy()
            x_interp = np.moveaxis(
                [interpolate_nan(x[:, :, c].copy()) for c in range(x.shape[-1])], 0, -1
            )
            self.X[k] = x_interp

        self.Y = [fill_label_holes(y) for y in tqdm(self.Y)]

    def end_process(self):

        self.terminate()
        self.queue.put("finished")

    def abort_process(self):

        self.terminate()
        self.queue.put("error")
