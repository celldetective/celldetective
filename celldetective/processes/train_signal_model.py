from multiprocessing import Process
import time
import os
import json
from glob import glob
import numpy as np
from art import tprint
from tensorflow.python.keras.callbacks import Callback

from celldetective.signals import SignalDetectionModel
from celldetective.log_manager import get_logger
from celldetective.utils.model_loaders import locate_signal_model

logger = get_logger(__name__)


class ProgressCallback(Callback):

    def __init__(self, queue=None, total_epochs=100, stop_event=None):
        super().__init__()
        self.queue = queue
        self.total_epochs = total_epochs
        self.current_step = 0
        self.t0 = time.time()
        self.stop_event = stop_event

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        # Update frame bar (bottom bar) for batch progress
        # logs has 'size' and 'batch'
        # We need total batches. Keras doesn't always pass it easily in logs unless we know steps_per_epoch.
        # But self.params['steps'] should have it if available.
        if self.params and "steps" in self.params:
            total_steps = self.params["steps"]
            batch_progress = ((batch + 1) / total_steps) * 100
            if self.queue is not None:
                # Send generic batch update (frequent)
                self.queue.put(
                    {
                        "frame_progress": batch_progress,
                        "frame_time": f"Batch {batch + 1}/{total_steps}",
                    }
                )

    def on_epoch_end(self, epoch, logs=None):
        if self.stop_event and self.stop_event.is_set():
            logger.info("Interrupting training...")
            self.model.stop_training = True
            self.stop_event.clear()

        self.current_step += 1
        # Send signal for progress bar
        sum_done = (self.current_step) / self.total_epochs * 100
        mean_exec_per_step = (time.time() - self.t0) / (self.current_step)
        pred_time = (self.total_epochs - self.current_step) * mean_exec_per_step

        # Format time string
        if pred_time > 60:
            time_str = f"{pred_time/60:.1f} min"
        else:
            time_str = f"{pred_time:.1f} s"

        if self.queue is not None:
            # Update Position bar (middle) for Epoch progress
            msg = {
                "pos_progress": sum_done,
                "pos_time": f"Epoch {self.current_step}/{self.total_epochs} (ETA: {time_str})",
                "frame_progress": 0,  # Reset batch bar
                "frame_time": "Batch 0/0",
            }
            # Attempt to extract metrics for plotting
            if logs:
                # Infer model type
                if "iou" in logs:
                    model_name = "Classifier"
                else:
                    model_name = "Regressor"

                # Send all scalar metrics
                msg["plot_data"] = {
                    "epoch": epoch + 1,  # 1-based for plot
                    "metrics": {
                        k: float(v) for k, v in logs.items() if not k.startswith("val_")
                    },
                    "val_metrics": {
                        k: float(v) for k, v in logs.items() if k.startswith("val_")
                    },
                    "model_name": model_name,
                    "total_epochs": self.params.get("epochs", self.total_epochs),
                }
            self.queue.put(msg)

    def on_training_result(self, result):
        if self.queue is not None:
            self.queue.put({"training_result": result})


class TrainSignalModelProcess(Process):

    def __init__(self, queue=None, process_args=None, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.queue = queue

        if process_args is not None:
            for key, value in process_args.items():
                setattr(self, key, value)

        tprint("Train event detection")
        self.read_instructions()
        self.extract_training_params()

        self.sum_done = 0
        self.t0 = time.time()

    def read_instructions(self):

        if os.path.exists(self.instructions):
            with open(self.instructions, "r") as f:
                self.training_instructions = json.load(f)
        else:
            logger.error("Training instructions could not be found. Abort.")
            self.abort_process()

        all_classes = []
        for d in self.training_instructions["ds"]:
            datasets = glob(d + os.sep + "*.npy")
            for dd in datasets:
                data = np.load(dd, allow_pickle=True)
                classes = np.unique([ddd["class"] for ddd in data])
                all_classes.extend(classes)
        all_classes = np.unique(all_classes)
        n_classes = len(all_classes)

        self.model_params = {
            k: self.training_instructions[k]
            for k in (
                "pretrained",
                "model_signal_length",
                "channel_option",
                "n_channels",
                "label",
            )
            if k in self.training_instructions
        }
        self.model_params.update({"n_classes": n_classes})
        self.train_params = {
            k: self.training_instructions[k]
            for k in (
                "model_name",
                "target_directory",
                "channel_option",
                "recompile_pretrained",
                "test_split",
                "augment",
                "epochs",
                "learning_rate",
                "batch_size",
                "validation_split",
                "normalization_percentile",
                "normalization_values",
                "normalization_clip",
            )
            if k in self.training_instructions
        }

    def neighborhood_postprocessing(self):

        # if neighborhood of interest in training instructions, write it in config!
        if "neighborhood_of_interest" in self.training_instructions:
            if self.training_instructions["neighborhood_of_interest"] is not None:

                model_path = locate_signal_model(
                    self.training_instructions["model_name"], path=None, pairs=True
                )
                complete_path = model_path  # +model
                complete_path = rf"{complete_path}"
                model_config_path = os.sep.join([complete_path, "config_input.json"])
                model_config_path = rf"{model_config_path}"

                f = open(model_config_path)
                config = json.load(f)
                config.update(
                    {
                        "neighborhood_of_interest": self.training_instructions[
                            "neighborhood_of_interest"
                        ],
                        "reference_population": self.training_instructions[
                            "reference_population"
                        ],
                        "neighbor_population": self.training_instructions[
                            "neighbor_population"
                        ],
                    }
                )
                json_string = json.dumps(config)
                with open(model_config_path, "w") as outfile:
                    outfile.write(json_string)

    def run(self):
        self.queue.put({"status": "Loading datasets..."})
        model = SignalDetectionModel(**self.model_params)

        total_epochs = self.train_params["epochs"] * 3
        cb = ProgressCallback(
            queue=self.queue,
            total_epochs=total_epochs,
            stop_event=getattr(self, "stop_event", None),
        )

        model.fit_from_directory(
            self.training_instructions["ds"], callbacks=[cb], **self.train_params
        )

        # Send results to GUI
        if hasattr(model, "dico"):
            result_keys = [
                "val_confusion",
                "test_confusion",
                "val_predictions",
                "val_ground_truth",
                "test_predictions",
                "test_ground_truth",
                "val_mse",
            ]
            results = {k: model.dico[k] for k in result_keys if k in model.dico}
            # Only send if we have something relevant
            if results:
                self.queue.put({"training_result": results})

        self.neighborhood_postprocessing()
        self.queue.put("finished")
        self.queue.close()

    def extract_training_params(self):

        self.training_instructions.update(
            {"n_channels": len(self.training_instructions["channel_option"])}
        )
        self.model_params["n_channels"] = self.training_instructions["n_channels"]
        if self.training_instructions["augmentation_factor"] > 1.0:
            self.training_instructions.update({"augment": True})
        else:
            self.training_instructions.update({"augment": False})
        self.training_instructions.update({"test_split": 0.0})

    def end_process(self):

        # self.terminate()

        # if self.model_type=="stardist_utils":
        # 	from stardist_utils.models import StarDist2D
        # 	self.model = StarDist2D(None, name=self.model_name, basedir=self.target_directory)
        # 	self.model.optimize_thresholds(self.X_val,self.Y_val)

        self.terminate()
        self.queue.put("finished")

    def abort_process(self):

        self.terminate()
        self.queue.put("error")
