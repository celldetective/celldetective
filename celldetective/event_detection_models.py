import json
import os
import random
import time
from glob import glob

import keras
import numpy as np
from scipy.ndimage import shift
from tensorflow.keras.callbacks import Callback

from matplotlib import pyplot as plt
from natsort import natsorted
from scipy.interpolate import interp1d
from sklearn.metrics import (
    jaccard_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    ModelCheckpoint,
    EarlyStopping,
    TensorBoard,
)
from tensorflow.keras.layers import (
    Activation,
    Add,
)
from tensorflow.keras.layers import (
    Input,
    ZeroPadding1D,
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Concatenate,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.models import Model

from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.config.experimental import (
    list_physical_devices,
    set_memory_growth,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, MeanIoU
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.losses import MeanSquaredError

from celldetective.utils.dataset_helpers import compute_weights, train_test_split
from celldetective.utils.plots.regression import regression_plot


def TimeHistory():
    """Create a TimeHistory callback instance."""
    cls = _get_time_history_class()
    return cls()


class SignalDetectionModel(object):
    """
    A class for creating and managing signal detection models for analyzing biological signals.

    This class provides functionalities to load a pretrained signal detection model or create one from scratch,
    preprocess input signals, train the model, and make predictions on new data.

    Parameters
    ----------
    path : str, optional
            Path to the directory containing the model and its configuration. This is used when loading a pretrained model.
    pretrained : str, optional
            Path to the pretrained model to load. If specified, the model and its configuration are loaded from this path.
    channel_option : list of str, optional
            Specifies the channels to be used for signal analysis. Default is ["live_nuclei_channel"].
    model_signal_length : int, optional
            The length of the input signals that the model expects. Default is 128.
    n_channels : int, optional
            The number of channels in the input signals. Default is 1.
    n_conv : int, optional
            The number of convolutional layers in the model. Default is 2.
    n_classes : int, optional
            The number of classes for the classification task. Default is 3.
    dense_collection : int, optional
            The number of units in the dense layer of the model. Default is 512.
    dropout_rate : float, optional
            The dropout rate applied to the dense layer of the model. Default is 0.1.
    label : str, optional
            A label for the model, used in naming and organizing outputs. Default is ''.

    Attributes
    ----------
    model_class : keras Model
            The classification model for predicting the class of signals.
    model_reg : keras Model
            The regression model for predicting the time of interest for signals.

    Methods
    -------
    load_pretrained_model()
            Loads the model and its configuration from the pretrained path.
    create_models_from_scratch()
            Creates new models for classification and regression from scratch.
    prep_gpu()
            Prepares GPU devices for training, if available.
    fit_from_directory(ds_folders, ...)
            Trains the model using data from specified directories.
    fit(x_train, y_time_train, y_class_train, ...)
            Trains the model using provided datasets.
    predict_class(x, ...)
            Predicts the class of input signals.
    predict_time_of_interest(x, ...)
            Predicts the time of interest for input signals.
    plot_model_history(mode)
            Plots the training history for the specified mode (classifier or regressor).
    evaluate_regression_model()
            Evaluates the regression model on test and validation data.
    gather_callbacks(mode)
            Gathers and prepares callbacks for training based on the specified mode.
    generate_sets()
            Generates training, validation, and test sets from loaded data.
    augment_training_set()
            Augments the training set with additional generated data.
    load_and_normalize(subset)
            Loads and normalizes signals from a subset of data.

    Notes
    -----
    - This class is designed to work with biological signal data, such as time series from microscopy imaging.
    - The model architecture and training configurations can be customized through the class parameters and methods.

    """

    def __init__(
        self,
        path=None,
        pretrained=None,
        channel_option=["live_nuclei_channel"],
        model_signal_length=128,
        n_channels=1,
        n_conv=2,
        n_classes=3,
        dense_collection=512,
        dropout_rate=0.1,
        label="",
    ):

        self.prep_gpu()

        self.model_signal_length = model_signal_length
        self.channel_option = channel_option
        self.pretrained = pretrained
        self.n_channels = n_channels
        self.n_conv = n_conv
        self.n_classes = n_classes
        self.dense_collection = dense_collection
        self.dropout_rate = dropout_rate
        self.label = label
        self.show_plots = True

        if self.pretrained is not None:
            print(f"Load pretrained models from {pretrained}...")
            test = self.load_pretrained_model()
            if test is None:
                self.pretrained = None
                print(
                    "Pretrained model could not be loaded. Check the log for error. Abort..."
                )
                return None
        else:
            print("Create models from scratch...")
            self.create_models_from_scratch()
            print("Models successfully created.")

    def load_pretrained_model(self):
        """
        Loads a pretrained model and its configuration from the specified path.

        This method attempts to load both the classification and regression models from the path specified during the
        class instantiation. It also loads the model configuration from a JSON file and updates the model attributes
        accordingly. If the models cannot be loaded, an error message is printed.

        Raises
        ------
        Exception
                If there is an error loading the model or the configuration file, an exception is raised with details.

        Notes
        -----
        - The models are expected to be saved in .h5 format with the filenames "classifier.h5" and "regressor.h5".
        - The configuration file is expected to be named "config_input.json" and located in the same directory as the models.

        """

        if self.pretrained.endswith(os.sep):
            self.pretrained = os.sep.join(self.pretrained.split(os.sep)[:-1])

        try:
            self.model_class = load_model(
                os.sep.join([self.pretrained, "classifier.h5"]),
                compile=False,
                custom_objects={"mse": MeanSquaredError()},
            )
            self.model_class.load_weights(
                os.sep.join([self.pretrained, "classifier.h5"])
            )
            self.model_class = self.freeze_encoder(self.model_class, 5)
            print("Classifier successfully loaded...")
        except Exception as e:
            print(f"Error {e}...")
            self.model_class = None
        try:
            self.model_reg = load_model(
                os.sep.join([self.pretrained, "regressor.h5"]),
                compile=False,
                custom_objects={"mse": MeanSquaredError()},
            )
            self.model_reg.load_weights(os.sep.join([self.pretrained, "regressor.h5"]))
            self.model_reg = self.freeze_encoder(self.model_reg, 5)
            print("Regressor successfully loaded...")
        except Exception as e:
            print(f"Error {e}...")
            self.model_reg = None

        if self.model_class is None and self.model_reg is None:
            return None

        # load config
        with open(os.sep.join([self.pretrained, "config_input.json"])) as config_file:
            model_config = json.load(config_file)
        self.config = model_config

        req_channels = model_config["channels"]
        print(f"Required channels read from pretrained model: {req_channels}")
        self.channel_option = req_channels
        if "normalize" in model_config:
            self.normalize = model_config["normalize"]
        if "normalization_percentile" in model_config:
            self.normalization_percentile = model_config["normalization_percentile"]
        if "normalization_values" in model_config:
            self.normalization_values = model_config["normalization_values"]
        if "normalization_clip" in model_config:
            self.normalization_clip = model_config["normalization_clip"]
        if "label" in model_config:
            self.label = model_config["label"]

        try:
            self.n_channels = self.model_class.layers[0].input_shape[0][-1]
            self.model_signal_length = self.model_class.layers[0].input_shape[0][-2]
            self.n_classes = self.model_class.layers[-1].output_shape[-1]
            model_class_input_shape = self.model_class.layers[0].input_shape[0]
            model_reg_input_shape = self.model_reg.layers[0].input_shape[0]
        except AttributeError:
            self.n_channels = self.model_class.input_shape[
                -1
            ]  # self.model_class.layers[0].input.shape[0][-1]
            self.model_signal_length = self.model_class.input_shape[
                -2
            ]  # self.model_class.layers[0].input[0].shape[0][-2]
            self.n_classes = self.model_class.output_shape[
                -1
            ]  # self.model_class.layers[-1].output[0].shape[-1]
            model_class_input_shape = self.model_class.input_shape
            model_reg_input_shape = self.model_reg.input_shape
        except Exception as e:
            print(e)

        assert (
            model_class_input_shape == model_reg_input_shape
        ), f"mismatch between input shape of classification: {self.model_class.layers[0].input_shape[0]} and regression {self.model_reg.layers[0].input_shape[0]} models... Error."

        return True

    def freeze_encoder(self, model, n_trainable_layers: int = 3):
        for layer in model.layers[
            : -min(n_trainable_layers, len(model.layers))
        ]:  # freeze everything except final Dense layer
            layer.trainable = False
        return model

    def create_models_from_scratch(self):
        """
        Initializes new models for classification and regression based on the specified parameters.

        This method creates new ResNet models for both classification and regression tasks using the parameters specified
        during class instantiation. The models are configured but not compiled or trained.

        Notes
        -----
        - The models are created using a custom ResNet architecture defined elsewhere in the codebase.
        - The models are stored in the `model_class` and `model_reg` attributes of the class.

        """

        self.model_class = ResNetModelCurrent(
            n_channels=self.n_channels,
            n_slices=self.n_conv,
            n_classes=3,
            dense_collection=self.dense_collection,
            dropout_rate=self.dropout_rate,
            header="classifier",
            model_signal_length=self.model_signal_length,
        )

        self.model_reg = ResNetModelCurrent(
            n_channels=self.n_channels,
            n_slices=self.n_conv,
            n_classes=self.n_classes,
            dense_collection=self.dense_collection,
            dropout_rate=self.dropout_rate,
            header="regressor",
            model_signal_length=self.model_signal_length,
        )

    def prep_gpu(self):
        """
        Prepares GPU devices for training by enabling memory growth.

        This method attempts to identify available GPU devices and configures TensorFlow to allow memory growth on each
        GPU. This prevents TensorFlow from allocating the total available memory on the GPU device upfront.

        Notes
        -----
        - This method should be called before any TensorFlow/Keras operations that might allocate GPU memory.
        - If no GPUs are detected, the method will pass silently.

        """

        try:

            physical_devices = list_physical_devices("GPU")
            for gpu in physical_devices:
                set_memory_growth(gpu, True)
        except Exception:
            pass

    def fit_from_directory(
        self,
        datasets,
        normalize=True,
        normalization_percentile=None,
        normalization_values=None,
        normalization_clip=None,
        channel_option=["live_nuclei_channel"],
        model_name=None,
        target_directory=None,
        augment=True,
        augmentation_factor=2,
        validation_split=0.20,
        test_split=0.0,
        batch_size=64,
        epochs=300,
        recompile_pretrained=False,
        learning_rate=0.01,
        loss_reg="mse",
        loss_class=None,
        show_plots=True,
        callbacks=None,
    ):
        """
        Trains the model using data from specified directories.

        This method prepares the dataset for training by loading and preprocessing data from specified directories,
        then trains the classification and regression models.

        Parameters
        ----------
        ds_folders : list of str
                List of directories containing the dataset files for training.
        callbacks : list, optional
                List of Keras callbacks to apply during training.
        normalize : bool, optional
                Whether to normalize the input signals (default is True).
        normalization_percentile : list or None, optional
                Percentiles for signal normalization (default is None).
        normalization_values : list or None, optional
                Specific values for signal normalization (default is None).
        normalization_clip : bool, optional
                Whether to clip the normalized signals (default is None).
        channel_option : list of str, optional
                Specifies the channels to be used for signal analysis (default is ["live_nuclei_channel"]).
        model_name : str, optional
                Name of the model for saving purposes (default is None).
        target_directory : str, optional
                Directory where the trained model and outputs will be saved (default is None).
        augment : bool, optional
                Whether to augment the training data (default is True).
        augmentation_factor : int, optional
                Factor by which to augment the training data (default is 2).
        validation_split : float, optional
                Fraction of the data to be used as validation set (default is 0.20).
        test_split : float, optional
                Fraction of the data to be used as test set (default is 0.0).
        batch_size : int, optional
                Batch size for training (default is 64).
        epochs : int, optional
                Number of epochs to train for (default is 300).
        recompile_pretrained : bool, optional
                Whether to recompile a pretrained model (default is False).
        learning_rate : float, optional
                Learning rate for the optimizer (default is 0.01).
        loss_reg : str or keras.losses.Loss, optional
                Loss function for the regression model (default is "mse").
        loss_class : str or keras.losses.Loss, optional
                Loss function for the classification model (default is CategoricalCrossentropy(from_logits=False)).

        Notes
        -----
        - The method automatically splits the dataset into training, validation, and test sets according to the specified splits.

        """

        # Lazy import for TensorFlow loss class
        if loss_class is None:
            loss_class = CategoricalCrossentropy(from_logits=False)

        if not hasattr(self, "normalization_percentile"):
            self.normalization_percentile = normalization_percentile
        if not hasattr(self, "normalization_values"):
            self.normalization_values = normalization_values
        if not hasattr(self, "normalization_clip"):
            self.normalization_clip = normalization_clip

        self.callbacks = callbacks
        self.normalize = normalize
        (
            self.normalization_percentile,
            self.normalization_values,
            self.normalization_clip,
        ) = _interpret_normalization_parameters(
            self.n_channels,
            self.normalization_percentile,
            self.normalization_values,
            self.normalization_clip,
        )

        self.datasets = [rf"{d}" if isinstance(d, str) else d for d in datasets]
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.test_split = test_split
        self.augment = augment
        self.augmentation_factor = augmentation_factor
        self.model_name = rf"{model_name}"
        self.target_directory = rf"{target_directory}"
        self.model_folder = os.sep.join([self.target_directory, self.model_name])
        self.recompile_pretrained = recompile_pretrained
        self.learning_rate = learning_rate
        self.loss_reg = loss_reg
        self.loss_class = loss_class
        self.show_plots = show_plots
        self.channel_option = channel_option

        assert self.n_channels == len(
            self.channel_option
        ), f"Mismatch between the channel option and the number of channels of the model..."

        if isinstance(self.datasets[0], dict):
            self.datasets = [self.datasets]

        self.list_of_sets = []
        for ds in self.datasets:
            if isinstance(ds, str):
                self.list_of_sets.extend(glob(os.sep.join([ds, "*.npy"])))
            else:
                self.list_of_sets.append(ds)

        print(f"Found {len(self.list_of_sets)} datasets...")

        self.prepare_sets()
        self.train_generic()

    def fit(
        self,
        x_train,
        y_time_train,
        y_class_train,
        normalize=True,
        normalization_percentile=None,
        normalization_values=None,
        normalization_clip=None,
        pad=True,
        validation_data=None,
        test_data=None,
        channel_option=["live_nuclei_channel", "dead_nuclei_channel"],
        model_name=None,
        target_directory=None,
        augment=True,
        augmentation_factor=3,
        validation_split=0.25,
        batch_size=64,
        epochs=300,
        recompile_pretrained=False,
        learning_rate=0.001,
        loss_reg="mse",
        loss_class=None,
    ):
        """
        Trains the model using provided datasets.

        Parameters
        ----------
        Same as `fit_from_directory`, but instead of loading data from directories, this method accepts preloaded and
        optionally preprocessed datasets directly.

        Notes
        -----
        - This method provides an alternative way to train the model when data is already loaded into memory, offering
          flexibility for data preprocessing steps outside this class.

        """

        # Lazy import for TensorFlow loss class
        if loss_class is None:
            loss_class = CategoricalCrossentropy(from_logits=False)

        self.normalize = normalize
        if not hasattr(self, "normalization_percentile"):
            self.normalization_percentile = normalization_percentile
        if not hasattr(self, "normalization_values"):
            self.normalization_values = normalization_values
        if not hasattr(self, "normalization_clip"):
            self.normalization_clip = normalization_clip
        (
            self.normalization_percentile,
            self.normalization_values,
            self.normalization_clip,
        ) = _interpret_normalization_parameters(
            self.n_channels,
            self.normalization_percentile,
            self.normalization_values,
            self.normalization_clip,
        )

        self.x_train = x_train
        self.y_class_train = y_class_train
        self.y_time_train = y_time_train
        self.channel_option = channel_option

        assert self.n_channels == len(
            self.channel_option
        ), f"Mismatch between the channel option and the number of channels of the model..."

        if pad:
            self.x_train = pad_to_model_length(self.x_train, self.model_signal_length)

        assert self.x_train.shape[1:] == (
            self.model_signal_length,
            self.n_channels,
        ), f"Shape mismatch between the provided training fluorescence signals and the model..."

        # If y-class is not one-hot encoded, encode it
        if self.y_class_train.shape[-1] != self.n_classes:
            self.class_weights = compute_weights(
                y=self.y_class_train,
                class_weight="balanced",
                classes=np.unique(self.y_class_train),
            )
            self.y_class_train = to_categorical(self.y_class_train, num_classes=3)

        if self.normalize:
            self.y_time_train = (
                self.y_time_train.astype(np.float32) / self.model_signal_length
            )
            self.x_train = normalize_signal_set(
                self.x_train,
                self.channel_option,
                normalization_percentile=self.normalization_percentile,
                normalization_values=self.normalization_values,
                normalization_clip=self.normalization_clip,
            )

        if validation_data is not None:
            try:
                self.x_val = validation_data[0]
                if pad:
                    self.x_val = pad_to_model_length(
                        self.x_val, self.model_signal_length
                    )
                self.y_class_val = validation_data[1]
                if self.y_class_val.shape[-1] != self.n_classes:
                    self.y_class_val = to_categorical(self.y_class_val, num_classes=3)
                self.y_time_val = validation_data[2]
                if self.normalize:
                    self.y_time_val = (
                        self.y_time_val.astype(np.float32) / self.model_signal_length
                    )
                    self.x_val = normalize_signal_set(
                        self.x_val,
                        self.channel_option,
                        normalization_percentile=self.normalization_percentile,
                        normalization_values=self.normalization_values,
                        normalization_clip=self.normalization_clip,
                    )

            except Exception as e:
                print(f"Could not load validation data, error {e}...")
        else:
            self.validation_split = validation_split

        if test_data is not None:
            try:
                self.x_test = test_data[0]
                if pad:
                    self.x_test = pad_to_model_length(
                        self.x_test, self.model_signal_length
                    )
                self.y_class_test = test_data[1]
                if self.y_class_test.shape[-1] != self.n_classes:
                    self.y_class_test = to_categorical(self.y_class_test, num_classes=3)
                self.y_time_test = test_data[2]
                if self.normalize:
                    self.y_time_test = (
                        self.y_time_test.astype(np.float32) / self.model_signal_length
                    )
                    self.x_test = normalize_signal_set(
                        self.x_test,
                        self.channel_option,
                        normalization_percentile=self.normalization_percentile,
                        normalization_values=self.normalization_values,
                        normalization_clip=self.normalization_clip,
                    )
            except Exception as e:
                print(f"Could not load test data, error {e}...")

        self.batch_size = batch_size
        self.epochs = epochs
        self.augment = augment
        self.augmentation_factor = augmentation_factor
        if self.augmentation_factor == 1:
            self.augment = False
        self.model_name = model_name
        self.target_directory = target_directory
        self.model_folder = os.sep.join([self.target_directory, self.model_name])
        self.recompile_pretrained = recompile_pretrained
        self.learning_rate = learning_rate
        self.loss_reg = loss_reg
        self.loss_class = loss_class

        self.train_generic()

    def train_generic(self):

        if not os.path.exists(self.model_folder):
            os.mkdir(self.model_folder)

        self.train_classifier()
        self.train_regressor()

        config_input = {
            "channels": self.channel_option,
            "model_signal_length": self.model_signal_length,
            "label": self.label,
            "normalize": self.normalize,
            "normalization_percentile": self.normalization_percentile,
            "normalization_values": self.normalization_values,
            "normalization_clip": self.normalization_clip,
        }
        json_string = json.dumps(config_input)
        with open(
            os.sep.join([self.model_folder, "config_input.json"]), "w"
        ) as outfile:
            outfile.write(json_string)

        # Free memory by clearing large training arrays
        import gc

        for attr in [
            "x_train",
            "x_val",
            "x_test",
            "x_set",
            "y_time_train",
            "y_time_val",
            "y_time_test",
            "y_time_set",
            "y_class_train",
            "y_class_val",
            "y_class_test",
            "y_class_set",
        ]:
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()

    def predict_class(
        self, x, normalize=True, pad=True, return_one_hot=False, interpolate=True
    ):
        """
        Predicts the class of input signals using the trained classification model.

        Parameters
        ----------
        x : ndarray
                The input signals for which to predict classes.
        normalize : bool, optional
                Whether to normalize the input signals (default is True).
        pad : bool, optional
                Whether to pad the input signals to match the model's expected signal length (default is True).
        return_one_hot : bool, optional
                Whether to return predictions in one-hot encoded format (default is False).
        interpolate : bool, optional
                Whether to interpolate the input signals (default is True).

        Returns
        -------
        ndarray
                The predicted classes for the input signals. If `return_one_hot` is True, predictions are returned in one-hot
                encoded format, otherwise as integer labels.

        Notes
        -----
        - The method processes the input signals according to the specified options to ensure compatibility with the model's
          input requirements.

        """

        self.x = np.copy(x)
        self.normalize = normalize
        self.pad = pad
        self.return_one_hot = return_one_hot
        # self.max_relevant_time = np.shape(self.x)[1]
        # print(f'Max relevant time: {self.max_relevant_time}')

        if self.pad:
            self.x = pad_to_model_length(self.x, self.model_signal_length)

        if self.normalize:
            self.x = normalize_signal_set(
                self.x,
                self.channel_option,
                normalization_percentile=self.normalization_percentile,
                normalization_values=self.normalization_values,
                normalization_clip=self.normalization_clip,
            )

        # implement auto interpolation here!!
        # self.x = self.interpolate_signals(self.x)

        # for i in range(5):
        # 	plt.plot(self.x[i,:,0])
        # 	plt.show()

        try:
            n_channels = self.model_class.layers[0].input_shape[0][-1]
            model_signal_length = self.model_class.layers[0].input_shape[0][-2]
        except AttributeError:
            n_channels = self.model_class.input_shape[-1]
            model_signal_length = self.model_class.input_shape[-2]

        assert (
            self.x.shape[-1] == n_channels
        ), f"Shape mismatch between the input shape and the model input shape..."
        assert (
            self.x.shape[-2] == model_signal_length
        ), f"Shape mismatch between the input shape and the model input shape..."

        self.class_predictions_one_hot = self.model_class.predict(self.x)
        self.class_predictions = self.class_predictions_one_hot.argmax(axis=1)

        if self.return_one_hot:
            return self.class_predictions_one_hot
        else:
            return self.class_predictions

    def predict_time_of_interest(
        self, x, class_predictions=None, normalize=True, pad=True
    ):
        """
        Predicts the time of interest for input signals using the trained regression model.

        Parameters
        ----------
        x : ndarray
                The input signals for which to predict times of interest.
        class_predictions : ndarray, optional
                The predicted classes for the input signals. If provided, time of interest predictions are only made for
                signals predicted to belong to a specific class (default is None).
        normalize : bool, optional
                Whether to normalize the input signals (default is True).
        pad : bool, optional
                Whether to pad the input signals to match the model's expected signal length (default is True).

        Returns
        -------
        ndarray
                The predicted times of interest for the input signals.

        Notes
        -----
        - The method processes the input signals according to the specified options and uses the regression model to
          predict times at which a particular event of interest occurs.

        """

        self.x = np.copy(x)
        self.normalize = normalize
        self.pad = pad
        # self.max_relevant_time = np.shape(self.x)[1]
        # print(f'Max relevant time: {self.max_relevant_time}')

        if class_predictions is not None:
            self.class_predictions = class_predictions

        if self.pad:
            self.x = pad_to_model_length(self.x, self.model_signal_length)

        if self.normalize:
            self.x = normalize_signal_set(
                self.x,
                self.channel_option,
                normalization_percentile=self.normalization_percentile,
                normalization_values=self.normalization_values,
                normalization_clip=self.normalization_clip,
            )

        try:
            n_channels = self.model_reg.layers[0].input_shape[0][-1]
            model_signal_length = self.model_reg.layers[0].input_shape[0][-2]
        except AttributeError:
            n_channels = self.model_reg.input_shape[-1]
            model_signal_length = self.model_reg.input_shape[-2]

        assert (
            self.x.shape[-1] == n_channels
        ), f"Shape mismatch between the input shape and the model input shape..."
        assert (
            self.x.shape[-2] == model_signal_length
        ), f"Shape mismatch between the input shape and the model input shape..."

        if np.any(self.class_predictions == 0):
            self.time_predictions = (
                self.model_reg.predict(self.x[self.class_predictions == 0])
                * self.model_signal_length
            )
            self.time_predictions = self.time_predictions[:, 0]
            self.time_predictions_recast = np.zeros(len(self.x)) - 1.0
            self.time_predictions_recast[self.class_predictions == 0] = (
                self.time_predictions
            )
        else:
            self.time_predictions_recast = np.zeros(len(self.x)) - 1.0
        return self.time_predictions_recast

    def interpolate_signals(self, x_set):
        """
        Interpolates missing values in the input signal set.

        Parameters
        ----------
        x_set : ndarray
                The input signal set with potentially missing values.

        Returns
        -------
        ndarray
                The input signal set with missing values interpolated.

        Notes
        -----
        - This method is useful for preparing signals that have gaps or missing time points before further processing
          or model training.

        """

        for i in range(len(x_set)):
            for k in range(x_set.shape[-1]):
                x = x_set[i, :, k]
                not_nan = np.logical_not(np.isnan(x))
                indices = np.arange(len(x))
                interp = interp1d(
                    indices[not_nan],
                    x[not_nan],
                    fill_value=(0.0, 0.0),
                    bounds_error=False,
                )
                x_set[i, :, k] = interp(indices)
        return x_set

    def train_classifier(self):
        """
        Trains the classifier component of the model to predict event classes in signals.

        This method compiles the classifier model (if not pretrained or if recompilation is requested) and
        trains it on the prepared dataset. The training process includes validation and early stopping based
        on precision to prevent overfitting.

        Notes
        -----
        - The classifier model predicts the class of each signal, such as live, dead, or miscellaneous.
        - Training parameters such as epochs, batch size, and learning rate are specified during class instantiation.
        - Model performance metrics and training history are saved for analysis.

        """

        # if pretrained model
        self.n_classes = 3

        if self.pretrained is not None:
            # if recompile
            if self.recompile_pretrained:
                print(
                    "Recompiling the pretrained classifier model... Warning, this action reinitializes all the weights; are you sure that this is what you intended?"
                )
                self.model_class.set_weights(
                    clone_model(self.model_class).get_weights()
                )
                self.model_class.compile(
                    optimizer=Adam(learning_rate=self.learning_rate),
                    loss=self.loss_class,
                    metrics=[
                        "accuracy",
                        Precision(),
                        Recall(),
                        MeanIoU(
                            num_classes=self.n_classes,
                            name="iou",
                            dtype=float,
                            sparse_y_true=False,
                            sparse_y_pred=False,
                        ),
                    ],
                )
            else:
                # Recompile to avoid crash
                self.model_class.compile(
                    optimizer=Adam(learning_rate=self.learning_rate),
                    loss=self.loss_class,
                    metrics=[
                        "accuracy",
                        Precision(),
                        Recall(),
                        MeanIoU(
                            num_classes=self.n_classes,
                            name="iou",
                            dtype=float,
                            sparse_y_true=False,
                            sparse_y_pred=False,
                        ),
                    ],
                )

        else:
            print("Compiling the classifier...")
            self.model_class.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss=self.loss_class,
                metrics=[
                    "accuracy",
                    Precision(),
                    Recall(),
                    MeanIoU(
                        num_classes=self.n_classes,
                        name="iou",
                        dtype=float,
                        sparse_y_true=False,
                        sparse_y_pred=False,
                    ),
                ],
            )

        self.gather_callbacks("classifier")

        # for i in range(30):
        # 	for j in range(self.x_train.shape[-1]):
        # 		plt.plot(self.x_train[i,:,j])
        # 	plt.show()

        if hasattr(self, "x_val"):

            self.history_classifier = self.model_class.fit(
                x=self.x_train,
                y=self.y_class_train,
                batch_size=self.batch_size,
                class_weight=self.class_weights,
                epochs=self.epochs,
                validation_data=(self.x_val, self.y_class_val),
                callbacks=self.cb,
                verbose=1,
            )
        else:
            self.history_classifier = self.model_class.fit(
                x=self.x_train,
                y=self.y_class_train,
                batch_size=self.batch_size,
                class_weight=self.class_weights,
                epochs=self.epochs,
                callbacks=self.cb,
                validation_split=self.validation_split,
                verbose=1,
            )

        if self.show_plots:
            self.plot_model_history(mode="classifier")

        # Set current classification model as the best model
        self.model_class = load_model(
            os.sep.join([self.model_folder, "classifier.h5"]),
            custom_objects={"mse": MeanSquaredError()},
        )
        self.model_class.load_weights(os.sep.join([self.model_folder, "classifier.h5"]))

        time_callback = next(
            (cb for cb in self.cb if type(cb).__name__ == "TimeHistory"), None
        )
        self.dico = {
            "history_classifier": self.history_classifier,
            "execution_time_classifier": time_callback.times if time_callback else [],
        }

        if hasattr(self, "x_test"):

            predictions = self.model_class.predict(self.x_test).argmax(axis=1)
            ground_truth = self.y_class_test.argmax(axis=1)
            assert (
                predictions.shape == ground_truth.shape
            ), "Mismatch in shape between the predictions and the ground truth..."

            title = "Test data"
            IoU_score = jaccard_score(ground_truth, predictions, average=None)
            balanced_accuracy = balanced_accuracy_score(ground_truth, predictions)
            precision = precision_score(ground_truth, predictions, average=None)
            recall = recall_score(ground_truth, predictions, average=None)

            print(f"Test IoU score: {IoU_score}")
            print(f"Test Balanced accuracy score: {balanced_accuracy}")
            print(f"Test Precision: {precision}")
            print(f"Test Recall: {recall}")

            # Confusion matrix on test set
            results = confusion_matrix(ground_truth, predictions)
            self.dico.update(
                {
                    "test_IoU": IoU_score,
                    "test_balanced_accuracy": balanced_accuracy,
                    "test_confusion": results,
                    "test_precision": precision,
                    "test_recall": recall,
                }
            )

            if self.show_plots:
                try:
                    ConfusionMatrixDisplay.from_predictions(
                        ground_truth,
                        predictions,
                        cmap="Blues",
                        normalize="pred",
                        display_labels=["event", "no event", "left censored"],
                    )
                    plt.savefig(
                        os.sep.join([self.model_folder, "test_confusion_matrix.png"]),
                        bbox_inches="tight",
                        dpi=300,
                    )
                    # plt.pause(3)
                    plt.close()
                except Exception as e:
                    print(e)
                    pass
            print("Test set: ", classification_report(ground_truth, predictions))

        if hasattr(self, "x_val"):
            predictions = self.model_class.predict(self.x_val).argmax(axis=1)
            ground_truth = self.y_class_val.argmax(axis=1)
            assert (
                ground_truth.shape == predictions.shape
            ), "Mismatch in shape between the predictions and the ground truth..."
            title = "Validation data"

            # Validation scores
            IoU_score = jaccard_score(ground_truth, predictions, average=None)
            balanced_accuracy = balanced_accuracy_score(ground_truth, predictions)
            precision = precision_score(ground_truth, predictions, average=None)
            recall = recall_score(ground_truth, predictions, average=None)

            print(f"Validation IoU score: {IoU_score}")
            print(f"Validation Balanced accuracy score: {balanced_accuracy}")
            print(f"Validation Precision: {precision}")
            print(f"Validation Recall: {recall}")

            # Confusion matrix on validation set
            results = confusion_matrix(ground_truth, predictions)
            self.dico.update(
                {
                    "val_IoU": IoU_score,
                    "val_balanced_accuracy": balanced_accuracy,
                    "val_confusion": results,
                    "val_precision": precision,
                    "val_recall": recall,
                }
            )

            if self.show_plots:
                try:
                    ConfusionMatrixDisplay.from_predictions(
                        ground_truth,
                        predictions,
                        cmap="Blues",
                        normalize="pred",
                        display_labels=["event", "no event", "left censored"],
                    )
                    plt.savefig(
                        os.sep.join(
                            [self.model_folder, "validation_confusion_matrix.png"]
                        ),
                        bbox_inches="tight",
                        dpi=300,
                    )
                    # plt.pause(3)
                    plt.close()
                except Exception as e:
                    print(e)
                    pass
            print("Validation set: ", classification_report(ground_truth, predictions))

            # Send result to GUI and wait
            for cb in self.cb:
                if hasattr(cb, "on_training_result"):
                    cb.on_training_result(self.dico)
            time.sleep(3)

    def train_regressor(self):
        """
        Trains the regressor component of the model to estimate the time of interest for events in signals.

        This method compiles the regressor model (if not pretrained or if recompilation is requested) and
        trains it on a subset of the prepared dataset containing signals with events. The training process
        includes validation and early stopping based on mean squared error to prevent overfitting.

        Notes
        -----
        - The regressor model estimates the time at which an event of interest occurs within each signal.
        - Only signals predicted to have an event by the classifier model are used for regressor training.
        - Model performance metrics and training history are saved for analysis.

        """

        # Compile model
        # if pretrained model
        if self.pretrained is not None:
            # if recompile
            if self.recompile_pretrained:
                print(
                    "Recompiling the pretrained regressor model... Warning, this action reinitializes all the weights; are you sure that this is what you intended?"
                )
                self.model_reg.set_weights(clone_model(self.model_reg).get_weights())
                self.model_reg.compile(
                    optimizer=Adam(learning_rate=self.learning_rate),
                    loss=self.loss_reg,
                    metrics=["mse", "mae"],
                )
            else:
                self.model_reg.compile(
                    optimizer=Adam(learning_rate=self.learning_rate),
                    loss=self.loss_reg,
                    metrics=["mse", "mae"],
                )

        else:
            print("Compiling the regressor...")
            self.model_reg.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss=self.loss_reg,
                metrics=["mse", "mae"],
            )

        self.gather_callbacks("regressor")

        # Train on subset of data with event

        subset = self.x_train[np.argmax(self.y_class_train, axis=1) == 0]
        # for i in range(30):
        # 	plt.plot(subset[i,:,0],c="tab:red")
        # 	plt.plot(subset[i,:,1],c="tab:blue")
        # 	plt.show()

        if hasattr(self, "x_val"):
            self.history_regressor = self.model_reg.fit(
                x=self.x_train[np.argmax(self.y_class_train, axis=1) == 0],
                y=self.y_time_train[np.argmax(self.y_class_train, axis=1) == 0],
                batch_size=self.batch_size,
                epochs=self.epochs * 2,
                validation_data=(
                    self.x_val[np.argmax(self.y_class_val, axis=1) == 0],
                    self.y_time_val[np.argmax(self.y_class_val, axis=1) == 0],
                ),
                callbacks=self.cb,
                verbose=1,
            )
        else:
            self.history_regressor = self.model_reg.fit(
                x=self.x_train[np.argmax(self.y_class_train, axis=1) == 0],
                y=self.y_time_train[np.argmax(self.y_class_train, axis=1) == 0],
                batch_size=self.batch_size,
                epochs=self.epochs * 2,
                callbacks=self.cb,
                validation_split=self.validation_split,
                verbose=1,
            )

        if self.show_plots:
            self.plot_model_history(mode="regressor")
        time_callback = next(
            (cb for cb in self.cb if type(cb).__name__ == "TimeHistory"), None
        )
        self.dico.update(
            {
                "history_regressor": self.history_regressor,
                "execution_time_regressor": (
                    time_callback.times if time_callback else []
                ),
            }
        )

        # Evaluate best model
        self.model_reg = load_model(
            os.sep.join([self.model_folder, "regressor.h5"]),
            custom_objects={"mse": MeanSquaredError()},
        )
        self.model_reg.load_weights(os.sep.join([self.model_folder, "regressor.h5"]))
        self.evaluate_regression_model()

        try:
            np.save(os.sep.join([self.model_folder, "scores.npy"]), self.dico)
        except Exception as e:
            print(e)

    def plot_model_history(self, mode="regressor"):
        """
        Generates and saves plots of the training history for the classifier or regressor model.

        Parameters
        ----------
        mode : str, optional
                Specifies which model's training history to plot. Options are "classifier" or "regressor". Default is "regressor".

        Notes
        -----
        - Plots include loss and accuracy metrics over epochs for the classifier, and loss metrics for the regressor.
        - The plots are saved as image files in the model's output directory.

        """

        if mode == "regressor":
            try:
                plt.plot(self.history_regressor.history["loss"])
                plt.plot(self.history_regressor.history["val_loss"])
                plt.title("model loss")
                plt.ylabel("loss")
                plt.xlabel("epoch")
                plt.yscale("log")
                plt.legend(["train", "val"], loc="upper left")
                # plt.pause(3)
                plt.savefig(
                    os.sep.join([self.model_folder, "regression_loss.png"]),
                    bbox_inches="tight",
                    dpi=300,
                )
                plt.close()
            except Exception as e:
                print(f"Error {e}; could not generate plot...")
        elif mode == "classifier":
            try:
                plt.plot(self.history_classifier.history["precision"])
                plt.plot(self.history_classifier.history["val_precision"])
                plt.title("model precision")
                plt.ylabel("precision")
                plt.xlabel("epoch")
                plt.legend(["train", "val"], loc="upper left")
                # plt.pause(3)
                plt.savefig(
                    os.sep.join([self.model_folder, "classification_loss.png"]),
                    bbox_inches="tight",
                    dpi=300,
                )
                plt.close()
            except Exception as e:
                print(f"Error {e}; could not generate plot...")
        else:
            return None

    def evaluate_regression_model(self):
        """
        Evaluates the performance of the trained regression model on test and validation datasets.

        This method calculates and prints mean squared error and mean absolute error metrics for the regression model's
        predictions. It also generates regression plots comparing predicted times of interest to true values.

        Notes
        -----
        - Evaluation is performed on both test and validation datasets, if available.
        - Regression plots and performance metrics are saved in the model's output directory.

        """
        mse = MeanSquaredError()
        mae = MeanAbsoluteError()

        if hasattr(self, "x_test"):

            print("Evaluate on test set...")
            predictions = self.model_reg.predict(
                self.x_test[np.argmax(self.y_class_test, axis=1) == 0],
                batch_size=self.batch_size,
            )[:, 0]
            ground_truth = self.y_time_test[np.argmax(self.y_class_test, axis=1) == 0]
            assert (
                predictions.shape == ground_truth.shape
            ), "Shape mismatch between predictions and ground truths..."

            test_mse = mse(ground_truth, predictions).numpy()
            test_mae = mae(ground_truth, predictions).numpy()
            print(f"MSE on test set: {test_mse}...")
            print(f"MAE on test set: {test_mae}...")
            if self.show_plots:
                regression_plot(
                    predictions,
                    ground_truth,
                    savepath=os.sep.join([self.model_folder, "test_regression.png"]),
                )
            self.dico.update({"test_mse": test_mse, "test_mae": test_mae})

        if hasattr(self, "x_val"):
            # Validation set
            predictions = self.model_reg.predict(
                self.x_val[np.argmax(self.y_class_val, axis=1) == 0],
                batch_size=self.batch_size,
            )[:, 0]
            ground_truth = self.y_time_val[np.argmax(self.y_class_val, axis=1) == 0]
            assert (
                predictions.shape == ground_truth.shape
            ), "Shape mismatch between predictions and ground truths..."

            val_mse = mse(ground_truth, predictions).numpy()
            val_mae = mae(ground_truth, predictions).numpy()

            if self.show_plots:
                regression_plot(
                    predictions,
                    ground_truth,
                    savepath=os.sep.join(
                        [self.model_folder, "validation_regression.png"]
                    ),
                )
            print(f"MSE on validation set: {val_mse}...")
            print(f"MAE on validation set: {val_mae}...")

            self.dico.update(
                {
                    "val_mse": val_mse,
                    "val_mae": val_mae,
                    "val_predictions": predictions,
                    "val_ground_truth": ground_truth,
                }
            )

            # Send result to GUI and wait
            for cb in self.cb:
                if hasattr(cb, "on_training_result"):
                    cb.on_training_result(self.dico)
            time.sleep(3)

    def gather_callbacks(self, mode):
        """
        Prepares a list of Keras callbacks for model training based on the specified mode.

        Parameters
        ----------
        mode : str
                The training mode for which callbacks are being prepared. Options are "classifier" or "regressor".

        Notes
        -----
        - Callbacks include learning rate reduction on plateau, early stopping, model checkpointing, and TensorBoard logging.
        - The list of callbacks is stored in the class attribute `cb` and used during model training.

        """

        self.cb = []

        if mode == "classifier":

            reduce_lr = ReduceLROnPlateau(
                monitor="val_iou",
                factor=0.5,
                patience=30,
                cooldown=10,
                min_lr=5e-10,
                min_delta=1.0e-10,
                verbose=1,
                mode="max",
            )
            self.cb.append(reduce_lr)
            csv_logger = CSVLogger(
                os.sep.join([self.model_folder, "log_classifier.csv"]),
                append=True,
                separator=";",
            )
            self.cb.append(csv_logger)
            checkpoint_path = os.sep.join([self.model_folder, "classifier.h5"])
            cp_callback = ModelCheckpoint(
                checkpoint_path,
                monitor="val_iou",
                mode="max",
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                save_freq="epoch",
            )
            self.cb.append(cp_callback)

            callback_stop = EarlyStopping(monitor="val_iou", mode="max", patience=100)
            self.cb.append(callback_stop)

        elif mode == "regressor":

            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=30,
                cooldown=10,
                min_lr=5e-10,
                min_delta=1.0e-10,
                verbose=1,
                mode="min",
            )
            self.cb.append(reduce_lr)

            csv_logger = CSVLogger(
                os.sep.join([self.model_folder, "log_regressor.csv"]),
                append=True,
                separator=";",
            )
            self.cb.append(csv_logger)

            checkpoint_path = os.sep.join([self.model_folder, "regressor.h5"])
            cp_callback = ModelCheckpoint(
                checkpoint_path,
                monitor="val_loss",
                mode="min",
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                save_freq="epoch",
            )
            self.cb.append(cp_callback)

            callback_stop = EarlyStopping(monitor="val_loss", mode="min", patience=200)
            self.cb.append(callback_stop)

        log_dir = self.model_folder + os.sep
        cb_tb = TensorBoard(log_dir=log_dir, update_freq="batch")
        self.cb.append(cb_tb)

        cb_time = TimeHistory()
        self.cb.append(cb_time)

        if hasattr(self, "callbacks") and self.callbacks is not None:
            self.cb.extend(self.callbacks)

    def prepare_sets(self):
        """
        Generates and preprocesses training, validation, and test sets from loaded annotations.

        This method loads signal data from annotation files, normalizes and interpolates the signals, and splits
        the dataset into training, validation, and test sets according to specified proportions.

        Notes
        -----
        - Signal annotations are expected to be stored in .npy format and contain required channels and event information.
        - The method applies specified normalization and interpolation options to prepare the signals for model training.

        """

        self.x_set = []
        self.y_time_set = []
        self.y_class_set = []

        if isinstance(self.list_of_sets[0], str):
            # Case 1: a list of npy files to be loaded
            for s in self.list_of_sets:

                signal_dataset = self.load_set(s)
                selected_signals, max_length = self.find_best_signal_match(
                    signal_dataset
                )
                signals_recast, classes, times_of_interest = (
                    self.cast_signals_into_training_data(
                        signal_dataset, selected_signals, max_length
                    )
                )
                signals_recast, times_of_interest = self.normalize_signals(
                    signals_recast, times_of_interest
                )

                self.x_set.extend(signals_recast)
                self.y_time_set.extend(times_of_interest)
                self.y_class_set.extend(classes)

        elif isinstance(self.list_of_sets[0], list):
            # Case 2: a list of sets (already loaded)
            for signal_dataset in self.list_of_sets:

                selected_signals, max_length = self.find_best_signal_match(
                    signal_dataset
                )
                signals_recast, classes, times_of_interest = (
                    self.cast_signals_into_training_data(
                        signal_dataset, selected_signals, max_length
                    )
                )
                signals_recast, times_of_interest = self.normalize_signals(
                    signals_recast, times_of_interest
                )

                self.x_set.extend(signals_recast)
                self.y_time_set.extend(times_of_interest)
                self.y_class_set.extend(classes)

        self.x_set = np.array(self.x_set).astype(np.float32)
        self.x_set = self.interpolate_signals(self.x_set)

        self.y_time_set = np.array(self.y_time_set).astype(np.float32)
        self.y_class_set = np.array(self.y_class_set).astype(np.float32)

        class_test = np.isin(self.y_class_set, [0, 1, 2])
        self.x_set = self.x_set[class_test]
        self.y_time_set = self.y_time_set[class_test]
        self.y_class_set = self.y_class_set[class_test]

        # Compute class weights and one-hot encode
        self.class_weights = compute_weights(self.y_class_set)
        self.nbr_classes = 3  # len(np.unique(self.y_class_set))
        self.y_class_set = to_categorical(self.y_class_set, num_classes=3)

        ds = train_test_split(
            self.x_set,
            self.y_time_set,
            self.y_class_set,
            validation_size=self.validation_split,
            test_size=self.test_split,
        )

        self.x_train = ds["x_train"]
        self.x_val = ds["x_val"]
        self.y_time_train = ds["y1_train"].astype(np.float32)
        self.y_time_val = ds["y1_val"].astype(np.float32)
        self.y_class_train = ds["y2_train"]
        self.y_class_val = ds["y2_val"]

        if self.test_split > 0:
            self.x_test = ds["x_test"]
            self.y_time_test = ds["y1_test"].astype(np.float32)
            self.y_class_test = ds["y2_test"]

        if self.augment:
            self.augment_training_set()

    def augment_training_set(self, time_shift=True):
        """
        Augments the training dataset with artificially generated data to increase model robustness.

        Parameters
        ----------
        time_shift : bool, optional
                Specifies whether to include time-shifted versions of signals in the augmented dataset. Default is True.

        Notes
        -----
        - Augmentation strategies include random time shifting and signal modifications to simulate variations in real data.
        - The augmented dataset is used for training the classifier and regressor models to improve generalization.

        """

        nbr_augment = self.augmentation_factor * len(self.x_train)
        randomize = np.arange(len(self.x_train))

        unique, counts = np.unique(
            self.y_class_train.argmax(axis=1), return_counts=True
        )
        frac = counts / sum(counts)
        weights = [frac[0] / f for f in frac]
        weights[0] = weights[0] * 3

        self.pre_augment_weights = weights / sum(weights)
        weights_array = [
            self.pre_augment_weights[a.argmax()] for a in self.y_class_train
        ]

        indices = random.choices(randomize, k=nbr_augment, weights=weights_array)

        x_train_aug = []
        y_time_train_aug = []
        y_class_train_aug = []

        counts = [0.0, 0.0, 0.0]
        # warning augmentation creates class 2 even if does not exist in data, need to address this
        for k in indices:
            counts[self.y_class_train[k].argmax()] += 1
            aug = augmenter(
                self.x_train[k],
                self.y_time_train[k],
                self.y_class_train[k],
                self.model_signal_length,
                time_shift=time_shift,
            )
            x_train_aug.append(aug[0])
            y_time_train_aug.append(aug[1])
            y_class_train_aug.append(aug[2])

        # Save augmented training set
        self.x_train = np.array(x_train_aug)
        self.y_time_train = np.array(y_time_train_aug)
        self.y_class_train = np.array(y_class_train_aug)

        self.class_weights = compute_weights(self.y_class_train.argmax(axis=1))
        print(f"New class weights: {self.class_weights}...")

    def load_set(self, signal_dataset):
        return np.load(signal_dataset, allow_pickle=True)

    def find_best_signal_match(self, signal_dataset):

        required_signals = self.channel_option
        available_signals = list(signal_dataset[0].keys())

        selected_signals = []
        for s in required_signals:
            pattern_test = [s in a for a in available_signals]
            if np.any(pattern_test):
                valid_columns = np.array(available_signals)[np.array(pattern_test)]
                if len(valid_columns) == 1:
                    selected_signals.append(valid_columns[0])
                else:
                    print(f"Found several candidate signals: {valid_columns}")
                    for vc in natsorted(valid_columns):
                        if "circle" in vc:
                            selected_signals.append(vc)
                            break
                    else:
                        selected_signals.append(valid_columns[0])
            else:
                return None

        key_to_check = selected_signals[0]  # self.channel_option[0]
        signal_lengths = [len(l[key_to_check]) for l in signal_dataset]
        max_length = np.amax(signal_lengths)

        return selected_signals, max_length

    def cast_signals_into_training_data(
        self, signal_dataset, selected_signals, max_length
    ):

        signals_recast = np.zeros((len(signal_dataset), max_length, self.n_channels))
        classes = np.zeros(len(signal_dataset))
        times_of_interest = np.zeros(len(signal_dataset))

        for k in range(len(signal_dataset)):

            for i in range(self.n_channels):
                try:
                    # take into account timeline for accurate time regression

                    if selected_signals[i].startswith("pair_"):
                        timeline = signal_dataset[k]["pair_FRAME"].astype(int)
                    elif selected_signals[i].startswith("reference_"):
                        timeline = signal_dataset[k]["reference_FRAME"].astype(int)
                    elif selected_signals[i].startswith("neighbor_"):
                        timeline = signal_dataset[k]["neighbor_FRAME"].astype(int)
                    else:
                        timeline = signal_dataset[k]["FRAME"].astype(int)
                    signals_recast[k, timeline, i] = signal_dataset[k][
                        selected_signals[i]
                    ]
                except:
                    print(
                        f"Attribute {selected_signals[i]} matched to {self.channel_option[i]} not found in annotation..."
                    )
                    pass

            classes[k] = signal_dataset[k]["class"]
            times_of_interest[k] = signal_dataset[k]["time_of_interest"]

        # Correct absurd times of interest
        times_of_interest[np.nonzero(classes)] = -1
        times_of_interest[(times_of_interest <= 0.0)] = -1

        return signals_recast, classes, times_of_interest

    def normalize_signals(self, signals_recast, times_of_interest):

        signals_recast = pad_to_model_length(signals_recast, self.model_signal_length)
        if self.normalize:
            signals_recast = normalize_signal_set(
                signals_recast,
                self.channel_option,
                normalization_percentile=self.normalization_percentile,
                normalization_values=self.normalization_values,
                normalization_clip=self.normalization_clip,
            )

        # Trivial normalization for time of interest
        times_of_interest /= self.model_signal_length

        return signals_recast, times_of_interest


def residual_block1D(
    x, number_of_filters, kernel_size=8, match_filter_size=True, connection="identity"
):
    """

    Create a 1D residual block.

    Parameters
    ----------
    x : Tensor
            Input tensor.
    number_of_filters : int
            Number of filters in the convolutional layers.
    match_filter_size : bool, optional
            Whether to match the filter size of the skip connection to the output. Default is True.

    Returns
    -------
    Tensor
            Output tensor of the residual block.

    Notes
    -----
    This function creates a 1D residual block by performing the original mapping followed by adding a skip connection
    and applying non-linear activation. The skip connection allows the gradient to flow directly to earlier layers and
    helps mitigate the vanishing gradient problem. The residual block consists of three convolutional layers with
    batch normalization and ReLU activation functions.

    If `match_filter_size` is True, the skip connection is adjusted to have the same number of filters as the output.
    Otherwise, the skip connection is kept as is.

    Examples
    --------
    >>> inputs = Input(shape=(10, 3))
    >>> x = residual_block1D(inputs, 64)
    # Create a 1D residual block with 64 filters and apply it to the input tensor.

    """

    # Create skip connection
    x_skip = x

    # Perform the original mapping
    if connection == "identity":
        x = Conv1D(
            number_of_filters, kernel_size=kernel_size, strides=1, padding="same"
        )(x_skip)
    elif connection == "projection":
        x = ZeroPadding1D(padding=kernel_size // 2)(x_skip)
        x = Conv1D(
            number_of_filters, kernel_size=kernel_size, strides=2, padding="valid"
        )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(number_of_filters, kernel_size=kernel_size, strides=1, padding="same")(x)
    x = BatchNormalization()(x)

    if match_filter_size and connection == "identity":
        x_skip = Conv1D(number_of_filters, kernel_size=1, padding="same")(x_skip)
    elif match_filter_size and connection == "projection":
        x_skip = Conv1D(number_of_filters, kernel_size=1, strides=2, padding="valid")(
            x_skip
        )

    # Add the skip connection to the regular mapping
    x = Add()([x, x_skip])

    # Nonlinearly activate the result
    x = Activation("relu")(x)

    # Return the result
    return x


def MultiscaleResNetModel(
    n_channels,
    n_classes=3,
    dropout_rate=0,
    dense_collection=0,
    use_pooling=True,
    header="classifier",
    model_signal_length=128,
):
    """

    Define a generic ResNet 1D encoder model.

    Parameters
    ----------
    n_channels : int
            Number of input channels.
    n_blocks : int
            Number of residual blocks in the model.
    n_classes : int, optional
            Number of output classes. Default is 3.
    dropout_rate : float, optional
            Dropout rate to be applied. Default is 0.
    dense_collection : int, optional
            Number of neurons in the dense layer. Default is 0.
    header : str, optional
            Type of the model header. "classifier" for classification, "regressor" for regression. Default is "classifier".
    model_signal_length : int, optional
            Length of the input signal. Default is 128.

    Returns
    -------
    keras.models.Model
            ResNet 1D encoder model.

    Notes
    -----
    This function defines a generic ResNet 1D encoder model with the specified number of input channels, residual
    blocks, output classes, dropout rate, dense collection, and model header. The model architecture follows the
    ResNet principles with 1D convolutional layers and residual connections. The final activation and number of
    neurons in the output layer are determined based on the header type.

    Examples
    --------
    >>> model = ResNetModel(n_channels=3, n_blocks=4, n_classes=2, dropout_rate=0.2)
    # Define a ResNet 1D encoder model with 3 input channels, 4 residual blocks, and 2 output classes.

    """
    if header == "classifier":
        final_activation = "softmax"
        neurons_final = n_classes
    elif header == "regressor":
        final_activation = "linear"
        neurons_final = 1
    else:
        return None

    inputs = Input(
        shape=(
            model_signal_length,
            n_channels,
        )
    )
    x = ZeroPadding1D(3)(inputs)
    x = Conv1D(64, kernel_size=7, strides=2, padding="valid", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ZeroPadding1D(1)(x)
    x_common = MaxPooling1D(pool_size=3, strides=2, padding="valid")(x)

    # Block 1
    x1 = residual_block1D(x_common, 64, kernel_size=7, connection="projection")
    x1 = residual_block1D(x1, 128, kernel_size=7, connection="projection")
    x1 = residual_block1D(x1, 256, kernel_size=7, connection="projection")
    x1 = GlobalAveragePooling1D()(x1)

    # Block 2
    x2 = residual_block1D(x_common, 64, kernel_size=5, connection="projection")
    x2 = residual_block1D(x2, 128, kernel_size=5, connection="projection")
    x2 = residual_block1D(x2, 256, kernel_size=5, connection="projection")
    x2 = GlobalAveragePooling1D()(x2)

    # Block 3
    x3 = residual_block1D(x_common, 64, kernel_size=3, connection="projection")
    x3 = residual_block1D(x3, 128, kernel_size=3, connection="projection")
    x3 = residual_block1D(x3, 256, kernel_size=3, connection="projection")
    x3 = GlobalAveragePooling1D()(x3)

    x_combined = Concatenate()([x1, x2, x3])
    x_combined = Flatten()(x_combined)

    if dense_collection > 0:
        x_combined = Dense(dense_collection)(x_combined)
    if dropout_rate > 0:
        x_combined = Dropout(dropout_rate)(x_combined)

    x_combined = Dense(neurons_final, activation=final_activation, name=header)(
        x_combined
    )
    model = Model(inputs, x_combined, name=header)

    return model


def ResNetModelCurrent(
    n_channels,
    n_slices,
    depth=2,
    use_pooling=True,
    n_classes=3,
    dropout_rate=0.1,
    dense_collection=512,
    header="classifier",
    model_signal_length=128,
):
    """
    Creates a ResNet-based model tailored for signal classification or regression tasks.

    This function constructs a 1D ResNet architecture with specified parameters. The model can be configured
    for either classification or regression tasks, determined by the `header` parameter. It consists of
    configurable ResNet blocks, global average pooling, optional dense layers, and dropout for regularization.

    Parameters
    ----------
    n_channels : int
            The number of channels in the input signal.
    n_slices : int
            The number of slices (or ResNet blocks) to use in the model.
    depth : int, optional
            The depth of the network, i.e., how many times the number of filters is doubled. Default is 2.
    use_pooling : bool, optional
            Whether to use MaxPooling between ResNet blocks. Default is True.
    n_classes : int, optional
            The number of classes for the classification task. Ignored for regression. Default is 3.
    dropout_rate : float, optional
            The dropout rate for regularization. Default is 0.1.
    dense_collection : int, optional
            The number of neurons in the dense layer following global pooling. If 0, the dense layer is omitted. Default is 512.
    header : str, optional
            Specifies the task type: "classifier" for classification or "regressor" for regression. Default is "classifier".
    model_signal_length : int, optional
            The length of the input signal. Default is 128.

    Returns
    -------
    keras.Model
            The constructed Keras model ready for training or inference.

    Notes
    -----
    - The model uses Conv1D layers for signal processing and applies global average pooling before the final classification
      or regression layer.
    - The choice of `final_activation` and `neurons_final` depends on the task: "softmax" and `n_classes` for classification,
      and "linear" and 1 for regression.
    - This function relies on a custom `residual_block1D` function for constructing ResNet blocks.

    Examples
    --------
    >>> model = ResNetModelCurrent(n_channels=1, n_slices=2, depth=2, use_pooling=True, n_classes=3, dropout_rate=0.1, dense_collection=512, header="classifier", model_signal_length=128)
    # Creates a ResNet model configured for classification with 3 classes.

    """

    if header == "classifier":
        final_activation = "softmax"
        neurons_final = n_classes
    elif header == "regressor":
        final_activation = "linear"
        neurons_final = 1
    else:
        return None

    inputs = Input(
        shape=(
            model_signal_length,
            n_channels,
        )
    )
    x2 = Conv1D(64, kernel_size=1, strides=1, padding="same")(inputs)

    n_filters = 64
    for k in range(depth):
        for i in range(n_slices):
            x2 = residual_block1D(x2, n_filters, kernel_size=8)
        n_filters *= 2
        if use_pooling and k != (depth - 1):
            x2 = MaxPooling1D()(x2)

    x2 = GlobalAveragePooling1D()(x2)
    if dense_collection > 0:
        x2 = Dense(dense_collection)(x2)
    if dropout_rate > 0:
        x2 = Dropout(dropout_rate)(x2)

    x2 = Dense(neurons_final, activation=final_activation, name=header)(x2)
    model = Model(inputs, x2, name=header)

    return model


def _get_time_history_class():
    """Factory function to get TimeHistory class with lazy TensorFlow import."""

    class TimeHistory(Callback):
        """
        A custom Keras callback to log the duration of each epoch during training.

        This callback records the time taken for each epoch during the model training process, allowing for
        monitoring of training efficiency and performance over time. The times are stored in a list, with each
        element representing the duration of an epoch in seconds.

        Attributes
        ----------
        times : list
                A list of times (in seconds) taken for each epoch during the training. This list is populated as the
                training progresses.

        Methods
        -------
        on_train_begin(logs={})
                Initializes the list of times at the beginning of training.

        on_epoch_begin(epoch, logs={})
                Records the start time of the current epoch.

        on_epoch_end(epoch, logs={})
                Calculates and appends the duration of the current epoch to the `times` list.

        Notes
        -----
        - This callback is intended to be used with the `fit` method of Keras models.
        - The time measurements are made using the `time.time()` function, which provides wall-clock time.

        Examples
        --------
        >>> from keras.models import Sequential
        >>> from keras.layers import Dense
        >>> model = Sequential([Dense(10, activation='relu', input_shape=(20,)), Dense(1)])
        >>> time_callback = TimeHistory()
        >>> model.compile(optimizer='adam', loss='mean_squared_error')
        >>> model.fit(x_train, y_train, epochs=10, callbacks=[time_callback])
        >>> print(time_callback.times)
        # This will print the time taken for each epoch during the training.

        """

        def on_train_begin(self, logs={}):
            self.times = []

        def on_epoch_begin(self, epoch, logs={}):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, epoch, logs={}):
            self.times.append(time.time() - self.epoch_time_start)

    return TimeHistory


def _interpret_normalization_parameters(
    n_channels, normalization_percentile, normalization_values, normalization_clip
):
    """
    Interprets and validates normalization parameters for each channel.

    This function ensures the normalization parameters are correctly formatted and expanded to match
    the number of channels in the dataset. It provides default values and expands single values into
    lists to match the number of channels if necessary.

    Parameters
    ----------
    n_channels : int
            The number of channels in the dataset.
    normalization_percentile : list of bool or bool, optional
            Specifies whether to normalize each channel based on percentile values. If a single bool is provided,
            it is expanded to a list matching the number of channels. Default is True for all channels.
    normalization_values : list of lists or list, optional
            Specifies the percentile values [lower, upper] for normalization for each channel. If a single pair
            is provided, it is expanded to match the number of channels. Default is [[0.1, 99.9]] for all channels.
    normalization_clip : list of bool or bool, optional
            Specifies whether to clip the normalized values for each channel to the range [0, 1]. If a single bool
            is provided, it is expanded to a list matching the number of channels. Default is False for all channels.

    Returns
    -------
    tuple
            A tuple containing three lists: `normalization_percentile`, `normalization_values`, and `normalization_clip`,
            each of length `n_channels`, representing the interpreted and validated normalization parameters for each channel.

    Raises
    ------
    AssertionError
            If the lengths of the provided lists do not match `n_channels`.

    Examples
    --------
    >>> n_channels = 2
    >>> normalization_percentile = True
    >>> normalization_values = [0.1, 99.9]
    >>> normalization_clip = False
    >>> params = _interpret_normalization_parameters(n_channels, normalization_percentile, normalization_values, normalization_clip)
    >>> print(params)
    # ([True, True], [[0.1, 99.9], [0.1, 99.9]], [False, False])

    """

    if normalization_percentile is None:
        normalization_percentile = [True] * n_channels
    if normalization_values is None:
        normalization_values = [[0.1, 99.9]] * n_channels
    if normalization_clip is None:
        normalization_clip = [False] * n_channels

    if isinstance(normalization_percentile, bool):
        normalization_percentile = [normalization_percentile] * n_channels
    if isinstance(normalization_clip, bool):
        normalization_clip = [normalization_clip] * n_channels
    if len(normalization_values) == 2 and not isinstance(normalization_values[0], list):
        normalization_values = [normalization_values] * n_channels

    assert len(normalization_values) == n_channels
    assert len(normalization_clip) == n_channels
    assert len(normalization_percentile) == n_channels

    return normalization_percentile, normalization_values, normalization_clip


def normalize_signal_set(
    signal_set,
    channel_option,
    percentile_alive=[0.01, 99.99],
    percentile_dead=[0.5, 99.999],
    percentile_generic=[0.01, 99.99],
    normalization_percentile=None,
    normalization_values=None,
    normalization_clip=None,
):
    """
    Normalizes a set of single-cell signals across specified channels using given percentile values or specific normalization parameters.

    This function applies normalization to each channel in the signal set based on the provided normalization parameters,
    which can be defined globally or per channel. The normalization process aims to scale the signal values to a standard
    range, improving the consistency and comparability of signal measurements across samples.

    Parameters
    ----------
    signal_set : ndarray
            A 3D numpy array representing the set of signals to be normalized, with dimensions corresponding to (samples, time points, channels).
    channel_option : list of str
            A list specifying the channels included in the signal set and their corresponding normalization strategy based on channel names.
    percentile_alive : list of float, optional
            The percentile values [lower, upper] used for normalization of signals from channels labeled as 'alive'. Default is [0.01, 99.99].
    percentile_dead : list of float, optional
            The percentile values [lower, upper] used for normalization of signals from channels labeled as 'dead'. Default is [0.5, 99.999].
    percentile_generic : list of float, optional
            The percentile values [lower, upper] used for normalization of signals from channels not specifically labeled as 'alive' or 'dead'.
            Default is [0.01, 99.99].
    normalization_percentile : list of bool or None, optional
            Specifies whether to normalize each channel based on percentile values. If None, the default percentile strategy is applied
            based on `channel_option`. If a list, it should match the length of `channel_option`.
    normalization_values : list of lists or None, optional
            Specifies the percentile values [lower, upper] or fixed values [min, max] for normalization for each channel. Overrides
            `percentile_alive`, `percentile_dead`, and `percentile_generic` if provided.
    normalization_clip : list of bool or None, optional
            Specifies whether to clip the normalized values for each channel to the range [0, 1]. If None, clipping is disabled by default.

    Returns
    -------
    ndarray
            The normalized signal set with the same shape as the input `signal_set`.

    Notes
    -----
    - The function supports different normalization strategies for 'alive', 'dead', and generic signal channels, which can be customized
      via `channel_option` and the percentile parameters.
    - Normalization parameters (`normalization_percentile`, `normalization_values`, `normalization_clip`) are interpreted and validated
      by calling `_interpret_normalization_parameters`.

    Examples
    --------
    >>> signal_set = np.random.rand(100, 128, 2)  # 100 samples, 128 time points, 2 channels
    >>> channel_option = ['alive', 'dead']
    >>> normalized_signals = normalize_signal_set(signal_set, channel_option)
    # Normalizes the signal set based on the default percentile values for 'alive' and 'dead' channels.

    """

    # Check normalization params are ok
    n_channels = len(channel_option)
    normalization_percentile, normalization_values, normalization_clip = (
        _interpret_normalization_parameters(
            n_channels,
            normalization_percentile,
            normalization_values,
            normalization_clip,
        )
    )
    for k, channel in enumerate(channel_option):

        zero_values = []
        for i in range(len(signal_set)):
            zeros_loc = np.where(signal_set[i, :, k] == 0)
            zero_values.append(zeros_loc)

        values = signal_set[:, :, k]

        if normalization_percentile[k]:
            min_val = np.nanpercentile(
                values[values != 0.0], normalization_values[k][0]
            )
            max_val = np.nanpercentile(
                values[values != 0.0], normalization_values[k][1]
            )
        else:
            min_val = normalization_values[k][0]
            max_val = normalization_values[k][1]

        signal_set[:, :, k] -= min_val
        signal_set[:, :, k] /= max_val - min_val

        if normalization_clip[k]:
            signal_set[:, :, k] = np.clip(signal_set[:, :, k], 0.0, 1.0)

        for i, z in enumerate(zero_values):
            signal_set[i, z, k] = 0.0

    return signal_set


def pad_to_model_length(signal_set, model_signal_length):
    """

    Pad the signal set to match the specified model signal length.

    Parameters
    ----------
    signal_set : array-like
            The signal set to be padded.
    model_signal_length : int
            The desired length of the model signal.

    Returns
    -------
    array-like
            The padded signal set.

    Notes
    -----
    This function pads the signal set with zeros along the second dimension (axis 1) to match the specified model signal
    length. The padding is applied to the end of the signals, increasing their length.

    Examples
    --------
    >>> signal_set = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    >>> padded_signals = pad_to_model_length(signal_set, 5)

    """

    padded = np.pad(
        signal_set,
        [(0, 0), (0, model_signal_length - signal_set.shape[1]), (0, 0)],
        mode="edge",
    )

    return padded


def augmenter(
    signal,
    time_of_interest,
    cclass,
    model_signal_length,
    time_shift=True,
    probability=0.95,
):
    """
    Randomly augments single-cell signals to simulate variations in noise, intensity ratios, and event times.

    This function applies random transformations to the input signal, including time shifts, intensity changes,
    and the addition of Gaussian noise, with the aim of increasing the diversity of the dataset for training robust models.

    Parameters
    ----------
    signal : ndarray
            A 1D numpy array representing the signal of a single cell to be augmented.
    time_of_interest : float
            The normalized time of interest (event time) for the signal, scaled to the range [0, 1].
    cclass : ndarray
            A one-hot encoded numpy array representing the class of the cell associated with the signal.
    model_signal_length : int
            The length of the signal expected by the model, used for scaling the time of interest.
    time_shift : bool, optional
            Specifies whether to apply random time shifts to the signal. Default is True.
    probability : float, optional
            The probability with which to apply the augmentation transformations. Default is 0.8.

    Returns
    -------
    tuple
            A tuple containing the augmented signal, the normalized time of interest, and the class of the cell.

    Raises
    ------
    AssertionError
            If the time of interest is provided but invalid for time shifting.

    Notes
    -----
    - Time shifting is not applied to cells of the class labeled as 'miscellaneous' (typically encoded as the class '2').
    - The time of interest is rescaled based on the model's expected signal length before and after any time shift.
    - Augmentation is applied with the specified probability to simulate realistic variability while maintaining
      some original signals in the dataset.

    """

    if np.amax(time_of_interest) <= 1.0:
        time_of_interest *= model_signal_length

    # augment with a certain probability
    r = random.random()
    if r <= probability:

        if time_shift:
            # do not time shift miscellaneous cells
            assert time_of_interest is not None, f"Please provide valid lysis times"
            signal, time_of_interest, cclass = random_time_shift(
                signal, time_of_interest, cclass, model_signal_length
            )

        # signal = random_intensity_change(signal) #maybe bad idea for non percentile-normalized signals
        signal = gauss_noise(signal)

    return signal, time_of_interest / model_signal_length, cclass


def random_intensity_change(signal):
    """

    Randomly change the intensity of a signal.

    Parameters
    ----------
    signal : array-like
            The input signal to be modified.

    Returns
    -------
    array-like
            The modified signal with randomly changed intensity.

    Notes
    -----
    This function applies a random intensity change to each channel of the input signal. The intensity change is
    performed by multiplying each channel with a random value drawn from a uniform distribution between 0.7 and 1.0.

    Examples
    --------
    >>> signal = np.array([[1, 2, 3], [4, 5, 6]])
    >>> modified_signal = random_intensity_change(signal)

    """

    for k in range(signal.shape[1]):
        signal[:, k] = signal[:, k] * np.random.uniform(0.7, 1.0)

    return signal


def gauss_noise(signal):
    """

    Add Gaussian noise to a signal.

    Parameters
    ----------
    signal : array-like
            The input signal to which noise will be added.

    Returns
    -------
    array-like
            The signal with Gaussian noise added.

    Notes
    -----
    This function adds Gaussian noise to the input signal. The noise is generated by drawing random values from a
    standard normal distribution and scaling them by a factor of 0.08 times the input signal. The scaled noise values
    are then added to the original signal.

    Examples
    --------
    >>> signal = np.array([1, 2, 3, 4, 5])
    >>> noisy_signal = gauss_noise(signal)

    """

    sig = 0.08 * np.random.uniform(0, 1)
    signal = signal + sig * np.random.normal(0, 1, signal.shape) * signal
    return signal


def random_time_shift(signal, time_of_interest, cclass, model_signal_length):
    """

    Randomly shift the signals to another time.

    Parameters
    ----------
    signal : array-like
            The signal to be shifted.
    time_of_interest : int or float
            The original time of interest for the signal. Use -1 if not applicable.
    model_signal_length : int
            The length of the model signal.

    Returns
    -------
    array-like
            The shifted fluorescence signal.
    int or float
            The new time of interest if available; otherwise, the original time of interest.

    Notes
    -----
    This function randomly selects a target time within the specified model signal length and shifts the
    signal accordingly. The shift is performed along the first dimension (axis 0) of the signal. The function uses
    nearest-neighbor interpolation for shifting.

    If the original time of interest (`time_of_interest`) is provided (not equal to -1), the function returns the
    shifted signal along with the new time of interest. Otherwise, it returns the shifted signal along with the
    original time of interest.

    The `max_time` is set to the `model_signal_length` unless the original time of interest is provided. In that case,
    `max_time` is set to `model_signal_length - 3` to prevent shifting too close to the edge.

    Examples
    --------
    >>> signal = np.array([[1, 2, 3], [4, 5, 6]])
    >>> shifted_signal, new_time = random_time_shift(signal, 1, 5)

    """

    min_time = 3
    max_time = model_signal_length

    return_target = False
    if time_of_interest != -1:
        return_target = True
        max_time = (
            model_signal_length + 1 / 3 * model_signal_length
        )  # bias to have a third of event class becoming no event
        min_time = -model_signal_length * 1 / 3

    times = np.linspace(
        min_time, max_time, 2000
    )  # symmetrize to create left-censored events
    target_time = np.random.choice(times)

    delta_t = target_time - time_of_interest
    signal = shift(signal, [delta_t, 0], order=0, mode="nearest")

    if target_time <= 0 and np.argmax(cclass) == 0:
        target_time = -1
        cclass = np.array([0.0, 0.0, 1.0]).astype(np.float32)
    if target_time >= model_signal_length and np.argmax(cclass) == 0:
        target_time = -1
        cclass = np.array([0.0, 1.0, 0.0]).astype(np.float32)

    if return_target:
        return signal, target_time, cclass
    else:
        return signal, time_of_interest, cclass
