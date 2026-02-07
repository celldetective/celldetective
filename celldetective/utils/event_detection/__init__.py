from celldetective.event_detection_models import SignalDetectionModel
from celldetective.utils.model_loaders import locate_signal_model


def _prep_event_detection_model(model_name=None, use_gpu=True):
    """
    Prepare event detection model for inference.

    Parameters
    ----------
    model_name : str, optional
        Name of the model to load. Default is None.
    use_gpu : bool, optional
        If True, use GPU for inference. Default is True.

    Returns
    -------
    SignalDetectionModel
        Loaded event detection model.
    """
    model_path = locate_signal_model(model_name)
    signal_model = SignalDetectionModel(pretrained=model_path)
    return signal_model
