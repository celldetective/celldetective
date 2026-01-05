from celldetective.signals import SignalDetectionModel
from celldetective.utils.model_loaders import locate_signal_model

def _prep_event_detection_model(model_name=None, use_gpu=True):
    model_path = locate_signal_model(model_name)
    signal_model = SignalDetectionModel(pretrained=model_path)
    return signal_model

