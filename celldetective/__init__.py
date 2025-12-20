from ._version import __version__
import os
from .log_manager import setup_global_logging, get_logger
# from .signals import SignalDetectionModel, ResNetModelCurrent, analyze_signals

def __getattr__(name):
	if name in ('SignalDetectionModel', 'ResNetModelCurrent', 'analyze_signals'):
		from . import signals
		return getattr(signals, name)
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Define default log path in user home
USER_LOG_DIR = os.path.join(os.path.expanduser("~"), ".celldetective", "logs")
GLOBAL_LOG_FILE = os.path.join(USER_LOG_DIR, "celldetective.log")

# Setup logging
setup_global_logging(log_file=GLOBAL_LOG_FILE)

# Expose logger
logger = get_logger()
