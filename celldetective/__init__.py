from ._version import __version__
import os
from .log_manager import setup_global_logging, get_logger

# Define default log path in user home
USER_LOG_DIR = os.path.join(os.path.expanduser("~"), ".celldetective", "logs")
GLOBAL_LOG_FILE = os.path.join(USER_LOG_DIR, "celldetective.log")

# Setup logging
setup_global_logging(log_file=GLOBAL_LOG_FILE)

# Expose logger
logger = get_logger()


def get_software_location() -> str:
    """
    Get the installation folder of celldetective.

    Returns
    -------
    str
            Path to the celldetective installation folder.
    """

    return rf"{os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]}"
