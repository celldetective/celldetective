from ._version import __version__
import os
import datetime
from .log_manager import setup_global_logging, get_logger, cleanup_old_logs

# Define default log directory in user home
USER_LOG_DIR = os.path.join(os.path.expanduser("~"), ".celldetective", "logs")

# Use one log file per session rather than a single ever-growing file. Worker processes
# spawned for segmentation/tracking/measurement re-import this package, so they inherit
# CELLDETECTIVE_SESSION_LOG and log to the *same* session file instead of each creating a
# new one. Only the first (parent) process mints the path and prunes stale logs.
_session_log = os.environ.get("CELLDETECTIVE_SESSION_LOG")
_new_session = _session_log is None
if _new_session:
    _timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _session_log = os.path.join(
        USER_LOG_DIR, f"celldetective_{_timestamp}_{os.getpid()}.log"
    )
    os.environ["CELLDETECTIVE_SESSION_LOG"] = _session_log

GLOBAL_LOG_FILE = _session_log

# Setup logging
setup_global_logging(log_file=GLOBAL_LOG_FILE)

# Prune logs older than a month (once per session, after the log dir is guaranteed to exist)
if _new_session:
    cleanup_old_logs(USER_LOG_DIR, max_age_days=30)

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
