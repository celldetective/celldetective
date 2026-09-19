from ._version import __version__
import os
import datetime
from importlib.resources import files as _files

# Prevent matplotlib circular import issue with partially initialized IPython
try:
    import IPython
except ImportError:
    pass

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


def get_package_location() -> str:
    """
    Get the celldetective package folder, the one holding the shipped data
    files (icons, models, help pages, configuration templates).

    Prefer this over `get_software_location`: it points straight at the package
    instead of its parent, so call sites do not have to append "celldetective"
    back onto the path.

    Returns
    -------
    str
            Path to the celldetective package folder.
    """

    try:
        return str(_files("celldetective"))
    except Exception:
        # Frozen bundles and exotic loaders may not expose a Traversable for
        # the package; fall back to locating this very module on disk.
        return os.path.dirname(os.path.realpath(__file__))


def get_software_location() -> str:
    """
    Get the installation folder of celldetective, i.e. the parent of the
    package folder.

    Call sites append "celldetective" to the returned path to reach the shipped
    data files; `get_package_location` gets there directly and is preferred for
    new code.

    Returns
    -------
    str
            Path to the celldetective installation folder.
    """

    return os.path.dirname(get_package_location())
