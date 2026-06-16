import logging
import os
import sys
import time
from typing import Any, Optional, Iterator
from contextlib import contextmanager

# Default formatters
CONSOLE_FORMAT = "[%(levelname)s] %(message)s"
FILE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Third-party loggers whose output we want to surface alongside our own
LIBRARY_LOGGERS = ("trackpy", "btrack", "cellpose", "stardist")


def setup_global_logging(
    level: int = logging.INFO, log_file: Optional[str] = None
) -> logging.Logger:
    """
    Sets up the global logger for the application.

    Parameters
    ----------
    level : int, optional
            The logging level (e.g., logging.INFO, logging.DEBUG). Default is logging.INFO.
    log_file : str, optional
            Path to a file where logs should be saved. If None, logs are only output to the console. Default is None.

    Returns
    -------
    logging.Logger
            The configured root logger.
    """
    root_logger = logging.getLogger("celldetective")
    root_logger.setLevel(level)
    root_logger.propagate = False  # Prevent double logging if attached to root

    # Clear existing handlers to avoid duplicates on reload. The library loggers must be
    # cleared too: a fresh console/file handler is built on every call, so without this
    # repeated calls would accumulate duplicate handlers on them (duplicate log lines).
    if root_logger.handlers:
        root_logger.handlers.clear()
    for lib in LIBRARY_LOGGERS:
        logging.getLogger(lib).handlers.clear()

    # Console Handler
    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    root_logger.addHandler(console_handler)

    # Always forward library logs to the console
    for lib in LIBRARY_LOGGERS:
        lib_logger = logging.getLogger(lib)
        lib_logger.setLevel(logging.INFO)
        lib_logger.addHandler(console_handler)

    # Optional Global File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
        root_logger.addHandler(file_handler)

        for lib in LIBRARY_LOGGERS:
            logging.getLogger(lib).addHandler(file_handler)

    # Hook to capture uncaught exceptions
    def handle_exception(
        exc_type: type, exc_value: Exception, exc_traceback: Any
    ) -> None:
        """
        Handle uncaught exceptions.

        Parameters
        ----------
        exc_type : type
            Exception type.
        exc_value : Exception
            Exception value.
        exc_traceback : traceback
            Exception traceback.
        """
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        root_logger.error(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_exception

    return root_logger


def cleanup_old_logs(log_dir: str, max_age_days: int = 30) -> None:
    """
    Delete ``.log`` files in ``log_dir`` whose last modification is older than ``max_age_days``.

    Called once per session (by the first process) to keep the log directory from growing
    without bound. The current session's freshly created log file is always newer than the
    cutoff, so it is never removed.

    Parameters
    ----------
    log_dir : str
        Directory containing the log files.
    max_age_days : int, optional
        Files older than this many days are deleted. Default is 30.
    """
    if not os.path.isdir(log_dir):
        return

    cutoff = time.time() - max_age_days * 86400
    for name in os.listdir(log_dir):
        if not name.endswith(".log"):
            continue
        path = os.path.join(log_dir, name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError as e:
            logging.getLogger("celldetective").warning(
                f"Could not delete old log file {path}: {e}"
            )


def get_logger(name: str = "celldetective") -> logging.Logger:
    """
    Returns a logger with the specified name, defaulting to the package logger.

    Parameters
    ----------
    name : str, optional
            The name of the logger to retrieve. Default is "celldetective".

    Returns
    -------
    logging.Logger
            The requested logger.
    """
    return logging.getLogger(name)


class QueueLoggingHandler(logging.Handler):
    """
    Forward log records to a multiprocessing queue.

    Worker processes (segmentation, tracking, measurement, signal analysis) run in their
    own spawned process, so their logs — and those of the libraries they call (cellpose,
    stardist, btrack, trackpy) — never reach the parent process by default. This handler
    ships each record to the parent through the existing progress queue so it can be
    re-emitted there and surface where the user is watching.

    The payload is a plain dict (``{"log_record": {...}}``) so it survives pickling and is
    easy for the parent-side reader to recognise.
    """

    def __init__(self, queue: Any) -> None:
        super().__init__()
        self.queue = queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put(
                {
                    "log_record": {
                        "name": record.name,
                        "levelno": record.levelno,
                        "msg": record.getMessage(),
                    }
                }
            )
        except Exception:
            # Never let logging break the worker
            pass


@contextmanager
def forward_logs_to_queue(
    queue: Any, logger_names: Iterator[str] = ("celldetective",) + LIBRARY_LOGGERS
) -> Iterator[None]:
    """
    Route records from the given loggers to ``queue`` only, for the duration of the block.

    Intended to run inside a worker child process. While active, each named logger emits
    *exclusively* to the queue (its own console/file handlers are detached and restored on
    exit), so the parent process becomes the single writer once it re-emits the records —
    no duplicate lines in the global log file. The library loggers are top-level (not
    children of ``celldetective``), so they must be listed explicitly; they do not
    propagate to the package logger.

    Parameters
    ----------
    queue : multiprocessing.Queue
        The queue back to the parent process.
    logger_names : iterable of str
        Loggers to forward. Defaults to the package logger plus the third-party libraries.
    """
    handler = QueueLoggingHandler(queue)
    handler.setLevel(logging.INFO)
    saved = {}
    for name in logger_names:
        lg = logging.getLogger(name)
        saved[name] = (lg.handlers[:], lg.propagate)
        lg.handlers = [handler]
        lg.propagate = False
    try:
        yield
    finally:
        for name, (handlers, propagate) in saved.items():
            lg = logging.getLogger(name)
            lg.handlers = handlers
            lg.propagate = propagate
        handler.close()


@contextmanager
def capture_library_logs(
    position_path: str,
    filename: str,
    logger_names: Iterator[str] = LIBRARY_LOGGERS,
) -> Iterator[None]:
    """
    Additionally write third-party library logs into a position-level file.

    Wraps the processing of one position so that records from the segmentation/tracking
    libraries (cellpose, stardist, btrack, trackpy) are appended to
    ``<position_path>/<filename>`` — making each position's ``log_{mode}.txt`` self-contained.

    Only the library loggers are attached (not ``celldetective``): the package logger's
    manifest lines are already written by :func:`positionlogger` inside ``write_log``, so
    including it here would duplicate them. The handler is added alongside any existing
    handlers (e.g. the queue forwarder) and removed on exit.

    Parameters
    ----------
    position_path : str
        Path to the position folder.
    filename : str
        Name of the log file inside the position folder (e.g. ``log_effectors.txt``).
    logger_names : iterable of str
        Library loggers to capture. Defaults to :data:`LIBRARY_LOGGERS`.
    """
    log_file = os.path.join(position_path, filename)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
    loggers = [logging.getLogger(name) for name in logger_names]
    for lg in loggers:
        lg.addHandler(file_handler)
    try:
        yield
    finally:
        for lg in loggers:
            lg.removeHandler(file_handler)
        file_handler.close()


@contextmanager
def positionlogger(
    position_path: str,
    filename: str = "log_preprocessing.txt",
    logger_name: str = "celldetective",
) -> Iterator[logging.Logger]:
    """
    Context manager to route logs to a file within a specific position folder.

    While the context is active, every record emitted through the ``celldetective``
    logger (including its child module loggers) is additionally written to
    ``<position_path>/<filename>`` in append mode, using the standard file format.
    The handler is removed on exit, so the routing is scoped to the ``with`` block.

    Parameters
    ----------
    position_path : str
        Path to the position folder.
    filename : str, optional
        Name of the log file inside the position folder. Default "log_preprocessing.txt".
    logger_name : str, optional
        Name of the logger to attach the handler to. Default "celldetective".

    Yields
    ------
    logging.Logger
        The logger with the position-scoped file handler attached.
    """
    logger = logging.getLogger(logger_name)
    log_file = os.path.join(position_path, filename)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
    logger.addHandler(file_handler)

    try:
        yield logger
    finally:
        file_handler.close()
        logger.removeHandler(file_handler)
