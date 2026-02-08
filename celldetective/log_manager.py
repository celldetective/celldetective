import logging
import os
import sys

# from contextlib import contextmanager

# Default formatters
CONSOLE_FORMAT = "[%(levelname)s] %(message)s"
FILE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


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

    # Clear existing handlers to avoid duplicates on reload
    if root_logger.handlers:
        root_logger.handlers.clear()

    # Console Handler
    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    root_logger.addHandler(console_handler)

    # Optional Global File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
        root_logger.addHandler(file_handler)

        for lib in ["trackpy", "btrack", "cellpose", "stardist"]:
            lib_logger = logging.getLogger(lib)
            lib_logger.setLevel(logging.INFO)
            if file_handler not in lib_logger.handlers:
                lib_logger.addHandler(file_handler)

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


# @contextmanager
# def positionlogger(position_path, logger_name="celldetective"):
#     """
#     context manager to dynamically route logs to a file within a specific position folder.
#
#     args:
#         position_path (str): path to the position folder.
#         logger_name (str): name of the logger to attach the handler to.
#     """
#     logger = logging.getlogger(logger_name)
#
#     # ensure logs/ directory exists in the position folder
#     log_dir = os.path.join(position_path, "logs")
#     os.makedirs(log_dir, exist_ok=true)
#
#     log_file = os.path.join(log_dir, "process.log")
#
#     # create file handler
#     file_handler = logging.filehandler(log_file)
#     file_handler.setformatter(logging.formatter(file_format))
#
#     # add handler
#     logger.addhandler(file_handler)
#
#     try:
#         yield logger
#     finally:
#         # remove handler to stop logging to this file
#         file_handler.close()
#         logger.removehandler(file_handler)
