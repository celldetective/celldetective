import logging
import os
import sys
from contextlib import contextmanager

# Default formatters
CONSOLE_FORMAT = "[%(levelname)s] %(message)s"
FILE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def setup_global_logging(level=logging.INFO, log_file=None):
    """
    Sets up the global logger for the application.
    """
    root_logger = logging.getLogger("celldetective")
    root_logger.setLevel(level)
    root_logger.propagate = False  # Prevent double logging if attached to root

    # Clear existing handlers to avoid duplicates on reload
    if root_logger.handlers:
        root_logger.handlers.clear()

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    root_logger.addHandler(console_handler)

    # Optional Global File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
        root_logger.addHandler(file_handler)

    return root_logger

def get_logger(name="celldetective"):
    """
    Returns a logger with the specified name, defaulting to the package logger.
    """
    return logging.getLogger(name)

@contextmanager
def PositionLogger(position_path, logger_name="celldetective"):
    """
    Context manager to dynamically route logs to a file within a specific position folder.
    
    Args:
        position_path (str): Path to the position folder.
        logger_name (str): Name of the logger to attach the handler to.
    """
    logger = logging.getLogger(logger_name)
    
    # Ensure logs/ directory exists in the position folder
    log_dir = os.path.join(position_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "process.log")
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
    
    # Add handler
    logger.addHandler(file_handler)
    
    try:
        yield logger
    finally:
        # Remove handler to stop logging to this file
        file_handler.close()
        logger.removeHandler(file_handler)
