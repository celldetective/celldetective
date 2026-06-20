import os

from celldetective.log_manager import get_logger

logger = get_logger(__name__)


def resolve_gpu_device(default: str = "0") -> str:
    """
    Resolve which GPU device index to expose to a worker.

    Reads the ``CELLDETECTIVE_GPU_DEVICE`` environment variable so multi-GPU
    machines can pick a device other than the hardcoded first one. The value is
    used as ``CUDA_VISIBLE_DEVICES`` and so may be a single index ("0", "1") or
    a comma-separated list ("0,1").

    Parameters
    ----------
    default : str, optional
        Device string to use when the environment variable is unset. Default "0".

    Returns
    -------
    str
        The device string to assign to ``CUDA_VISIBLE_DEVICES``.
    """
    device = os.environ.get("CELLDETECTIVE_GPU_DEVICE", "").strip()
    if device == "":
        return default
    return device


def configure_memory_growth():
    """
    Enable TensorFlow GPU memory growth so TF does not pre-allocate all VRAM.

    This must run *before* any TensorFlow op first touches the GPU; once the GPU
    context is initialized the setting is rejected by TF (and the resulting
    error is swallowed here). Calling it early in a worker process — before the
    first model is built — keeps a heavy model (e.g. StarDist) from grabbing all
    the device memory and starving a later model (e.g. event detection) that
    runs in the same process during a combined pipeline.

    Safe to call when no GPU is visible (``CUDA_VISIBLE_DEVICES=-1``) or when
    TensorFlow is not installed: it becomes a no-op.

    Returns
    -------
    bool
        True if at least one GPU was found and configured, False otherwise.
    """
    try:
        from tensorflow.config import list_physical_devices
        from tensorflow.config.experimental import set_memory_growth
    except Exception as e:
        logger.debug(f"TensorFlow unavailable; skipping memory-growth setup: {e}")
        return False

    try:
        gpus = list_physical_devices("GPU")
        for gpu in gpus:
            set_memory_growth(gpu, True)
        if gpus:
            logger.info(f"Configured memory growth on {len(gpus)} GPU(s).")
        return bool(gpus)
    except Exception as e:
        logger.debug(f"GPU memory growth configuration failed: {e}")
        return False


def auto_find_gpu():
    """
    Automatically detects the presence of GPU devices in the system.

    This function checks if any GPU devices are available for use by querying the system's physical devices.
    It is a utility function to simplify the process of determining whether GPU-accelerated computing can be
    leveraged in data processing or model training tasks.

    Returns
    -------
    bool
            True if one or more GPU devices are detected, False otherwise.

    Notes
    -----
    - The function uses TensorFlow's `list_physical_devices` method to query available devices, specifically
      looking for 'GPU' devices.
    - This function is useful for dynamically adjusting computation strategies based on available hardware resources.

    Examples
    --------
    >>> has_gpu = auto_find_gpu()
    >>> print(f"GPU available: {has_gpu}")
    # GPU available: True or False based on the system's hardware configuration.
    """
    from tensorflow.config import list_physical_devices

    gpus = list_physical_devices("GPU")
    if len(gpus) > 0:
        use_gpu = True
    else:
        use_gpu = False

    return use_gpu