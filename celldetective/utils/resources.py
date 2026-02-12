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