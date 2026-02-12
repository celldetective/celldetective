import numpy as np
import pytest
from celldetective.filters import (
    abs_filter,
    otsu_filter,
    local_filter,
    filter_image,
    gauss_filter,
    median_filter,
    maximum_filter,
    minimum_filter,
    percentile_filter,
    subtract_filter,
    ln_filter,
    variance_filter,
    std_filter,
    laplace_filter,
    dog_filter,
    multiotsu_filter,
    niblack_filter,
    sauvola_filter,
    log_filter,
    tophat_filter,
    invert_filter,
)


@pytest.fixture
def sample_image():
    """Creates a sample 2D image for testing."""
    return np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=float)


def test_abs_filter_extra_args(sample_image):
    """Test abs_filter tolerates extra positional arguments."""
    # Should not raise TypeError
    try:
        abs_filter(sample_image, 10, "dummy")
    except TypeError as e:
        pytest.fail(f"abs_filter raised TypeError with extra args: {e}")


def test_otsu_filter_extra_args(sample_image):
    """Test otsu_filter tolerates extra positional arguments."""
    try:
        otsu_filter(sample_image, 10, "dummy")
    except TypeError as e:
        pytest.fail(f"otsu_filter raised TypeError with extra args: {e}")


def test_local_filter_extra_args(sample_image):
    """Test local_filter tolerates extra positional arguments."""
    try:
        local_filter(sample_image, 10, "dummy")
    except TypeError as e:
        pytest.fail(f"local_filter raised TypeError with extra args: {e}")


def test_filter_image_pipeline(sample_image):
    """Test filter_image pipeline with filters receiving extra arguments."""
    # Define a pipeline where 'abs' receives an extra argument (2)
    filters = [["abs", 2], ["otsu", "extra_arg"]]

    try:
        filter_image(sample_image, filters)
    except TypeError as e:
        pytest.fail(f"filter_image pipeline raised TypeError: {e}")


@pytest.mark.parametrize(
    "filter_func",
    [
        gauss_filter,
        median_filter,
        maximum_filter,
        minimum_filter,
        percentile_filter,
        subtract_filter,
        ln_filter,
        variance_filter,
        std_filter,
        laplace_filter,
        dog_filter,
        multiotsu_filter,
        niblack_filter,
        sauvola_filter,
        log_filter,
        tophat_filter,
        invert_filter,
    ],
)
def test_all_filters_extra_args(filter_func, sample_image):
    """Test standard filters tolerate extra positional arguments."""
    # Prepare minimal valid arguments for specific filters if needed
    kwargs = {}
    args = []

    # Some filters require specific positional arguments or kwargs to run without error regardless of *args
    if filter_func == median_filter:
        args = [3]  # size
    elif filter_func == maximum_filter:
        args = [3]  # size
    elif filter_func == minimum_filter:
        args = [3]  # size
    elif filter_func == percentile_filter:
        args = [50, 3]  # percentile, size
    elif filter_func == subtract_filter:
        args = [10]  # value
    elif filter_func == variance_filter:
        args = [3]  # size
    elif filter_func == std_filter:
        args = [3]  # size
    elif filter_func == log_filter:
        args = [2.0, 1.0, True]  # blob_size, sigma, interpolate
    elif filter_func == dog_filter:
        args = [2.0, 1.0, 2.0, True]  # blob_size, sigma_low, sigma_high, interpolate
    elif filter_func == gauss_filter:
        args = [1.0, True]  # sigma, interpolate
    elif filter_func == ln_filter:
        args = [True]  # interpolate
    elif filter_func == laplace_filter:
        import numpy as np

        args = [float, True]  # output, interpolate
    elif filter_func == multiotsu_filter:
        args = [3]  # classes
    elif filter_func == tophat_filter:
        args = [3, 4, True]  # size, connectivity, interpolate

    # Add dummy extra arguments
    extra_args = [123, "extra"]

    try:
        filter_func(sample_image, *args, *extra_args, **kwargs)
    except TypeError as e:
        pytest.fail(f"{filter_func.__name__} raised TypeError with extra args: {e}")
    except Exception:
        # We only care about TypeError here (argument mismatch).
        # Other errors (e.g., runtime math errors) are acceptable for this specific test of argument handling.
        pass
