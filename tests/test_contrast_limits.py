"""
Contrast limits must not lean on a NumPy fallback for a lazy stack.

A lazily loaded movie is a dask array, which does not implement
``nanpercentile``. Calling it there used to go through NumPy's fallback, which
dask warns about on every viewer opened and says may stop working.
"""

import logging
import warnings

import numpy as np
import pytest

from celldetective.utils.experiment import _get_contrast_limits

da = pytest.importorskip("dask.array")


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable all logging to avoid Windows OSError with pytest capture."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _stack(seed=0, shape=(3, 64, 64, 2)):
    return np.random.default_rng(seed).random(shape) * 1000


class TestContrastLimits:

    def test_a_dask_stack_raises_no_warning(self):
        stack = da.from_array(_stack(), chunks=(1, 32, 32, 2))
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            limits = _get_contrast_limits(stack)
        assert limits is not None
        assert len(limits) == 2

    def test_a_dask_stack_agrees_with_its_numpy_equivalent(self):
        array = _stack(seed=1)
        expected = _get_contrast_limits(array)
        got = _get_contrast_limits(da.from_array(array, chunks=(1, 32, 32, 2)))
        assert got == pytest.approx(expected)

    def test_the_limits_are_plain_floats(self):
        limits = _get_contrast_limits(da.from_array(_stack(), chunks=(1, 32, 32, 2)))
        for lo, hi in limits:
            assert isinstance(lo, float) and isinstance(hi, float)
            assert lo <= hi

    def test_a_numpy_stack_still_works(self):
        limits = _get_contrast_limits(_stack(shape=(2, 16, 16, 3)))
        assert limits is not None
        assert len(limits) == 3
