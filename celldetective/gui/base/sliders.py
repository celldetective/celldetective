"""
Float sliders that cannot abort the process on an invalid range.

superqt's float sliders abort the whole Python process (no exception is raised) when their range
is empty, reversed or NaN, which is what the intensity range of a blank or uniform image gives.
These subclasses normalise the range with :func:`safe_slider_range`, so any range set on them, by
a viewer or through :class:`QuickSliderLayout`, is safe. Import the float sliders from here
rather than from superqt; ``tests/gui/test_sliders.py`` enforces it.

They also make the editable value labels accept only a dot as decimal separator. superqt validates
the typed text with the system locale but parses it with :func:`float`, so a comma, accepted under
e.g. a French locale, raised an uncaught ``ValueError``.
"""

from qtpy.QtCore import QLocale
from superqt import QLabeledDoubleRangeSlider as _QLabeledDoubleRangeSlider
from superqt import QLabeledDoubleSlider as _QLabeledDoubleSlider

from celldetective.gui.base.utils import safe_slider_range


def _dot_decimal_labels(*labels) -> None:
    """Make the validator of each slider label follow the C locale (dot decimals, no grouping)."""
    locale = QLocale.c()
    locale.setNumberOptions(QLocale.RejectGroupSeparator)
    for label in labels:
        validator = label.validator()
        if validator is not None and validator.locale() != locale:
            validator.setLocale(locale)


class QLabeledDoubleSlider(_QLabeledDoubleSlider):
    """:class:`superqt.QLabeledDoubleSlider` whose range is always valid."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        _dot_decimal_labels(self._label)

    def setRange(self, min: float, max: float) -> None:
        super().setRange(*safe_slider_range(min, max))


class QLabeledDoubleRangeSlider(_QLabeledDoubleRangeSlider):
    """:class:`superqt.QLabeledDoubleRangeSlider` whose range is always valid."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        _dot_decimal_labels(self._min_label, self._max_label, *self._handle_labels)

    def _on_value_changed(self, v) -> None:
        # superqt recreates the handle labels when the number of handles changes.
        super()._on_value_changed(v)
        _dot_decimal_labels(*self._handle_labels)

    def setRange(self, min: float, max: float) -> None:
        super().setRange(*safe_slider_range(min, max))
