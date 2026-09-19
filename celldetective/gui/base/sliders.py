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

The integer :class:`QLabeledSlider` is wrapped too: superqt gives its label the range of the
slider when it is created (0-99) and never updates it, so any value above 99 was shown as 99.

Last, the float sliders' labels are wide enough for the value with all its decimals. superqt sizes a label
from ``str(minimum)`` and ``str(maximum)`` ("1.0"), so "0.500" did not fit and lost its first
digit.
"""

from qtpy.QtCore import QLocale, QSize
from qtpy.QtGui import QFontMetrics
from superqt import QLabeledDoubleRangeSlider as _QLabeledDoubleRangeSlider
from superqt import QLabeledDoubleSlider as _QLabeledDoubleSlider
from superqt import QLabeledSlider as _QLabeledSlider

from celldetective.gui.base.utils import safe_slider_range


def _dot_decimal_labels(*labels) -> None:
    """Make the validator of each slider label follow the C locale (dot decimals, no grouping)."""
    locale = QLocale.c()
    locale.setNumberOptions(QLocale.RejectGroupSeparator)
    for label in labels:
        validator = label.validator()
        if validator is not None and validator.locale() != locale:
            validator.setLocale(locale)


def _fit_decimals(*labels) -> None:
    """Widen each slider label by what its decimals add to the bounds superqt measures."""
    for label in labels:
        get_size = getattr(label, "_get_size", None)
        if get_size is None or getattr(get_size, "fits_decimals", False):
            continue

        def _get_size(label=label, get_size=get_size) -> QSize:
            size = get_size()
            fm = QFontMetrics(label.font())
            bounds = (label.minimum(), label.maximum())
            shown = max(fm.horizontalAdvance(f"{v:.{label.decimals()}f}") for v in bounds)
            measured = max(fm.horizontalAdvance(str(v)[:18]) for v in bounds)
            return QSize(size.width() + max(0, shown - measured), size.height())

        _get_size.fits_decimals = True
        label._get_size = _get_size
        label._update_size()


class QLabeledSlider(_QLabeledSlider):
    """:class:`superqt.QLabeledSlider` whose label follows the range of the slider."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._slider.rangeChanged.connect(self._sync_label_range)
        self._sync_label_range(self.minimum(), self.maximum())

    def _sync_label_range(self, min_: int, max_: int) -> None:
        self._label.setRange(min_, max_)
        # superqt sized the label for the previous range, before this slot ran.
        self._label._update_size()
        self._label.setValue(self.value())


class QLabeledDoubleSlider(_QLabeledDoubleSlider):
    """:class:`superqt.QLabeledDoubleSlider` whose range is always valid."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        _dot_decimal_labels(self._label)
        _fit_decimals(self._label)

    def setDecimals(self, prec: int) -> None:
        super().setDecimals(prec)
        self._label._update_size()

    def setRange(self, min: float, max: float) -> None:
        super().setRange(*safe_slider_range(min, max))


class QLabeledDoubleRangeSlider(_QLabeledDoubleRangeSlider):
    """:class:`superqt.QLabeledDoubleRangeSlider` whose range is always valid."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        _dot_decimal_labels(self._min_label, self._max_label, *self._handle_labels)
        _fit_decimals(self._min_label, self._max_label)

    def setDecimals(self, prec: int) -> None:
        super().setDecimals(prec)
        self._min_label._update_size()
        self._max_label._update_size()

    def _on_value_changed(self, v) -> None:
        # superqt recreates the handle labels when the number of handles changes.
        super()._on_value_changed(v)
        _dot_decimal_labels(*self._handle_labels)

    def setRange(self, min: float, max: float) -> None:
        super().setRange(*safe_slider_range(min, max))
