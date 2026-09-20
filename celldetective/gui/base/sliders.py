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

Last, the float sliders' labels are wide enough for the value with all its decimals. superqt sizes a
label on the text of its bounds and leaves barely a pixel around it, so "0.500" did not fit and lost
its first digit.
"""

from qtpy.QtCore import QLocale
from qtpy.QtGui import QDoubleValidator, QFontMetrics
from superqt import QLabeledDoubleRangeSlider as _QLabeledDoubleRangeSlider
from superqt import QLabeledDoubleSlider as _QLabeledDoubleSlider
from superqt import QLabeledSlider as _QLabeledSlider

from celldetective.gui.base.utils import safe_slider_range


def _c_locale() -> QLocale:
    locale = QLocale.c()
    locale.setNumberOptions(QLocale.RejectGroupSeparator)
    return locale


def _dot_decimal_labels(*labels) -> None:
    """Make each slider label read dot decimals only, whatever the system locale.

    The labels are ``QDoubleSpinBox`` subclasses: they validate and parse the typed text with
    their own locale, so setting it to C rules the decimal separator. That alone still lets a
    comma through as a group separator on the range labels, whose range spans millions, so the
    line edit also gets a C-locale validator. It is deliberately permissive on range and
    precision -- both change as the slider's range does, and the spin box still clamps the value
    when editing ends; its only job is to keep the separators out.
    """
    locale = _c_locale()
    for label in labels:
        if label.locale() != locale:
            label.setLocale(locale)
        line_edit = label.lineEdit()
        if not isinstance(line_edit.validator(), QDoubleValidator):
            validator = QDoubleValidator(-1e18, 1e18, 15, label)
            validator.setLocale(locale)
            line_edit.setValidator(validator)


#: What a spin box needs beyond the text itself: the inner margins of its line
#: edit and the room the blinking cursor sits in.
_LABEL_PADDING = 8


def _fit_decimals(*labels) -> None:
    """Keep each slider label wide enough for its bounds with all their decimals.

    superqt sizes a label from the text of its bounds and leaves only a couple of pixels
    around it, so a value such as "0.500" is drawn right up against the frame and loses its
    first digit under some styles. The sizing pass is wrapped rather than replaced -- it is
    :meth:`_update_size`, which superqt itself calls whenever the range, the precision or the
    mode changes -- and the label is only ever widened, never narrowed.
    """
    for label in labels:
        update_size = getattr(label, "_update_size", None)
        if update_size is None or getattr(update_size, "fits_decimals", False):
            continue

        def _update_size(*args, label=label, update_size=update_size) -> None:
            update_size(*args)
            fm = QFontMetrics(label.font())
            widest = max(
                fm.horizontalAdvance(f"{v:.{label.decimals()}f}"[:18])
                for v in (label.minimum(), label.maximum())
            )
            needed = widest + _LABEL_PADDING
            if label.width() < needed:
                label.setFixedWidth(needed)

        _update_size.fits_decimals = True
        label._update_size = _update_size
        # superqt connected the *original* method to the slider's `rangeChanged`
        # before this wrapper existed, and Qt's own `setRange` does not go through
        # the Python overrides that would call the wrapper instead, so it is
        # connected too -- last, hence after superqt's own sizing pass.
        slider = getattr(label, "_slider", None)
        if slider is not None:
            slider.rangeChanged.connect(_update_size)
        _update_size()


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
