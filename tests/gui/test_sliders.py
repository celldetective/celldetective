import ast
from pathlib import Path

import pytest

import celldetective
from celldetective.gui.base.sliders import (
    QLabeledDoubleRangeSlider,
    QLabeledDoubleSlider,
    QLabeledSlider,
)

GUI_DIR = Path(celldetective.__file__).parent / "gui"
FLOAT_SLIDERS = {"QLabeledDoubleSlider", "QLabeledDoubleRangeSlider", "QLabeledSlider"}


@pytest.mark.parametrize("slider_class", [QLabeledDoubleSlider, QLabeledDoubleRangeSlider])
@pytest.mark.parametrize(
    "bounds, expected",
    [
        ((5.0, 5.0), (5.0, 6.0)),  # uniform image
        ((3.0, 1.0), (3.0, 4.0)),  # reversed
        ((float("nan"), float("nan")), (0.0, 1.0)),  # all-NaN image
        ((0.0, float("nan")), (0.0, 1.0)),
        ((0.0, 10.0), (0.0, 10.0)),  # valid range is untouched
    ],
)
def test_float_sliders_survive_invalid_ranges(qtbot, slider_class, bounds, expected):
    # The superqt originals abort the process on these ranges.
    slider = slider_class()
    qtbot.addWidget(slider)
    slider.setRange(*bounds)
    slider.show()
    assert (slider.minimum(), slider.maximum()) == pytest.approx(expected)


def test_float_slider_label_fits_its_decimals(qtbot):
    # superqt sized the label for "1.0": "0.500" lost its first digit.
    slider = QLabeledDoubleSlider()
    qtbot.addWidget(slider)
    slider.setDecimals(3)
    slider.setRange(0, 1)
    slider.setValue(0.5)
    slider.show()
    label = slider._label
    assert label.text() == "0.500"
    # the text plus the line edit's inner margins and the cursor
    assert label.width() >= label.fontMetrics().horizontalAdvance("0.500") + 8


def test_int_slider_label_follows_range(qtbot):
    # superqt kept the label at the slider's initial range (0-99): 300 epochs showed 99.
    slider = QLabeledSlider()
    qtbot.addWidget(slider)
    slider.setRange(1, 3000)
    slider.setValue(300)
    assert slider._label.text() == "300"
    slider.setRange(0, 50)
    assert slider.value() == 50
    assert slider._label.text() == "50"


def test_gui_imports_float_sliders_from_celldetective():
    offenders = []
    for path in GUI_DIR.rglob("*.py"):
        if path.name == "sliders.py" and path.parent.name == "base":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module == "superqt":
                if FLOAT_SLIDERS & {alias.name for alias in node.names}:
                    offenders.append(str(path.relative_to(GUI_DIR)))
    assert not offenders, (
        "Import QLabeledDoubleSlider / QLabeledDoubleRangeSlider from "
        f"celldetective.gui.base.sliders, not superqt: {offenders}"
    )


@pytest.mark.parametrize("slider_class", [QLabeledDoubleSlider, QLabeledDoubleRangeSlider])
def test_float_slider_labels_reject_comma_decimals(qtbot, slider_class):
    # superqt parses the label text with float(), which fails on a locale comma.
    from qtpy.QtCore import QLocale
    from qtpy.QtGui import QValidator

    QLocale.setDefault(QLocale(QLocale.French))
    try:
        slider = slider_class()
        qtbot.addWidget(slider)
        slider.setRange(0.0, 2.0)
        if slider_class is QLabeledDoubleRangeSlider:
            slider.setValue((0.25, 1.0, 1.5))  # new handle labels are created
            labels = [slider._min_label, slider._max_label, *slider._handle_labels]
        else:
            labels = [slider._label]
        for label in labels:
            # superqt turned SliderLabel from a QDoubleSpinBox, which owns a line
            # edit and parses the text itself, into a QLineEdit, which is one and
            # parses with float(); the package pins no version, so both are read.
            line_edit = label.lineEdit() if hasattr(label, "lineEdit") else label
            validator = line_edit.validator()
            assert validator.validate("0,75", 4)[0] == QValidator.Invalid
            assert validator.validate("0.75", 4)[0] == QValidator.Acceptable
            if hasattr(label, "valueFromText") and label.decimals() >= 2:
                assert label.valueFromText("0.75") == 0.75
    finally:
        QLocale.setDefault(QLocale.c())
