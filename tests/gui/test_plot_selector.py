import pytest
from PyQt5.QtWidgets import QApplication
from celldetective.gui.base.plot_selector import (
    PlotSelectorWidget,
    StatsSelectorWidget,
    SelectableCard,
)
import os


def test_plot_selector_widget(qtbot):
    widget = PlotSelectorWidget()
    if qtbot:
        qtbot.addWidget(widget)

    # Check if cards are created
    assert len(widget.cards) == 11
    assert "histogram" in widget.cards
    assert "boxenplot" in widget.cards

    # Test selection
    card_hist = widget.cards["histogram"]

    # Test internal logic
    card_hist.setChecked(True)
    assert card_hist.isChecked()
    assert "histogram" in widget.get_selection()

    card_hist.setChecked(False)
    assert not card_hist.isChecked()
    assert "histogram" not in widget.get_selection()

    # Test multiple selection
    widget.cards["histogram"].setChecked(True)
    widget.cards["KDE plot"].setChecked(True)

    selection = widget.get_selection()
    assert "histogram" in selection
    assert "KDE plot" in selection
    assert len(selection) == 2


def test_stats_selector_widget(qtbot):
    widget = StatsSelectorWidget()
    if qtbot:
        qtbot.addWidget(widget)

    # Check if cards are created
    assert len(widget.cards) == 2
    assert "Compute KS test\np-value?" in widget.cards
    assert "Compute effect size?\n(Cliff's Delta)" in widget.cards

    # Test selection
    card = widget.cards["Compute KS test\np-value?"]
    card.setChecked(True)
    assert card.isChecked()
    assert "Compute KS test\np-value?" in widget.get_selection()

    card.setChecked(False)
    assert not card.isChecked()
    assert "Compute KS test\np-value?" not in widget.get_selection()


if __name__ == "__main__":
    app = QApplication([])
    test_plot_selector_widget(None)
    test_stats_selector_widget(None)
    print("Test passed!")
