"""
Unit tests for the shared model channel-selection widget.

The widget maps a model's declared input slots onto the experiment's own
channels, and is used both by the main window's channel dialog and by the napari
single-frame panel. What is pinned here is the seeding order -- stored mapping,
then name match, then None -- because that is what decides which channel a model
is actually fed when nobody touches the dropdowns.
"""

import logging

import pytest
from PyQt5.QtWidgets import QLabel

from celldetective.gui.base.model_channel_selection import (
    NO_CHANNEL,
    ModelChannelSelection,
)


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable all logging to avoid Windows OSError with pytest capture."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


EXP_CHANNELS = ["brightfield_channel", "live_nuclei_channel", "dead_nuclei_channel"]


def _widget(qtbot, required, selected=None, available=None, **kwargs):
    """Build a widget and hand it to qtbot so it is destroyed with the test."""
    widget = ModelChannelSelection(
        required_channels=required,
        available_channels=EXP_CHANNELS if available is None else available,
        selected_channels=selected,
        **kwargs,
    )
    qtbot.addWidget(widget)
    return widget


class TestRows:
    """One dropdown per input slot, offering every channel plus None."""

    def test_one_dropdown_per_slot(self, qtbot):
        widget = _widget(qtbot, ["brightfield_channel", "live_nuclei_channel"])
        assert len(widget.channel_cbs) == 2

    def test_every_channel_is_offered_plus_none(self, qtbot):
        widget = _widget(qtbot, ["brightfield_channel"])
        combo = widget.channel_cbs[0]
        offered = [combo.itemText(i) for i in range(combo.count())]
        assert offered == EXP_CHANNELS + [NO_CHANNEL]

    def test_a_model_without_inputs_builds_anyway(self, qtbot):
        widget = _widget(qtbot, [])
        assert widget.channel_cbs == []
        assert widget.selected_channels() == []

    def test_an_experiment_without_channels_still_offers_none(self, qtbot):
        widget = _widget(qtbot, ["brightfield_channel"], available=[])
        assert widget.selected_channels() == [NO_CHANNEL]


class TestSeeding:
    """
    The stored mapping wins, then a name match, then None.

    Models such as CP_cyto3 declare channel names no real experiment uses; without
    the stored mapping every slot would open on None and the model could not be
    run at all.
    """

    def test_a_slot_named_after_a_channel_opens_on_it(self, qtbot):
        widget = _widget(qtbot, ["live_nuclei_channel"])
        assert widget.selected_channels() == ["live_nuclei_channel"]

    def test_an_unknown_slot_opens_on_none(self, qtbot):
        widget = _widget(qtbot, ["fluorescenceuv"])
        assert widget.selected_channels() == [NO_CHANNEL]

    def test_the_stored_mapping_is_preferred(self, qtbot):
        widget = _widget(
            qtbot,
            ["fluorescenceuv", "None"],
            selected=["dead_nuclei_channel", NO_CHANNEL],
        )
        assert widget.selected_channels() == ["dead_nuclei_channel", NO_CHANNEL]

    def test_the_stored_mapping_overrides_a_name_match(self, qtbot):
        widget = _widget(
            qtbot, ["live_nuclei_channel"], selected=["brightfield_channel"]
        )
        assert widget.selected_channels() == ["brightfield_channel"]

    def test_a_stale_entry_falls_back_slot_by_slot(self, qtbot):
        # The first entry names a channel this experiment does not have, so that
        # slot falls back to its own name match; the second is honoured.
        widget = _widget(
            qtbot,
            ["live_nuclei_channel", "brightfield_channel"],
            selected=["gone_channel", "dead_nuclei_channel"],
        )
        assert widget.selected_channels() == [
            "live_nuclei_channel",
            "dead_nuclei_channel",
        ]

    def test_a_short_mapping_leaves_the_remaining_slots_to_the_name_match(self, qtbot):
        widget = _widget(
            qtbot,
            ["fluorescenceuv", "live_nuclei_channel"],
            selected=["brightfield_channel"],
        )
        assert widget.selected_channels() == [
            "brightfield_channel",
            "live_nuclei_channel",
        ]


class TestReporting:
    """What the widget reports back, and what the caller does with it."""

    def test_selected_channels_follows_the_dropdowns(self, qtbot):
        widget = _widget(qtbot, ["brightfield_channel"])
        widget.channel_cbs[0].setCurrentIndex(
            widget.channel_cbs[0].findText("dead_nuclei_channel")
        )
        assert widget.selected_channels() == ["dead_nuclei_channel"]

    def test_is_empty_when_every_slot_is_none(self, qtbot):
        widget = _widget(qtbot, ["fluorescenceuv", "fluorescenceir"])
        assert widget.is_empty()

    def test_is_not_empty_when_one_slot_is_fed(self, qtbot):
        widget = _widget(qtbot, ["fluorescenceuv", "live_nuclei_channel"])
        assert not widget.is_empty()

    def test_a_model_without_inputs_is_not_reported_as_empty(self, qtbot):
        # Nothing to assign is not the same as everything left unassigned: the
        # caller must not refuse to run such a model.
        widget = _widget(qtbot, [])
        assert not widget.is_empty()


class TestPresentation:
    """The requirement label is optional, the dropdowns are not."""

    def _labels(self, widget):
        return [label.text() for label in widget.findChildren(QLabel)]

    def test_the_slot_the_dropdown_feeds_is_printed(self, qtbot):
        widget = _widget(qtbot, ["fluorescenceuv"])
        assert "Req: fluorescenceuv" in self._labels(widget)

    def test_the_requirement_can_be_hidden(self, qtbot):
        widget = _widget(qtbot, ["fluorescenceuv"], show_requirement=False)
        assert "Req: fluorescenceuv" not in self._labels(widget)
        assert len(widget.channel_cbs) == 1
