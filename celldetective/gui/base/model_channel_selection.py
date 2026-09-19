"""
The channel-selection rows shared by every place a model's inputs are mapped.

A deep-learning model declares the inputs it was trained on -- ``brightfield``,
``live_nuclei``, ... -- and the experiment has channels of its own, named by
whoever acquired them. Mapping one onto the other is the same job whether it is
done in the main window before a full-stack run or in the napari viewer on a
single frame, so it is done here once.
"""

from typing import List, Optional, Sequence

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from celldetective.gui.base.components import CelldetectiveWidget

#: What an unused input slot is called, in the dropdowns and in the
#: ``selected_channels`` mapping written to a model's ``config_input.json``.
NO_CHANNEL = "None"


class ModelChannelSelection(CelldetectiveWidget):
    """
    One dropdown per model input slot, listing the experiment's channels.

    The rows are seeded in decreasing order of specificity: the mapping already
    chosen for this model, then a plain name match between the slot and an
    experiment channel, then :data:`NO_CHANNEL`. Nothing is written anywhere --
    the widget only reports what is selected, and the caller decides whether that
    is worth persisting.

    Parameters
    ----------
    required_channels : sequence of str
        The model's input slots, in the order it expects them, as read from the
        ``channels`` entry of its ``config_input.json``.
    available_channels : sequence of str
        The experiment's channel names.
    selected_channels : sequence of str, optional
        A mapping to seed the dropdowns with, typically the ``selected_channels``
        entry of the model configuration. Entries that do not name an available
        channel are ignored, one by one, so a stale mapping degrades to a name
        match rather than being dropped whole.
    show_requirement : bool, optional
        Whether to print the slot the dropdown feeds above it. True by default.
    parent : QWidget, optional
        The parent widget.

    Attributes
    ----------
    channel_cbs : list of QComboBox
        The dropdowns, one per input slot, in the model's own order.
    """

    def __init__(
        self,
        required_channels: Sequence[str],
        available_channels: Sequence[str],
        selected_channels: Optional[Sequence[str]] = None,
        show_requirement: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:

        super().__init__(parent)

        self.required_channels = [str(c) for c in (required_channels or [])]
        self.available_channels = [str(c) for c in (available_channels or [])]
        self.show_requirement = show_requirement
        self.options = self.available_channels + [NO_CHANNEL]
        self.channel_cbs: List[QComboBox] = []

        self.channel_layout = QVBoxLayout(self)
        self.channel_layout.setContentsMargins(0, 0, 0, 0)
        self.populate_widgets(selected_channels)

    def populate_widgets(self, selected_channels: Optional[Sequence[str]]) -> None:
        """
        Build one row per input slot.

        Parameters
        ----------
        selected_channels : sequence of str or None
            The mapping to seed the dropdowns with.
        """

        if not self.required_channels:
            self.channel_layout.addWidget(
                QLabel("This model declares no input channels.")
            )
            return

        for k, slot in enumerate(self.required_channels):

            combo = QComboBox()
            combo.addItems(self.options)
            combo.setToolTip(f"Feeds the model's '{slot}' input.")
            combo.setCurrentIndex(
                self._default_index(combo, slot, selected_channels, k)
            )
            self.channel_cbs.append(combo)

            hbox_channel = QHBoxLayout()
            hbox_channel.addWidget(QLabel(f"channel {k+1}: "), 33)
            if self.show_requirement:
                ch_vbox = QVBoxLayout()
                ch_vbox.addWidget(QLabel(f"Req: {slot}"), alignment=Qt.AlignLeft)
                ch_vbox.addWidget(combo)
                hbox_channel.addLayout(ch_vbox, 66)
            else:
                hbox_channel.addWidget(combo, 66)
            self.channel_layout.addLayout(hbox_channel)

    def _default_index(
        self,
        combo: QComboBox,
        slot: str,
        selected_channels: Optional[Sequence[str]],
        k: int,
    ) -> int:
        """
        Pick the option a slot opens on.

        Parameters
        ----------
        combo : QComboBox
            The dropdown being seeded, already filled with :attr:`options`.
        slot : str
            The input slot this dropdown feeds.
        selected_channels : sequence of str or None
            The mapping to prefer, when it names an available channel.
        k : int
            Position of the slot in the model's input list.

        Returns
        -------
        int
            Index of the option to select. Never negative: :data:`NO_CHANNEL` is
            always among the options, so there is always something to fall back
            on.
        """

        if selected_channels is not None and k < len(selected_channels):
            idx = combo.findText(str(selected_channels[k]))
            if idx >= 0:
                return idx

        idx = combo.findText(slot)
        if idx >= 0:
            return idx

        return combo.findText(NO_CHANNEL)

    def selected_channels(self) -> List[str]:
        """
        The mapping as it currently stands.

        Returns
        -------
        list of str
            One experiment channel name -- or :data:`NO_CHANNEL` -- per input
            slot, in the model's own order. Empty when the model declares no
            inputs.
        """

        return [combo.currentText() for combo in self.channel_cbs]

    def is_empty(self) -> bool:
        """
        Whether every input slot is left unassigned.

        Returns
        -------
        bool
            True when there is at least one slot and none of them is fed by an
            experiment channel, which no model can be run on.
        """

        selected = self.selected_channels()
        return bool(selected) and all(ch == NO_CHANNEL for ch in selected)
