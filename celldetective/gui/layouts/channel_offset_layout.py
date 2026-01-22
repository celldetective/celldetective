from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QComboBox, QPushButton, QHBoxLayout
from fonticon_mdi6 import MDI6
from superqt.fonticon import icon

from celldetective.gui.base.styles import Styles
from celldetective.gui.gui_utils import ThresholdLineEdit
from celldetective.utils.parsing import _extract_channel_indices_from_config
from celldetective import get_logger

logger = get_logger(__name__)

class ChannelOffsetOptionsLayout(QVBoxLayout, Styles):

    def __init__(self, parent_window=None, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.parent_window = parent_window
        if hasattr(self.parent_window.parent_window, "exp_config"):
            self.attr_parent = self.parent_window.parent_window
        else:
            self.attr_parent = self.parent_window.parent_window.parent_window

        self.channel_names = self.attr_parent.exp_channels

        self.setContentsMargins(15, 15, 15, 15)
        self.generate_widgets()
        self.add_to_layout()

    def generate_widgets(self):

        self.channel_lbl = QLabel("Channel: ")
        self.channels_cb = QComboBox()
        self.channels_cb.addItems(self.channel_names)

        self.shift_lbl = QLabel("Shift: ")
        self.shift_h_lbl = QLabel("(h): ")
        self.shift_v_lbl = QLabel("(v): ")

        self.set_shift_btn = QPushButton()
        self.set_shift_btn.setIcon(icon(MDI6.image_check, color="k"))
        self.set_shift_btn.setStyleSheet(self.button_select_all)
        self.set_shift_btn.setToolTip("Set the channel shift.")
        self.set_shift_btn.clicked.connect(self.open_offset_viewer)

        self.add_correction_btn = QPushButton("Add correction")
        self.add_correction_btn.setStyleSheet(self.button_style_sheet_2)
        self.add_correction_btn.setIcon(icon(MDI6.plus, color="#1565c0"))
        self.add_correction_btn.setToolTip("Add correction.")
        self.add_correction_btn.setIconSize(QSize(25, 25))
        self.add_correction_btn.clicked.connect(self.add_instructions_to_parent_list)

        self.vertical_shift_le = ThresholdLineEdit(
            init_value=0,
            connected_buttons=[self.add_correction_btn],
            placeholder="vertical shift [pixels]",
            value_type="float",
        )
        self.horizontal_shift_le = ThresholdLineEdit(
            init_value=0,
            connected_buttons=[self.add_correction_btn],
            placeholder="vertical shift [pixels]",
            value_type="float",
        )

    def add_to_layout(self):

        channel_ch_hbox = QHBoxLayout()
        channel_ch_hbox.addWidget(self.channel_lbl, 25)
        channel_ch_hbox.addWidget(self.channels_cb, 75)
        self.addLayout(channel_ch_hbox)

        shift_hbox = QHBoxLayout()
        shift_hbox.addWidget(self.shift_lbl, 25)

        shift_subhbox = QHBoxLayout()
        shift_subhbox.addWidget(self.shift_h_lbl, 10)
        shift_subhbox.addWidget(self.horizontal_shift_le, 75 // 2)
        shift_subhbox.addWidget(self.shift_v_lbl, 10)
        shift_subhbox.addWidget(self.vertical_shift_le, 75 // 2)
        shift_subhbox.addWidget(self.set_shift_btn, 5)

        shift_hbox.addLayout(shift_subhbox, 75)
        self.addLayout(shift_hbox)

        btn_hbox = QHBoxLayout()
        btn_hbox.addWidget(self.add_correction_btn, 95)
        self.addLayout(btn_hbox)

    def add_instructions_to_parent_list(self):

        self.generate_instructions()
        self.parent_window.protocol_layout.protocols.append(self.instructions)
        correction_description = ""
        for index, (key, value) in enumerate(self.instructions.items()):
            if index > 0:
                correction_description += ", "
            correction_description += str(key) + " : " + str(value)
        self.parent_window.protocol_layout.protocol_list.addItem(correction_description)

    def generate_instructions(self):

        self.instructions = {
            "correction_type": "offset",
            "target_channel": self.channels_cb.currentText(),
            "correction_horizontal": self.horizontal_shift_le.get_threshold(),
            "correction_vertical": self.vertical_shift_le.get_threshold(),
        }

    def set_target_channel(self):

        channel_indices = _extract_channel_indices_from_config(
            self.attr_parent.exp_config, [self.channels_cb.currentText()]
        )
        self.target_channel = channel_indices[0]

    def open_offset_viewer(self):
        from celldetective.gui.viewers.channel_offset_viewer import ChannelOffsetViewer

        self.attr_parent.locate_image()
        self.set_target_channel()

        if self.attr_parent.current_stack is not None:
            self.viewer = ChannelOffsetViewer(
                parent_window=self,
                stack_path=self.attr_parent.current_stack,
                channel_names=self.attr_parent.exp_channels,
                n_channels=len(self.channel_names),
                channel_cb=True,
                target_channel=self.target_channel,
                window_title="offset viewer",
            )
            self.viewer.show()
