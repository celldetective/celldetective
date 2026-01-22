from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QTabWidget, QSizePolicy, QListWidget, QPushButton, QHBoxLayout
from fonticon_mdi6 import MDI6
from superqt.fonticon import icon

from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.gui.base.styles import Styles


class ProtocolDesignerLayout(QVBoxLayout, Styles):
    """Multi tabs and list widget configuration for background correction
    in preprocessing and measurements
    """

    def __init__(
        self,
        parent_window=None,
        tab_layouts=[],
        tab_names=[],
        title="",
        list_title="",
        *args,
    ):

        super().__init__(*args)

        self.title = title
        self.parent_window = parent_window
        self.channel_names = self.parent_window.channel_names
        self.tab_layouts = tab_layouts
        self.tab_names = tab_names
        self.list_title = list_title
        self.protocols = []
        assert len(self.tab_layouts) == len(self.tab_names)

        self.generate_widgets()
        self.generate_layout()

    def generate_widgets(self):

        self.title_lbl = QLabel(self.title)
        self.title_lbl.setStyleSheet(
            """
			font-weight: bold;
			padding: 0px;
			"""
        )

        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        for k in range(len(self.tab_layouts)):
            wg = CelldetectiveWidget()
            self.tab_layouts[k].parent_window = self
            wg.setLayout(self.tab_layouts[k])
            self.tabs.addTab(wg, self.tab_names[k])

        self.protocol_list_lbl = QLabel(self.list_title)
        self.protocol_list = QListWidget()

        self.delete_protocol_btn = QPushButton("")
        self.delete_protocol_btn.setStyleSheet(self.button_select_all)
        self.delete_protocol_btn.setIcon(icon(MDI6.trash_can, color="black"))
        self.delete_protocol_btn.setToolTip("Remove.")
        self.delete_protocol_btn.setIconSize(QSize(20, 20))
        self.delete_protocol_btn.clicked.connect(self.remove_protocol_from_list)

    def generate_layout(self):

        self.correction_layout = QVBoxLayout()

        self.background_correction_layout = QVBoxLayout()
        self.background_correction_layout.setContentsMargins(0, 0, 0, 0)
        self.title_layout = QHBoxLayout()
        self.title_layout.addWidget(self.title_lbl, 100, alignment=Qt.AlignCenter)
        self.background_correction_layout.addLayout(self.title_layout)
        self.background_correction_layout.addWidget(self.tabs)
        self.correction_layout.addLayout(self.background_correction_layout)

        self.addLayout(self.correction_layout)

        self.list_layout = QVBoxLayout()
        list_header_layout = QHBoxLayout()
        list_header_layout.addWidget(self.protocol_list_lbl)
        list_header_layout.addWidget(self.delete_protocol_btn, alignment=Qt.AlignRight)
        self.list_layout.addLayout(list_header_layout)
        self.list_layout.addWidget(self.protocol_list)

        self.addLayout(self.list_layout)

    def remove_protocol_from_list(self):

        current_item = self.protocol_list.currentRow()
        if current_item > -1:
            del self.protocols[current_item]
            self.protocol_list.takeItem(current_item)
