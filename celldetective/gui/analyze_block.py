from PyQt5.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QSpacerItem,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon
import os
from celldetective.gui.base.styles import Styles
from glob import glob

from celldetective.gui.base.components import generic_message
from celldetective.gui.base.utils import center_window


class AnalysisPanel(QFrame, Styles):
    def __init__(self, parent_window, title=None):

        super().__init__()
        self.parent_window = parent_window
        self.title = title
        if self.title is None:
            self.title = ""
        self.exp_channels = self.parent_window.exp_channels
        self.exp_dir = self.parent_window.exp_dir
        self.soft_path = self.parent_window.parent_window.soft_path
        self.pop_exists = False

        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.grid = QVBoxLayout(self)
        self.grid.setSpacing(20)
        self.generate_header()

    def generate_header(self):
        """
        Read the mode and prepare a collapsable block to process a specific cell population.

        """

        panel_title = QLabel("Survival")
        panel_title.setStyleSheet(
            """
			font-weight: bold;
			padding: 0px;
			"""
        )

        self.grid.addWidget(panel_title, alignment=Qt.AlignCenter)

        self.survival_btn = QPushButton("plot survival")
        self.survival_btn.setIcon(
            QIcon(
                QIcon(
                    os.sep.join(
                        [self.soft_path, "celldetective", "icons", "survival2.png"]
                    )
                )
            )
        )
        self.survival_btn.setStyleSheet(self.button_style_sheet_2)
        self.survival_btn.setIconSize(QSize(35, 35))
        self.survival_btn.clicked.connect(self.configure_survival)
        self.grid.addWidget(self.survival_btn)

        signal_lbl = QLabel("Single-cell signals")
        signal_lbl.setStyleSheet(
            """
			font-weight: bold;
			padding: 0px;
			"""
        )

        self.grid.addWidget(signal_lbl, alignment=Qt.AlignCenter)

        self.plot_signal_btn = QPushButton("plot signals")
        self.plot_signal_btn.setIcon(
            QIcon(
                QIcon(
                    os.sep.join(
                        [self.soft_path, "celldetective", "icons", "signals_icon.png"]
                    )
                )
            )
        )
        self.plot_signal_btn.setStyleSheet(self.button_style_sheet_2)
        self.plot_signal_btn.setIconSize(QSize(35, 35))
        self.plot_signal_btn.clicked.connect(self.configure_plot_signals)
        self.grid.addWidget(self.plot_signal_btn)

        vertical_spacer = QSpacerItem(
            20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding
        )
        self.grid.addItem(vertical_spacer)

    def check_for_tables(self):

        for population in self.parent_window.populations:
            tables = glob(
                self.exp_dir
                + os.sep.join(
                    ["W*", "*", "output", "tables", f"trajectories_{population}.csv"]
                )
            )
            if len(tables) > 0:
                self.pop_exists = True

    def configure_survival(self):
        from celldetective.gui.survival_ui import ConfigSurvival

        self.check_for_tables()
        if self.pop_exists:
            self.config_survival = ConfigSurvival(self)
            self.config_survival.show()
            center_window(self.config_survival)
        else:
            generic_message("No population table could be found... Abort...")
            return None

    def configure_plot_signals(self):
        from celldetective.gui.plot_signals_ui import ConfigSignalPlot

        self.check_for_tables()
        if self.pop_exists:
            self.config_signal_plot = ConfigSignalPlot(self)
            self.config_signal_plot.show()
            center_window(self.config_signal_plot)
        else:
            generic_message("No population table could be found... Abort...")
            return None
