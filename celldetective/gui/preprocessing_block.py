import json
import os
from glob import glob

import numpy as np
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from fonticon_mdi6 import MDI6
from superqt.fonticon import icon

from celldetective import get_software_location
from celldetective.gui.base.control_panel_block import ControlPanelBlock
from celldetective.gui.base.styles import Styles
from celldetective.gui.base.help_panel import HelpButton, open_help, open_help_menu
from celldetective.gui.layouts import (
    BackgroundFitCorrectionLayout,
    BackgroundModelFreeCorrectionLayout,
    ChannelOffsetOptionsLayout,
    ProtocolDesignerLayout,
)
from celldetective.utils.experiment import extract_experiment_channels
from celldetective import get_logger

logger = get_logger(__name__)


class PreprocessingPanel(ControlPanelBlock, Styles):

    def __init__(self, parent_window: QMainWindow) -> None:
        """
        Initialize the PreprocessingPanel.

        Parameters
        ----------
        parent_window : QMainWindow
            The parent window.
        """

        super().__init__("PREPROCESSING", parent_window)
        self.exp_channels = self.parent_window.exp_channels
        self.exp_dir = self.parent_window.exp_dir
        self.wells = np.array(self.parent_window.wells, dtype=str)
        exp_config = self.exp_dir + "config.ini"
        self.channel_names, self.channels = extract_experiment_channels(self.exp_dir)
        self.channel_names = np.array(self.channel_names)
        self.background_correction = []
        self.onlyFloat = QDoubleValidator()
        self.onlyInt = QIntValidator()

        self.generate_header()

    def generate_header(self):
        """
        Prepare the collapsable block holding the preprocessing options.
        """

        self.populate_contents()
        self.set_content(self.ContentsFrame)

    def populate_contents(self):
        """
        Populate the content area with preprocessing options.
        """

        self.ContentsFrame = QFrame()
        self.grid_contents = QGridLayout(self.ContentsFrame)

        self.model_free_correction_layout = BackgroundModelFreeCorrectionLayout(self)
        self.fit_correction_layout = BackgroundFitCorrectionLayout(self)

        self.protocol_layout = ProtocolDesignerLayout(
            parent_window=self,
            tab_layouts=[self.fit_correction_layout, self.model_free_correction_layout],
            tab_names=["Fit", "Model-free"],
            title="BACKGROUND CORRECTION",
            list_title="Corrections to apply:",
        )

        self.help_background_btn = HelpButton("Help me correct the background")
        self.help_background_btn.clicked.connect(self.help_background)

        self.protocol_layout.title_layout.addWidget(
            self.help_background_btn, 5, alignment=Qt.AlignRight
        )

        self.channel_offset_correction_layout = QVBoxLayout()

        self.channel_shift_lbl = QLabel("CHANNEL OFFSET CORRECTION")
        self.channel_shift_lbl.setStyleSheet(
            """
			font-weight: bold;
			padding: 0px;
			"""
        )
        self.channel_offset_correction_layout.addWidget(
            self.channel_shift_lbl, alignment=Qt.AlignCenter
        )

        self.channel_offset_options_layout = ChannelOffsetOptionsLayout(self)
        self.channel_offset_correction_layout.addLayout(
            self.channel_offset_options_layout
        )

        self.protocol_layout.correction_layout.addWidget(QLabel(""))
        self.protocol_layout.correction_layout.addLayout(
            self.channel_offset_correction_layout
        )

        self.grid_contents.addLayout(self.protocol_layout, 0, 0, 1, 4)

        self.submit_preprocessing_btn = QPushButton("Submit")
        self.submit_preprocessing_btn.setStyleSheet(self.button_style_sheet)
        self.submit_preprocessing_btn.clicked.connect(self.launch_preprocessing)

        self.grid_contents.addWidget(self.submit_preprocessing_btn, 1, 0, 1, 4)

    def add_offset_instructions_to_parent_list(self):
        """
        Add offset instructions to the parent list.
        """
        logger.info("adding instructions")

    def launch_preprocessing(self):
        """
        Launch the preprocessing task based on selected options.
        """

        msgBox1 = QMessageBox()
        msgBox1.setIcon(QMessageBox.Question)
        msgBox1.setText(
            "Do you want to apply the preprocessing\nto all wells and positions?"
        )
        msgBox1.setWindowTitle("Selection")
        msgBox1.setStandardButtons(
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
        )
        returnValue = msgBox1.exec()
        if returnValue == QMessageBox.Cancel:
            return None
        elif returnValue == QMessageBox.Yes:
            self.parent_window.well_list.selectAll()
            self.parent_window.position_list.selectAll()
        elif returnValue == QMessageBox.No:
            msgBox2 = QMessageBox()
            msgBox2.setIcon(QMessageBox.Question)
            msgBox2.setText(
                "Do you want to apply the preprocessing\nto the positions selected at the top only?"
            )
            msgBox2.setWindowTitle("Selection")
            msgBox2.setStandardButtons(
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            returnValue = msgBox2.exec()
            if returnValue == QMessageBox.Cancel:
                return None
            if returnValue == QMessageBox.No:
                return None

        logger.info("Proceed with correction...")

        # if self.parent_window.well_list.currentText()=='*':
        # 	well_option = "*"
        # else:
        well_option = self.parent_window.well_list.getSelectedIndices()
        position_option = self.parent_window.position_list.getSelectedIndices()

        for k, correction_protocol in enumerate(self.protocol_layout.protocols):

            movie_prefix = None
            export_prefix = "Corrected"
            if k > 0:
                # switch source stack to cumulate multi-channel preprocessing
                movie_prefix = "Corrected"
                export_prefix = None

            if correction_protocol["correction_type"] == "model-free":
                logger.info(f"Model-free correction; movie_prefix={movie_prefix} export_prefix={export_prefix}")
                from celldetective.gui.workers import ProgressWindow
                from celldetective.processes.background_correction import (
                    BackgroundCorrectionProcess,
                )

                process_args = {
                    "exp_dir": self.exp_dir,
                    "well_option": well_option,
                    "position_option": position_option,
                    "movie_prefix": movie_prefix,
                    "export_prefix": export_prefix,
                    "export": True,
                    "return_stacks": False,
                    "activation_protocol": [["gauss", 2], ["std", 4]],
                    "correction_type": "model-free",  # Explicitly set type
                }
                process_args.update(correction_protocol)

                self.job = ProgressWindow(
                    BackgroundCorrectionProcess,
                    parent_window=None,
                    title="Model-Free Background Correction",
                    position_info=False,
                    process_args=process_args,
                )
                result = self.job.exec_()
                if result == QDialog.Rejected:
                    logger.info("Background correction cancelled.")
                    return None

            elif correction_protocol["correction_type"] == "fit":
                logger.info(
                    f"Fit correction; movie_prefix={movie_prefix} export_prefix={export_prefix} correction_protocol={correction_protocol}"
                )
                from celldetective.gui.workers import ProgressWindow
                from celldetective.processes.background_correction import (
                    BackgroundCorrectionProcess,
                )

                process_args = {
                    "exp_dir": self.exp_dir,
                    "well_option": well_option,
                    "position_option": position_option,
                    "movie_prefix": movie_prefix,
                    "export_prefix": export_prefix,
                    "export": True,
                    "return_stacks": False,
                    "activation_protocol": [["gauss", 2], ["std", 4]],
                }
                process_args.update(correction_protocol)

                self.job = ProgressWindow(
                    BackgroundCorrectionProcess,
                    parent_window=None,
                    title="Fit Background Correction",
                    position_info=False,
                    process_args=process_args,
                )
                result = self.job.exec_()
                if result == QDialog.Rejected:
                    logger.info("Background correction cancelled.")
                    return None
            elif correction_protocol["correction_type"] == "offset":
                logger.info(
                    f"Offset correction; {movie_prefix=} {export_prefix=} {correction_protocol=}"
                )
                from celldetective.gui.workers import ProgressWindow
                from celldetective.processes.background_correction import (
                    BackgroundCorrectionProcess,
                )

                process_args = {
                    "exp_dir": self.exp_dir,
                    "well_option": well_option,
                    "position_option": position_option,
                    "movie_prefix": movie_prefix,
                    "export_prefix": export_prefix,
                    "export": True,
                    "return_stacks": False,
                    # Offset specific args if any, otherwise they are in correction_protocol
                }
                process_args.update(correction_protocol)

                self.job = ProgressWindow(
                    BackgroundCorrectionProcess,
                    parent_window=None,
                    title="Offset Correction",
                    position_info=False,
                    process_args=process_args,
                )
                result = self.job.exec_()
                if result == QDialog.Rejected:
                    logger.info("Correction cancelled.")
                    return None
        logger.info("Done.")

    def locate_image(self):
        """
        Load the first frame of the first movie found in the experiment folder as a sample.
        """

        logger.info(f"{self.parent_window.pos}")
        movies = glob(
            self.parent_window.pos
            + os.sep.join(["movie", f"{self.parent_window.movie_prefix}*.tif"])
        )

        if len(movies) == 0:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("Please select a position containing a movie...")
            msgBox.setWindowTitle("Warning")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Ok:
                self.current_stack = None
                return None
        else:
            self.current_stack = movies[0]

    def help_background(self):
        """
        Helper to choose a background correction.
        """

        open_help(
            "preprocessing.json",
            "Correcting the background",
            docs_url="https://celldetective.readthedocs.io/en/latest/preprocessing.html",
            parent=self,
        )

