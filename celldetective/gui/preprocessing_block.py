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
from celldetective.gui.gui_utils import generic_message
from celldetective.preprocessing import (
    _configured_movie_prefix,
    already_corrected_movies,
)
from celldetective.gui.layouts import (
    BackgroundFitCorrectionLayout,
    BackgroundModelFreeCorrectionLayout,
    ChannelOffsetOptionsLayout,
    ProtocolDesignerLayout,
    RegistrationOptionsLayout,
)
from celldetective.utils.experiment import extract_experiment_channels
from celldetective import get_logger

logger = get_logger(__name__)

# Window title of each correction type launched from the protocol list.
CORRECTION_TITLES = {
    "model-free": "Model-Free Background Correction",
    "fit": "Fit Background Correction",
    "offset": "Offset Correction",
    "registration": "Stack Registration",
}


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

        self.registration_layout = QVBoxLayout()
        self.registration_lbl = QLabel("STACK REGISTRATION")
        self.registration_lbl.setStyleSheet(
            """
			font-weight: bold;
			padding: 0px;
			"""
        )
        self.registration_layout.addWidget(
            self.registration_lbl, alignment=Qt.AlignCenter
        )
        self.registration_options_layout = RegistrationOptionsLayout(self)
        self.registration_layout.addLayout(self.registration_options_layout)

        self.protocol_layout.correction_layout.addWidget(QLabel(""))
        self.protocol_layout.correction_layout.addLayout(self.registration_layout)

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
            # With several wells, the position list is the one of the first well: "all" must
            # not be sent as its indices, since wells can hold different numbers of positions.
            well_option = "*"
            position_option = "*"
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
            well_option = self.parent_window.well_list.getSelectedIndices()
            position_option = self.parent_window.position_list.getSelectedIndices()
            if not well_option or not position_option:
                generic_message("Please select at least one well and one position.", "warning")
                return None

        logger.info("Proceed with correction...")

        if self.protocol_layout.protocols and not self.confirm_correcting_corrected_movies(
            well_option, position_option
        ):
            logger.info("Preprocessing cancelled: the movies are already corrected.")
            return None

        # The steps after the first read what the first one wrote, not any "Corrected" stack.
        chained_prefix = "Corrected_" + _configured_movie_prefix(self.exp_dir)

        for k, correction_protocol in enumerate(self.protocol_layout.protocols):

            movie_prefix = None
            export_prefix = "Corrected"
            if k > 0:
                # switch source stack to cumulate multi-channel preprocessing
                movie_prefix = chained_prefix
                export_prefix = None

            correction_type = correction_protocol["correction_type"]
            if correction_type not in CORRECTION_TITLES:
                continue
            logger.info(
                f"{CORRECTION_TITLES[correction_type]}; {movie_prefix=} {export_prefix=} {correction_protocol=}"
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
            }
            if correction_type in ("model-free", "fit"):
                process_args["activation_protocol"] = [["gauss", 2], ["std", 4]]
            process_args.update(correction_protocol)

            self.job = ProgressWindow(
                BackgroundCorrectionProcess,
                parent_window=None,
                title=CORRECTION_TITLES[correction_type],
                position_info=False,
                process_args=process_args,
            )
            result = self.job.exec_()
            if result == QDialog.Rejected:
                logger.info(f"{CORRECTION_TITLES[correction_type]} cancelled.")
                return None
        logger.info("Done.")

    def confirm_correcting_corrected_movies(self, well_option, position_option) -> bool:
        """
        Warn if the movie prefix selects stacks that a previous correction wrote.

        Parameters
        ----------
        well_option, position_option : str or list of int
            Selection of wells and positions.

        Returns
        -------
        bool
            True if there is nothing to warn about or the user accepts correcting them again.
        """

        movies = already_corrected_movies(self.exp_dir, well_option, position_option)
        if not movies:
            return True

        names = sorted({(os.path.basename(path), output) for path, output in movies})
        examples = "\n\n".join(
            f"Reads: {name}\nWrites: {output}" for name, output in names[:3]
        )
        if len(names) > 3:
            examples += f"\n\n... and {len(names) - 3} more."

        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setWindowTitle("Already corrected")
        msgBox.setText(
            f"{len(movies)} selected position(s) would read a stack that a previous "
            "correction wrote, and correct it a second time."
        )
        msgBox.setInformativeText(
            f"{examples}\n\nTo correct the raw movies instead, cancel and "
            "change the movie prefix of the experiment."
        )
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        msgBox.button(QMessageBox.Yes).setText("Correct again")
        msgBox.setDefaultButton(QMessageBox.Cancel)
        return msgBox.exec() == QMessageBox.Yes

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

