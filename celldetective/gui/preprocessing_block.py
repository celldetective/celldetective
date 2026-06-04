import json
import os
from glob import glob

import numpy as np
from PyQt5.QtCore import QSize, QTimer, Qt
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
    QWidget,
    QHBoxLayout,
)
from fonticon_mdi6 import MDI6
from superqt.fonticon import icon

from celldetective import get_software_location
from celldetective.gui.base.styles import Styles
from celldetective.gui.base.utils import center_window
from celldetective.gui.gui_utils import help_generic
from celldetective.gui.layouts import (
    BackgroundFitCorrectionLayout,
    BackgroundModelFreeCorrectionLayout,
    BackgroundRollingBallCorrectionLayout,
    ChannelOffsetOptionsLayout,
    ProtocolDesignerLayout,
    FourierRegistrationOptionsLayout,
)
from celldetective.utils.experiment import extract_experiment_channels
from celldetective import get_logger

logger = get_logger(__name__)


class PreprocessingDesignerDialog(QDialog, Styles):
    """Floating dialog for configuring preprocessing protocols cleanly without distorting the sidebar"""

    def __init__(self, parent_panel: "PreprocessingPanel") -> None:
        super().__init__(parent_panel)
        self.parent_panel = parent_panel
        self.setWindowTitle("Configure Preprocessing Pipeline")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self.resize(760, 500)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(10)

        # Add the persistent designer container widget
        self.layout.addWidget(self.parent_panel.designer_container_widget)
        self.parent_panel.designer_container_widget.setVisible(True)

        # Spacer/separator
        from celldetective.gui.base.components import QHSeperationLine
        self.layout.addWidget(QHSeperationLine())

        # Buttons row
        self.btn_layout = QHBoxLayout()
        self.btn_layout.setContentsMargins(0, 5, 0, 5)

        self.close_btn = QPushButton("Save & Close")
        self.close_btn.setIcon(icon(MDI6.check, color="black"))
        self.close_btn.setStyleSheet(self.parent_panel.button_select_all)
        self.close_btn.setIconSize(QSize(18, 18))
        self.close_btn.clicked.connect(self.accept)

        self.submit_btn = QPushButton("Submit & Run Pipeline")
        self.submit_btn.setIcon(icon(MDI6.play_circle, color="white"))
        self.submit_btn.setStyleSheet(self.parent_panel.button_style_sheet)
        self.submit_btn.setIconSize(QSize(18, 18))
        self.submit_btn.clicked.connect(self.run_pipeline)

        self.btn_layout.addWidget(self.close_btn, 40)
        self.btn_layout.addWidget(self.submit_btn, 60)
        self.layout.addLayout(self.btn_layout)

        # Style the dialog
        self.setStyleSheet(
            """
            QDialog {
                background-color: #f8fafc;
            }
            """
        )

    def run_pipeline(self):
        self.layout.removeWidget(self.parent_panel.designer_container_widget)
        self.parent_panel.designer_container_widget.setParent(None)
        super().accept()
        self.parent_panel.launch_preprocessing()

    def accept(self) -> None:
        self.layout.removeWidget(self.parent_panel.designer_container_widget)
        self.parent_panel.designer_container_widget.setParent(None)
        super().accept()

    def reject(self) -> None:
        self.layout.removeWidget(self.parent_panel.designer_container_widget)
        self.parent_panel.designer_container_widget.setParent(None)
        super().reject()

    def closeEvent(self, event):
        self.layout.removeWidget(self.parent_panel.designer_container_widget)
        self.parent_panel.designer_container_widget.setParent(None)
        self.parent_panel.update_pipeline_summary()
        super().closeEvent(event)


class PreprocessingPanel(QFrame, Styles):

    def __init__(self, parent_window: QMainWindow) -> None:
        """
        Initialize the PreprocessingPanel.

        Parameters
        ----------
        parent_window : QMainWindow
            The parent window.
        """

        super().__init__()
        self.parent_window = parent_window
        self.exp_channels = self.parent_window.exp_channels
        self.exp_dir = self.parent_window.exp_dir
        self.wells = np.array(self.parent_window.wells, dtype=str)
        exp_config = self.exp_dir + "config.ini"
        self.channel_names, self.channels = extract_experiment_channels(self.exp_dir)
        self.channel_names = np.array(self.channel_names)
        self.background_correction = []
        self.onlyFloat = QDoubleValidator()
        self.onlyInt = QIntValidator()

        # Pre-initialize layouts once to preserve designer states across modal dialog launches
        self.initialize_designer()

        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.grid = QGridLayout(self)

        self.generate_header()

    def generate_header(self):
        """
        Read the mode and prepare a collapsable block to process a specific cell population.

        """

        panel_title = QLabel(f"PREPROCESSING")
        panel_title.setStyleSheet(
            """
			font-weight: bold;
			padding: 0px;
			"""
        )

        self.grid.addWidget(panel_title, 0, 0, 1, 4, alignment=Qt.AlignCenter)

        self.collapse_btn = QPushButton()
        self.collapse_btn.setIcon(icon(MDI6.chevron_down, color="black"))
        self.collapse_btn.setIconSize(QSize(25, 25))
        self.collapse_btn.setStyleSheet(self.button_select_all)
        self.grid.addWidget(self.collapse_btn, 0, 0, 1, 4, alignment=Qt.AlignRight)

        self.populate_contents()

        self.grid.addWidget(self.ContentsFrame, 1, 0, 1, 4, alignment=Qt.AlignTop)
        self.collapse_btn.clicked.connect(
            lambda: self.ContentsFrame.setHidden(not self.ContentsFrame.isHidden())
        )
        self.collapse_btn.clicked.connect(self.collapse_advanced)
        self.ContentsFrame.hide()

    def collapse_advanced(self):
        """
        Collapse or expand the advanced settings panel.
        """

        panels_open = [
            not p.ContentsFrame.isHidden()
            for p in self.parent_window.ProcessPopulations
        ]
        interactions_open = not self.parent_window.NeighPanel.ContentsFrame.isHidden()
        preprocessing_open = (
            not self.parent_window.PreprocessingPanel.ContentsFrame.isHidden()
        )
        is_open = np.array(panels_open + [interactions_open, preprocessing_open])

        if self.ContentsFrame.isHidden():
            self.collapse_btn.setIcon(icon(MDI6.chevron_down, color="black"))
            self.collapse_btn.setIconSize(QSize(20, 20))
            if len(is_open[is_open]) == 0:
                self.parent_window.scroll.setMinimumHeight(int(550))
                self.parent_window.adjustSize()
        else:
            self.collapse_btn.setIcon(icon(MDI6.chevron_up, color="black"))
            self.collapse_btn.setIconSize(QSize(20, 20))
            # Sidebar locked height to prevent control panel distortion
            self.parent_window.scroll.setMinimumHeight(int(550))
            self.parent_window.adjustSize()

            def safe_center():
                """
                Safely center the window.
                """
                try:
                    center_window(self.window())
                except RuntimeError as e:
                    logger.debug(f"Window centering failed: {e}")

            try:
                QTimer.singleShot(10, safe_center)
            except Exception as e:
                logger.debug(f"Window centering trigger failed: {e}")

    def initialize_designer(self):
        """
        Pre-initialize the entire designer layouts and widgets so they keep state across dialog opens.
        """
        from celldetective.gui.base.components import CelldetectiveWidget
        from PyQt5.QtWidgets import QTabWidget, QVBoxLayout

        self.model_free_correction_layout = BackgroundModelFreeCorrectionLayout(self)
        self.fit_correction_layout = BackgroundFitCorrectionLayout(self)
        self.rolling_ball_correction_layout = BackgroundRollingBallCorrectionLayout(self)
        self.channel_offset_options_layout = ChannelOffsetOptionsLayout(self)
        self.fourier_registration_options_layout = FourierRegistrationOptionsLayout(self)

        # Create nested Background Correction tab widget
        self.background_tabs = QTabWidget()
        self.background_tabs.setStyleSheet(self.qtab_style)

        self.fit_wg = CelldetectiveWidget()
        self.fit_wg.setLayout(self.fit_correction_layout)
        self.background_tabs.addTab(self.fit_wg, "Fit")

        self.mf_wg = CelldetectiveWidget()
        self.mf_wg.setLayout(self.model_free_correction_layout)
        self.background_tabs.addTab(self.mf_wg, "Model-free")

        self.rb_wg = CelldetectiveWidget()
        self.rb_wg.setLayout(self.rolling_ball_correction_layout)
        self.background_tabs.addTab(self.rb_wg, "Rolling ball")

        self.background_layout = QVBoxLayout()
        self.background_layout.setContentsMargins(0, 0, 0, 0)
        self.background_layout.addWidget(self.background_tabs)

        self.protocol_layout = ProtocolDesignerLayout(
            parent_window=self,
            tab_layouts=[
                self.background_layout,
                self.channel_offset_options_layout,
                self.fourier_registration_options_layout
            ],
            tab_names=[
                "Background Correction",
                "Channel Offset",
                "Time Registration"
            ],
            title="PREPROCESSING OPERATIONS",
            list_title="Corrections to apply:",
        )

        # Manually link parent window for background sub-layouts
        self.fit_correction_layout.parent_window = self.protocol_layout
        self.model_free_correction_layout.parent_window = self.protocol_layout
        self.rolling_ball_correction_layout.parent_window = self.protocol_layout

        self.help_background_btn = QPushButton()
        self.help_background_btn.setIcon(icon(MDI6.help_circle, color=self.help_color))
        self.help_background_btn.setIconSize(QSize(20, 20))
        self.help_background_btn.clicked.connect(self.help_background)
        self.help_background_btn.setStyleSheet(self.button_select_all)
        self.help_background_btn.setToolTip("Help.")

        self.protocol_layout.title_layout.addWidget(
            self.help_background_btn, 5, alignment=Qt.AlignRight
        )

        # Create persistent widget to house the layout in memory securely
        self.designer_container_widget = QWidget()
        self.designer_container_widget.setLayout(self.protocol_layout)

    def populate_contents(self):
        """
        Populate the content area with a compact, non-stretching sidebar control interface.
        """
        self.ContentsFrame = QFrame()
        self.grid_contents = QGridLayout(self.ContentsFrame)
        self.grid_contents.setContentsMargins(10, 10, 10, 10)
        self.grid_contents.setSpacing(8)

        # Status/Summary Label
        summary_lbl = QLabel("Active Pipeline:")
        summary_lbl.setStyleSheet("font-weight: bold; color: #64748b; font-size: 11px;")

        self.active_pipeline_summary = QLabel("<i>No operations configured.</i>")
        self.active_pipeline_summary.setTextFormat(Qt.RichText)
        self.active_pipeline_summary.setWordWrap(True)
        self.active_pipeline_summary.setStyleSheet("color: #1e293b; font-size: 11px; padding: 6px; background-color: #f1f5f9; border-radius: 4px;")

        self.grid_contents.addWidget(summary_lbl, 0, 0, 1, 4)
        self.grid_contents.addWidget(self.active_pipeline_summary, 1, 0, 1, 4)

        # Action buttons
        self.configure_pipeline_btn = QPushButton("Configure Pipeline...")
        self.configure_pipeline_btn.setIcon(icon(MDI6.cog, color="#1565c0"))
        self.configure_pipeline_btn.setStyleSheet(self.button_style_sheet_2)
        self.configure_pipeline_btn.setIconSize(QSize(18, 18))
        self.configure_pipeline_btn.setToolTip("Open the floating Designer dialog to configure background correction, offsets, and drift registration.")
        self.configure_pipeline_btn.clicked.connect(self.open_designer_dialog)

        self.submit_preprocessing_btn = QPushButton("Submit")
        self.submit_preprocessing_btn.setStyleSheet(self.button_style_sheet)
        self.submit_preprocessing_btn.setIcon(icon(MDI6.play_circle, color="white"))
        self.submit_preprocessing_btn.setIconSize(QSize(18, 18))
        self.submit_preprocessing_btn.setEnabled(False)
        self.submit_preprocessing_btn.clicked.connect(self.launch_preprocessing)

        self.grid_contents.addWidget(self.configure_pipeline_btn, 2, 0, 1, 4)
        self.grid_contents.addWidget(self.submit_preprocessing_btn, 3, 0, 1, 4)

        # Initial summary update
        self.update_pipeline_summary()

    def open_designer_dialog(self):
        """Open the floating dialog containing the ProtocolDesignerLayout"""
        dialog = PreprocessingDesignerDialog(self)
        dialog.exec_()
        self.update_pipeline_summary()

    def update_pipeline_summary(self):
        """Update the readable summary in the sidebar of what steps are currently configured"""
        if not hasattr(self, "protocol_layout") or not hasattr(self, "active_pipeline_summary"):
            return

        steps = []
        for proto in self.protocol_layout.protocols:
            ptype = proto.get("correction_type", "")
            if ptype == "model-free":
                steps.append("Background (MF)")
            elif ptype == "fit":
                steps.append("Background (Fit)")
            elif ptype == "rolling-ball" or ptype == "rolling_ball":
                steps.append("Background (RB)")
            elif ptype == "offset":
                steps.append("Offset")
            elif ptype == "registration":
                steps.append("Registration")
            else:
                steps.append(str(ptype).capitalize())

        if not steps:
            self.active_pipeline_summary.setText("<i>No operations configured.</i>")
            self.submit_preprocessing_btn.setEnabled(False)
        else:
            self.active_pipeline_summary.setText(" ➔ ".join(steps))
            self.submit_preprocessing_btn.setEnabled(True)

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
            elif correction_protocol["correction_type"] == "registration":
                logger.info(
                    f"Fourier registration; {movie_prefix=} {export_prefix=} {correction_protocol=}"
                )
                from celldetective.gui.workers import ProgressWindow
                from celldetective.processes.background_correction import (
                    BackgroundCorrectionProcess,
                )

                current_export_prefix = export_prefix
                if current_export_prefix == "Corrected":
                    current_export_prefix = "Aligned"

                process_args = {
                    "exp_dir": self.exp_dir,
                    "well_option": well_option,
                    "position_option": position_option,
                    "movie_prefix": movie_prefix,
                    "export_prefix": current_export_prefix,
                    "export": True,
                    "return_stacks": False,
                }
                process_args.update(correction_protocol)

                self.job = ProgressWindow(
                    BackgroundCorrectionProcess,
                    parent_window=None,
                    title="Fourier Image Registration",
                    position_info=False,
                    process_args=process_args,
                )
                result = self.job.exec_()
                if result == QDialog.Rejected:
                    logger.info("Registration cancelled.")
                    return None
                
                if correction_protocol.get("plot_trajectory", True) and getattr(self.job, "plot_data", None):
                    from celldetective.gui.workers import DriftTrajectoryPlotDialog
                    dialog = DriftTrajectoryPlotDialog(self.job.plot_data, parent_window=self.parent_window)
                    dialog.exec_()
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
        Helper to choose a proper cell population structure.
        """

        dict_path = os.sep.join(
            [
                get_software_location(),
                "celldetective",
                "gui",
                "help",
                "preprocessing.json",
            ]
        )

        with open(dict_path) as f:
            d = json.load(f)

        suggestion = help_generic(d)
        if isinstance(suggestion, str):
            logger.info(f"{suggestion=}")
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setTextFormat(Qt.RichText)
            msgBox.setText(suggestion)
            msgBox.setWindowTitle("Info")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Ok:
                return None
