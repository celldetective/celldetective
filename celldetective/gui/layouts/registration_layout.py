from typing import Optional

from PyQt5.QtCore import QSize
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QComboBox,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
)
from fonticon_mdi6 import MDI6
from superqt.fonticon import icon

from celldetective.gui.base.styles import Styles
from celldetective.gui.gui_utils import ThresholdLineEdit
from celldetective.utils.parsing import _extract_channel_indices_from_config
from celldetective import get_logger

logger = get_logger(__name__)


class RegistrationOptionsLayout(QVBoxLayout, Styles):
    """Options of the stack registration by Fourier phase cross-correlation."""

    def __init__(
        self, parent_window: Optional[QMainWindow] = None, *args, **kwargs
    ) -> None:
        """
        Initialize the RegistrationOptionsLayout.

        Parameters
        ----------
        parent_window : QMainWindow, optional
            The preprocessing panel, which holds the protocol list.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """

        super().__init__(*args, **kwargs)

        self.parent_window = parent_window
        self.channel_names = self.parent_window.exp_channels

        self.setContentsMargins(15, 15, 15, 15)
        self.generate_widgets()
        self.add_to_layout()

    def generate_widgets(self):
        """Generate the widgets."""

        self.channel_lbl = QLabel("Channel: ")
        self.channels_cb = QComboBox()
        self.channels_cb.addItems(self.channel_names)
        self.channels_cb.setToolTip(
            "Channel on which the drift is estimated.\nThe shift is applied to all channels."
        )

        self.add_correction_btn = QPushButton("Add correction")
        self.add_correction_btn.setStyleSheet(self.button_style_sheet_2)
        self.add_correction_btn.setIcon(icon(MDI6.plus, color="#1565c0"))
        self.add_correction_btn.setToolTip("Add correction.")
        self.add_correction_btn.setIconSize(QSize(25, 25))
        self.add_correction_btn.clicked.connect(self.add_instructions_to_parent_list)

        self.radius_lbl = QLabel("Radius: ")
        self.radius_le = QLineEdit()
        validator = QDoubleValidator()
        validator.setBottom(0.0)
        self.radius_le.setValidator(validator)
        self.radius_le.setPlaceholderText("full frame")
        self.radius_le.setToolTip(
            "Radius [px] of the disk centred on the image inside which the correlation is computed.\n"
            "Structures outside (edge artefacts, vignetting, dust) are ignored.\n"
            "Leave empty to use the full frame."
        )

        self.roi_viewer_btn = QPushButton()
        self.roi_viewer_btn.setIcon(icon(MDI6.image_check, color="k"))
        self.roi_viewer_btn.setStyleSheet(self.button_select_all)
        self.roi_viewer_btn.setToolTip(
            "Tune the correlation radius and Tukey α on a frame of the current position."
        )
        self.roi_viewer_btn.clicked.connect(self.open_roi_viewer)

        self.alpha_lbl = QLabel("Tukey α: ")
        self.alpha_le = ThresholdLineEdit(
            init_value=0.25,
            connected_buttons=[self.add_correction_btn],
            placeholder="taper fraction in [0, 1]",
            value_type="float",
            bottom=0.0,
        )
        self.alpha_le.setToolTip(
            "Fraction of the correlation region smoothly tapered to zero.\n"
            "0 = no taper, 1 = Hann window."
        )

        self.upsample_lbl = QLabel("Upsampling: ")
        self.upsample_le = ThresholdLineEdit(
            init_value=10,
            connected_buttons=[self.add_correction_btn],
            placeholder="sub-pixel precision factor",
            value_type="int",
            bottom=1,
        )
        self.upsample_le.setToolTip(
            "Shifts are estimated with a 1/upsampling pixel precision."
        )

        self.downscale_lbl = QLabel("Downscale: ")
        self.downscale_le = ThresholdLineEdit(
            init_value=1,
            connected_buttons=[self.add_correction_btn],
            placeholder="block-averaging factor",
            value_type="int",
            bottom=1,
        )
        self.downscale_le.setToolTip(
            "Estimate the drift on the registration channel reduced by this factor\n"
            "(block averaging), then apply the rescaled shift at full resolution.\n"
            "Faster on large images. 1 = no downscaling."
        )

        self.reference_lbl = QLabel("Reference: ")
        self.reference_cb = QComboBox()
        self.reference_cb.addItems(["previous", "first"])
        self.reference_cb.setToolTip(
            "previous: correlate each frame with the previous one and accumulate the shifts.\n"
            "first: correlate each frame with the first frame."
        )

    def add_to_layout(self):
        """Add widgets to the layout."""

        radius_hbox = QHBoxLayout()
        radius_hbox.addWidget(self.radius_le, 95)
        radius_hbox.addWidget(self.roi_viewer_btn, 5)

        for lbl, widget in [
            (self.channel_lbl, self.channels_cb),
            (self.radius_lbl, radius_hbox),
            (self.alpha_lbl, self.alpha_le),
            (self.upsample_lbl, self.upsample_le),
            (self.downscale_lbl, self.downscale_le),
            (self.reference_lbl, self.reference_cb),
        ]:
            hbox = QHBoxLayout()
            hbox.addWidget(lbl, 25)
            if isinstance(widget, QHBoxLayout):
                hbox.addLayout(widget, 75)
            else:
                hbox.addWidget(widget, 75)
            self.addLayout(hbox)

        btn_hbox = QHBoxLayout()
        btn_hbox.addWidget(self.add_correction_btn, 95)
        self.addLayout(btn_hbox)

    def add_instructions_to_parent_list(self):
        """Add instructions to the parent protocol list."""

        if not self.generate_instructions():
            return
        self.parent_window.protocol_layout.protocols.append(self.instructions)
        correction_description = ", ".join(
            f"{key} : {value}" for key, value in self.instructions.items()
        )
        self.parent_window.protocol_layout.protocol_list.addItem(correction_description)

    def generate_instructions(self) -> bool:
        """
        Generate the instructions dictionary.

        Returns
        -------
        bool
            False if a parameter is invalid, in which case a warning is shown.
        """

        alpha = self.alpha_le.get_threshold()
        upsample = self.upsample_le.get_threshold()
        downscale = self.downscale_le.get_threshold()
        if alpha is None or upsample is None or downscale is None:
            return False
        try:
            radius = self._parse_radius()
        except ValueError:
            self._warn("The radius must be a number, or empty for the full frame.")
            return False
        if upsample < 1:
            self._warn("The upsampling factor must be at least 1.")
            return False
        if downscale < 1:
            self._warn("The downscaling factor must be at least 1.")
            return False
        if alpha > 1.0:
            self._warn("The Tukey α must be between 0 and 1.")
            return False
        if radius is not None and radius <= 0:
            self._warn("The radius must be strictly positive, or empty for the full frame.")
            return False

        self.instructions = {
            "correction_type": "registration",
            "target_channel": self.channels_cb.currentText(),
            "radius": radius,
            "tukey_alpha": alpha,
            "upsample_factor": int(upsample),
            "downscale": int(downscale),
            "reference": self.reference_cb.currentText(),
        }
        return True

    def open_roi_viewer(self):
        """Open a frame of the current position to tune the correlation disk."""
        from celldetective.gui.viewers.registration_roi_viewer import (
            RegistrationROIViewer,
        )

        self.parent_window.locate_image()
        stack_path = getattr(self.parent_window, "current_stack", None)
        if stack_path is None:
            return

        target_channel = self.channels_cb.currentIndex()
        exp_config = getattr(self.parent_window.parent_window, "exp_config", None)
        if exp_config is not None:
            target_channel = _extract_channel_indices_from_config(
                exp_config, [self.channels_cb.currentText()]
            )[0]

        try:
            initial_radius = self._parse_radius()
        except ValueError:
            initial_radius = None
        alpha = self.alpha_le.get_threshold(show_warning=False)

        self.viewer = RegistrationROIViewer(
            self,
            stack_path=stack_path,
            channel_names=self.channel_names,
            n_channels=len(self.channel_names),
            channel_cb=True,
            target_channel=target_channel,
            window_title="Registration ROI",
            initial_radius=initial_radius,
            tukey_alpha=0.25 if alpha is None else alpha,
        )
        self.viewer.show()

    def _parse_radius(self):
        """
        Read the radius field.

        Returns
        -------
        float or None
            The radius, or None when the field is empty (full frame).

        Raises
        ------
        ValueError
            If the field holds text that is not a number, e.g. an intermediate input such as
            ``"1e"`` that the validator lets through.
        """
        radius_text = self.radius_le.text().strip().replace(",", ".")
        return float(radius_text) if radius_text else None

    @staticmethod
    def _warn(text: str):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setText(text)
        msgBox.setWindowTitle("Warning")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec()
