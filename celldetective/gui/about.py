from PyQt5.QtWidgets import QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.gui.base.utils import center_window
from celldetective._version import __version__


class AboutWidget(CelldetectiveWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("About celldetective")
        self.setMaximumWidth(320)
        center_window(self)

        logo = QPixmap(self.celldetective_logo_path)

        # Create the layout
        layout = QVBoxLayout(self)
        img_label = QLabel("")
        img_label.setPixmap(logo)
        layout.addWidget(img_label, alignment=Qt.AlignCenter)

        self.soft_name = QLabel("celldetective")
        self.soft_name.setStyleSheet(
            """font-weight: bold;
										font-size: 18px;
									"""
        )
        layout.addWidget(self.soft_name, alignment=Qt.AlignCenter)

        self.version_lbl = QLabel(
            f'Version {__version__} <a href="https://github.com/celldetective/celldetective'
            f'/releases">(release notes)</a>'
        )
        self.version_lbl.setOpenExternalLinks(True)
        layout.addWidget(self.version_lbl, alignment=Qt.AlignCenter)

        self.lab_lbl = QLabel(
            "Developed at Laboratoire Adhésion et Inflammation (LAI) INSERM U1067 CNRS UMR 7333"
        )
        self.lab_lbl.setWordWrap(True)
        layout.addWidget(self.lab_lbl, alignment=Qt.AlignCenter)

        self.centuri_mention = QLabel(
            "The project leading to this publication has received funding from France 2030, the French Government "
            "program managed by the French National Research Agency (ANR-16-CONV-0001) and from Excellence Initiative "
            "of Aix-Marseille University - AMIDEX')"
        )
        self.centuri_mention.setWordWrap(True)
        layout.addWidget(self.centuri_mention, alignment=Qt.AlignCenter)
