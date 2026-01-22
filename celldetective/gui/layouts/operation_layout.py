from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QButtonGroup, QRadioButton, QHBoxLayout


class OperationLayout(QVBoxLayout):
    """docstring for ClassName"""

    def __init__(self, ratio=(0.25, 0.75), *args):

        super().__init__(*args)

        self.ratio = ratio
        self.generate_widgets()
        self.generate_layout()

    def generate_widgets(self):

        self.operation_lbl = QLabel("Operation: ")
        self.operation_group = QButtonGroup()
        self.subtract_btn = QRadioButton("Subtract")
        self.divide_btn = QRadioButton("Divide")
        self.subtract_btn.toggled.connect(self.activate_clipping_options)
        self.divide_btn.toggled.connect(self.activate_clipping_options)

        self.operation_group.addButton(self.subtract_btn)
        self.operation_group.addButton(self.divide_btn)

        self.clip_group = QButtonGroup()
        self.clip_btn = QRadioButton("Clip")
        self.clip_not_btn = QRadioButton("Do not clip")

        self.clip_group.addButton(self.clip_btn)
        self.clip_group.addButton(self.clip_not_btn)

    def generate_layout(self):

        operation_layout = QHBoxLayout()
        operation_layout.addWidget(self.operation_lbl, 100 * int(self.ratio[0]))
        operation_layout.addWidget(
            self.subtract_btn, 100 * int(self.ratio[1]) // 2, alignment=Qt.AlignCenter
        )
        operation_layout.addWidget(
            self.divide_btn, 100 * int(self.ratio[1]) // 2, alignment=Qt.AlignCenter
        )
        self.addLayout(operation_layout)

        clip_layout = QHBoxLayout()
        clip_layout.addWidget(QLabel(""), 100 * int(self.ratio[0]))
        clip_layout.addWidget(
            self.clip_btn, 100 * int(self.ratio[1]) // 4, alignment=Qt.AlignCenter
        )
        clip_layout.addWidget(
            self.clip_not_btn, 100 * int(self.ratio[1]) // 4, alignment=Qt.AlignCenter
        )
        clip_layout.addWidget(QLabel(""), 100 * int(self.ratio[1]) // 2)
        self.addLayout(clip_layout)

        self.subtract_btn.click()
        self.clip_not_btn.click()

    def activate_clipping_options(self):

        if self.subtract_btn.isChecked():
            self.clip_btn.setEnabled(True)
            self.clip_not_btn.setEnabled(True)
        else:
            self.clip_btn.setEnabled(False)
            self.clip_not_btn.setEnabled(False)
