import configparser
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QLabel, QHBoxLayout, QLineEdit, QPushButton
from PyQt5.QtCore import Qt
import configparser
from celldetective.gui import Styles

class ConfigEditor(QWidget, Styles):
    
    def __init__(self, parent_window):

        """
        Load and edit the experiment config.
        """

        super().__init__()

        self.parent_window = parent_window
        self.config_path = self.parent_window.exp_config

        self.setGeometry(500,200,400,700)

        self.setWindowTitle("Configuration")

        # Create the main layout
        self.layout = QVBoxLayout()

        # Create a scroll area to contain the main layout
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        scroll_content.setLayout(self.scroll_layout)
        scroll.setWidget(scroll_content)

        # Add a label for each section of the config file
        self.labels = []
        self.edit_boxes = []
        self.save_button = None
        self.sections = {}

        # Set the main layout
        self.scroll_layout.addStretch()
        self.layout.addWidget(scroll)
        self.setLayout(self.layout)

        self.load_config()

    def load_config(self):
        file_name = self.config_path
        #self.file_edit.setText(file_name)

        config = configparser.ConfigParser(interpolation=None)
        config.read(file_name)

        # Create a layout for each section of the config file
        for section in config.sections():
            section_layout = QVBoxLayout()
            section_label = QLabel('[{}]'.format(section))
            self.labels.append(section_label)

            # Create an editor box for each parameter in the section
            for key, value in config.items(section):
                edit_box_layout = QHBoxLayout()
                label = QLabel(key)
                edit_box = QLineEdit(value)
                edit_box_layout.addWidget(label)
                edit_box_layout.addWidget(edit_box)
                section_layout.addLayout(edit_box_layout)

                self.edit_boxes.append(edit_box)
                self.sections[key] = (section, edit_box)

            # Add the section label and editor boxes to the main layout
            self.scroll_layout.addWidget(section_label)
            self.scroll_layout.addLayout(section_layout)

        # Add a save button
        save_layout = QHBoxLayout()
        save_button = QPushButton('Save')
        save_button.setStyleSheet(self.button_style_sheet)
        save_button.clicked.connect(self.save_config)
        save_button.setShortcut("Return")
        #save_button.setIcon(QIcon_from_svg(self.parent.abs_path+f"/icons/save.svg", color='white'))

        #save_layout.addStretch()
        save_layout.addWidget(save_button, alignment=Qt.AlignTop)

        # Add the save button to the main layout
        self.layout.addLayout(save_layout)
        self.save_button = save_button

    def save_config(self):
        # Save the configuration to the file
        file_name = self.config_path
            
        config = configparser.ConfigParser(interpolation=None)

        # Update the values in the config object

        for key, (section, edit_box) in self.sections.items():
            if not config.has_section(section):
                config.add_section(section)
            config.set(section, key, edit_box.text())

        # Write the config object to the file
        with open(file_name, 'w') as f:
            config.write(f)

        self.parent_window.load_configuration()
        self.close()