import os

from PyQt5.QtWidgets import (
    QGridLayout,
    QMessageBox,
    QLineEdit,
    QVBoxLayout,
    QComboBox,
    QPushButton,
    QLabel,
    QHBoxLayout,
    QCheckBox,
    QFileDialog,
)
from PyQt5.QtCore import Qt, QSize, QAbstractTableModel
from PyQt5.QtGui import QDoubleValidator, QIntValidator

from celldetective.gui.base.list_widget import ListWidget
from celldetective.gui.base.styles import Styles
from celldetective.gui.base.components import CelldetectiveWidget
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6


from celldetective.gui.base.utils import center_window
from celldetective import get_software_location

from celldetective.filters import *
from os import sep
import json


class PreprocessingLayout(QVBoxLayout, Styles):
    """
    A widget that allows user to choose preprocessing filters for an image
    """

    def __init__(self, parent_window=None, apply_btn_option=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.parent_window = parent_window
        self.apply_btn_option = apply_btn_option
        self.generate_components()
        self.add_to_layout()

    def add_to_layout(self):

        self.setContentsMargins(20, 20, 20, 20)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.preprocess_lbl, 85, alignment=Qt.AlignLeft)
        button_layout.addWidget(self.delete_filter_btn, 5)
        button_layout.addWidget(self.add_filter_btn, 5)
        button_layout.addWidget(self.help_prefilter_btn, 5)
        self.addLayout(button_layout, 25)

        self.addWidget(self.list, 70)
        if self.apply_btn_option:
            self.addWidget(self.apply_btn, 5)

    def generate_components(self):

        self.list = ListWidget(FilterChoice, [])

        self.preprocess_lbl = QLabel("Preprocessing")
        self.preprocess_lbl.setStyleSheet("font-weight: bold;")

        self.delete_filter_btn = QPushButton()
        self.delete_filter_btn.setStyleSheet(self.button_select_all)
        self.delete_filter_btn.setIcon(icon(MDI6.trash_can, color="black"))
        self.delete_filter_btn.setToolTip("Remove filter")
        self.delete_filter_btn.setIconSize(QSize(20, 20))
        self.delete_filter_btn.clicked.connect(self.list.removeSel)

        self.add_filter_btn = QPushButton()
        self.add_filter_btn.setStyleSheet(self.button_select_all)
        self.add_filter_btn.setIcon(icon(MDI6.filter_plus, color="black"))
        self.add_filter_btn.setToolTip("Add filter")
        self.add_filter_btn.setIconSize(QSize(20, 20))
        self.add_filter_btn.clicked.connect(self.list.addItem)

        self.help_prefilter_btn = QPushButton()
        self.help_prefilter_btn.setIcon(icon(MDI6.help_circle, color=self.help_color))
        self.help_prefilter_btn.setIconSize(QSize(20, 20))
        self.help_prefilter_btn.clicked.connect(self.help_prefilter)
        self.help_prefilter_btn.setStyleSheet(self.button_select_all)
        self.help_prefilter_btn.setToolTip("Help.")

        if self.apply_btn_option:
            self.apply_btn = QPushButton("Apply")
            self.apply_btn.setIcon(icon(MDI6.filter_cog_outline, color="white"))
            self.apply_btn.setIconSize(QSize(20, 20))
            self.apply_btn.setStyleSheet(self.button_style_sheet)
            self.apply_btn.clicked.connect(self.parent_window.preprocess_image)

    def help_prefilter(self):
        """
        Helper for prefiltering strategy
        """

        dict_path = os.sep.join(
            [
                get_software_location(),
                "celldetective",
                "gui",
                "help",
                "prefilter-for-segmentation.json",
            ]
        )

        with open(dict_path) as f:
            d = json.load(f)

        suggestion = help_generic(d)
        if isinstance(suggestion, str):
            print(f"{suggestion=}")
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setTextFormat(Qt.RichText)
            msgBox.setText(
                f"The suggested technique is to {suggestion}.\nSee a tutorial <a href='https://celldetective.readthedocs.io/en/latest/segment.html'>here</a>."
            )
            msgBox.setWindowTitle("Info")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Ok:
                return None


class PreprocessingLayout2(PreprocessingLayout):

    def __init__(self, fraction=75, *args, **kwargs):

        self.fraction = fraction
        super().__init__(apply_btn_option=False, *args, **kwargs)
        self.preprocess_lbl.setText("Preprocessing: ")
        self.preprocess_lbl.setStyleSheet("")
        self.setContentsMargins(0, 0, 0, 0)

    def add_to_layout(self):

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)
        main_layout.addWidget(self.preprocess_lbl, self.fraction, alignment=Qt.AlignTop)

        list_grid = QGridLayout()
        list_grid.addWidget(self.list, 0, 0, 2, 2)
        list_grid.addWidget(self.add_filter_btn, 0, 2, 1, 1)
        list_grid.addWidget(self.delete_filter_btn, 1, 2, 1, 1)
        main_layout.addLayout(list_grid, 100 - self.fraction)
        self.add_filter_btn.setFixedWidth(35)  # Ensure the button width is fixed
        self.delete_filter_btn.setFixedWidth(35)
        list_grid.setColumnStretch(2, 0)

        self.addLayout(main_layout)


class PandasModel(QAbstractTableModel):
    """
    from https://stackoverflow.com/questions/31475965/fastest-way-to-populate-qtableview-from-pandas-data-frame
    """

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data
        self.colors = dict()

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
            if role == Qt.BackgroundRole:
                color = self.colors.get((index.row(), index.column()))
                if color is not None:
                    return color
        return None

    def headerData(self, rowcol, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[rowcol]
        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            return self._data.index[rowcol]
        return None

    def change_color(self, row, column, color):
        ix = self.index(row, column)
        self.colors[(row, column)] = color
        self.dataChanged.emit(ix, ix, (Qt.BackgroundRole,))


class GenericOpColWidget(CelldetectiveWidget):

    def __init__(self, parent_window, column=None, title=""):

        super().__init__()

        self.parent_window = parent_window
        self.column = column
        self.title = title

        self.setWindowTitle(self.title)
        # Create the QComboBox and add some items

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(30, 30, 30, 30)

        self.sublayout = QVBoxLayout()

        self.measurements_cb = QComboBox()
        self.measurements_cb.addItems(list(self.parent_window.data.columns))
        if self.column is not None:
            idx = self.measurements_cb.findText(self.column)
            self.measurements_cb.setCurrentIndex(idx)

        measurement_layout = QHBoxLayout()
        measurement_layout.addWidget(QLabel("measurements: "), 25)
        measurement_layout.addWidget(self.measurements_cb, 75)
        self.sublayout.addLayout(measurement_layout)

        self.layout.addLayout(self.sublayout)

        self.submit_btn = QPushButton("Compute")
        self.submit_btn.setStyleSheet(self.button_style_sheet)
        self.submit_btn.clicked.connect(self.launch_operation)
        self.layout.addWidget(self.submit_btn, 30)

        self.setAttribute(Qt.WA_DeleteOnClose)
        center_window(self)

    def launch_operation(self):

        self.compute()
        self.parent_window.model = PandasModel(self.parent_window.data)
        self.parent_window.table_view.setModel(self.parent_window.model)
        self.close()

    def compute(self):
        pass


class QuickSliderLayout(QHBoxLayout):
    """
    A layout class that combines a QLabel and a QSlider in a horizontal box layout.

    This layout provides a convenient way to include a slider with an optional label and configurable
    parameters such as range, precision, and step size. It allows for both integer and decimal step values,
    making it versatile for different types of input adjustments.

    Parameters
    ----------
    label : str, optional
            The label to be displayed next to the slider (default is None).
    slider : QSlider
            The slider widget to be added to the layout.
    layout_ratio : tuple of float, optional
            Defines the width ratio between the label and the slider in the layout. The first element is the
            ratio for the label, and the second is for the slider (default is (0.25, 0.75)).
    slider_initial_value : int or float, optional
            The initial value to set for the slider (default is 1).
    slider_range : tuple of int or float, optional
            A tuple specifying the minimum and maximum values for the slider (default is (0, 1)).
    slider_tooltip : str, optional
            Tooltip text to display when hovering over the slider (default is None).
    decimal_option : bool, optional
            If True, the slider allows decimal values with a specified precision (default is True).
    precision : float, optional
            The step size for the slider when `decimal_option` is enabled (default is 1.0E-03).

    Attributes
    ----------
    qlabel : QLabel
            The label widget that displays the provided label text (only if `label` is provided).
    slider : QSlider
            The slider widget that allows the user to select a value.
    """

    def __init__(
        self,
        label=None,
        slider=None,
        layout_ratio=(0.25, 0.75),
        slider_initial_value=1,
        slider_range=(0, 1),
        slider_tooltip=None,
        decimal_option=True,
        precision=3,
        *args,
    ):
        super().__init__(*args)

        if label is not None and isinstance(label, str):
            self.qlabel = QLabel(label)
            self.addWidget(self.qlabel, int(100 * layout_ratio[0]))

        self.slider = slider
        self.slider.setOrientation(Qt.Horizontal)
        if decimal_option:
            self.slider.setSingleStep(1.0 * (10 ** (-precision)))
            self.slider.setTickInterval(1.0 * (10 ** (-precision)))
            self.slider.setDecimals(precision)
        else:
            self.slider.setSingleStep(1)
            self.slider.setTickInterval(1)

        self.slider.setRange(*slider_range)
        self.slider.setValue(slider_initial_value)
        if isinstance(slider_tooltip, str):
            self.slider.setToolTip(slider_tooltip)

        self.addWidget(self.slider, int(100 * layout_ratio[1]))


class ExportPlotBtn(QPushButton, Styles):
    """
    A custom QPushButton widget for exporting a matplotlib figure.

    This class combines a QPushButton with functionality to export a given matplotlib
    figure (`fig`) to an image file. The button includes an icon and a tooltip for easy
    user interaction. When clicked, a file dialog is opened allowing the user to specify
    the location and file format to save the plot.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
            The matplotlib figure object to be exported.
    export_dir : str, optional
            The default directory where the file will be saved. If not provided, the current
            working directory will be used.
    *args : tuple
            Additional positional arguments passed to the parent `QPushButton` constructor.
    **kwargs : dict
            Additional keyword arguments passed to the parent `QPushButton` constructor.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
            The figure that will be saved when the button is clicked.
    export_dir : str or None
            The default directory where the file dialog will initially point when saving the image.

    Methods
    -------
    save_plot():
            Opens a file dialog to choose the file name and location for saving the figure.
            The figure is then saved in the specified format and location.
    """

    def __init__(self, fig, export_dir=None, *args, **kwargs):

        super().__init__()

        self.export_dir = export_dir
        self.fig = fig

        self.setText("")
        self.setIcon(icon(MDI6.content_save, color="black"))
        self.setStyleSheet(self.button_select_all)
        self.setToolTip("Export figure.")
        self.setIconSize(QSize(20, 20))
        self.clicked.connect(self.save_plot)

    def save_plot(self):
        """
        Opens a file dialog for the user to specify the location and name to save the plot.

        If the user selects a file, the figure is saved with tight layout and 300 DPI resolution.
        Supported formats include PNG, JPG, SVG, and XPM.
        """

        if self.export_dir is not None:
            guess_dir = self.export_dir + sep + "plot.png"
        else:
            guess_dir = "plot.png"
        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save Image", guess_dir, "Images (*.png *.xpm *.jpg *.svg)"
        )  # , options=options
        if fileName:
            self.fig.tight_layout()
            self.fig.savefig(fileName, bbox_inches="tight", dpi=300)


class FilterChoice(CelldetectiveWidget):

    def __init__(self, parent_window):

        super().__init__()
        self.parent_window = parent_window
        self.setWindowTitle("Add filter")
        # Create the QComboBox and add some items
        center_window(self)

        self.default_params = {
            "gauss_filter": {"sigma": 2},
            "median_filter": {"size": 4},
            "maximum_filter": {"size": 4},
            "minimum_filter": {"size": 4},
            "percentile_filter": {"percentile": 99, "size": 4},
            "variance_filter": {"size": 4},
            "std_filter": {"size": 4},
            "laplace_filter": None,
            "abs_filter": None,
            "ln_filter": None,
            "invert_filter": {"value": 65535},
            "subtract_filter": {"value": 1},
            "dog_filter": {"blob_size": 30},
            "log_filter": {"blob_size": 30},
            "tophat_filter": {"size": 4, "connectivity": 4},
            "otsu_filter": None,
            "multiotsu_filter": {"classes": 3},
            "local_filter": {"block_size": 73, "method": "mean", "offset": 0},
            "niblack_filter": {"window_size": 15, "k": 0.2},
            # 'sauvola_filter': {'window_size': 15, 'k': 0.2}
        }

        layout = QVBoxLayout(self)
        self.combo_box = QComboBox(self)
        self.combo_box.addItems(list(self.default_params.keys()))
        self.combo_box.currentTextChanged.connect(self.update_arguments)
        layout.addWidget(self.combo_box)

        self.floatValidator = QDoubleValidator()
        self.arguments_le = [QLineEdit() for i in range(3)]
        for i in range(3):
            self.arguments_le[i].setValidator(self.floatValidator)

        self.arguments_labels = [QLabel("") for i in range(3)]
        for i in range(2):
            hbox = QHBoxLayout()
            hbox.addWidget(self.arguments_labels[i], 40)
            hbox.addWidget(self.arguments_le[i], 60)
            layout.addLayout(hbox)

        self.add_btn = QPushButton("Add")
        self.add_btn.setStyleSheet(self.button_style_sheet)
        self.add_btn.clicked.connect(self.add_current_feature)
        layout.addWidget(self.add_btn)

        self.combo_box.setCurrentIndex(0)
        self.update_arguments()
        center_window(self)

    def add_current_feature(self):

        filtername = self.combo_box.currentText()
        self.parent_window.list_widget.addItems([filtername])

        filter_instructions = [filtername.split("_")[0]]
        for a in self.arguments_le:

            arg = a.text().replace(",", ".")
            arg_num = arg

            if (arg != "") and arg_num.replace(".", "").replace(",", "").isnumeric():
                num = float(arg)
                if num.is_integer():
                    num = int(num)
                filter_instructions.append(num)
            elif arg != "":
                filter_instructions.append(arg)

        print(f"You added filter {filter_instructions}.")

        self.parent_window.items.append(filter_instructions)
        self.close()

    def update_arguments(self):

        selected_filter = self.combo_box.currentText()
        arguments = self.default_params[selected_filter]
        if arguments is not None:
            args = list(arguments.keys())
            for i in range(len(args)):
                self.arguments_labels[i].setEnabled(True)
                self.arguments_le[i].setEnabled(True)

                self.arguments_labels[i].setText(args[i])
                self.arguments_le[i].setText(str(arguments[args[i]]))

            if len(args) < 2:
                for i in range(len(args), 2):
                    self.arguments_labels[i].setEnabled(False)
                    self.arguments_labels[i].setText("")
                    self.arguments_le[i].setEnabled(False)
        else:
            for i in range(2):
                self.arguments_labels[i].setEnabled(False)
                self.arguments_le[i].setEnabled(False)
                self.arguments_labels[i].setText("")


class OperationChoice(CelldetectiveWidget):
    """
    Mini window to select an operation from numpy to apply on the ROI.

    """

    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.setWindowTitle("Add feature")
        # Create the QComboBox and add some items
        self.combo_box = QComboBox(self)
        center_window(self)

        self.combo_box.addItems(
            [
                "mean",
                "median",
                "average",
                "std",
                "var",
                "nanmedian",
                "nanmean",
                "nanstd",
                "nanvar",
            ]
        )

        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self.add_current_feature)

        # Create the layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.combo_box)
        layout.addWidget(self.add_btn)

    def add_current_feature(self):
        filtername = self.combo_box.currentText()
        self.parent_window.list_widget.addItems([filtername])
        self.close()


class GeometryChoice(CelldetectiveWidget, Styles):

    def __init__(self, parent_window):

        super().__init__()
        self.parent_window = parent_window
        self.setWindowTitle("Set distances")
        center_window(self)

        # Create the QComboBox and add some items

        self.dist_label = QLabel("Distance [px]: ")
        self.dist_le = QLineEdit("10")

        self.dist_outer_label = QLabel("Max distance [px]")
        self.dist_outer_le = QLineEdit("100")
        self.outer_to_hide = [self.dist_outer_le, self.dist_outer_label]

        self.outer_btn = QCheckBox("outer distance")
        self.outer_btn.clicked.connect(self.activate_outer_value)

        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self.add_current_feature)
        self.add_btn.setStyleSheet(self.button_style_sheet)

        # Create the layout
        layout = QVBoxLayout(self)
        dist_layout = QHBoxLayout()
        dist_layout.addWidget(self.dist_label, 30)
        dist_layout.addWidget(self.dist_le, 70)

        self.dist_outer_layout = QHBoxLayout()
        self.dist_outer_layout.addWidget(self.dist_outer_label, 30)
        self.dist_outer_layout.addWidget(self.dist_outer_le, 70)

        layout.addLayout(dist_layout)
        layout.addLayout(self.dist_outer_layout)
        layout.addWidget(self.outer_btn)
        layout.addWidget(self.add_btn)

        for el in self.outer_to_hide:
            el.hide()

    def activate_outer_value(self):
        if self.outer_btn.isChecked():
            self.dist_label.setText("Min distance [px]: ")
            for el in self.outer_to_hide:
                el.show()
        else:
            self.dist_label.setText("Distance [px]: ")
            for el in self.outer_to_hide:
                el.hide()

    def add_current_feature(self):

        value = self.dist_le.text()
        if self.outer_btn.isChecked():
            value2 = self.dist_outer_le.text()
            values = [value + "-" + value2]
        else:
            values = [value]
        self.parent_window.list_widget.addItems(values)
        self.close()


class DistanceChoice(CelldetectiveWidget):

    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.setWindowTitle("Set distances")
        self.floatValidator = QDoubleValidator()
        center_window(self)

        # Create the QComboBox and add some items

        self.dist_label = QLabel("Distance [px]: ")
        self.dist_le = QLineEdit("10")
        self.dist_le.setValidator(self.floatValidator)

        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self.add_current_feature)

        # Create the layout
        layout = QVBoxLayout(self)
        dist_layout = QHBoxLayout()
        dist_layout.addWidget(self.dist_label, 30)
        dist_layout.addWidget(self.dist_le, 70)

        layout.addLayout(dist_layout)
        layout.addWidget(self.add_btn)

    def add_current_feature(self):
        value = self.dist_le.text().replace(",", ".")
        values = [value]
        self.parent_window.list_widget.addItems(values)
        self.close()


class ThresholdLineEdit(QLineEdit):
    """
    A custom QLineEdit widget to manage and validate threshold values.

    This class extends QLineEdit to input and manage threshold values (either float or int),
    with optional validation and interaction with connected QPushButtons. The widget can
    validate the input and enable/disable buttons based on whether a valid threshold is set.

    Parameters
    ----------
    init_value : float or int, optional
            The initial threshold value to display in the input field (default is 2.0).
    connected_buttons : QPushButton or list of QPushButton, optional
            QPushButton(s) that should be enabled/disabled based on the validity of the threshold
            value (default is None).
    placeholder : str, optional
            Placeholder text to show when no value is entered in the input field
            (default is 'px > thresh are masked').
    value_type : str, optional
            Specifies the type of threshold value, either 'float' or 'int' (default is 'float').

    Methods
    -------
    enable_btn():
            Enables or disables connected QPushButtons based on the validity of the threshold value.
    set_threshold(value):
            Sets the input field to the given threshold value.
    get_threshold(show_warning=True):
            Retrieves the current threshold value from the input field, returning it as a float or int.
            If invalid, optionally displays a warning dialog.

    Example
    -------
    >>> threshold_input = ThresholdLineEdit(init_value=5, value_type='int')
    >>> print(threshold_input.get_threshold())
    5
    """

    def __init__(
        self,
        init_value=2.0,
        connected_buttons=None,
        placeholder="px > thresh are masked",
        value_type="float",
        *args,
    ):
        super().__init__(*args)

        self.init_value = init_value
        self.value_type = value_type
        self.connected_buttons = connected_buttons
        self.setPlaceholderText(placeholder)

        if self.value_type == "float":
            self.setValidator(QDoubleValidator())
        else:
            self.init_value = int(self.init_value)
            self.setValidator(QIntValidator())

        if self.connected_buttons is not None:
            self.textChanged.connect(self.enable_btn)
        self.set_threshold(self.init_value)

    def enable_btn(self):
        """
        Enable or disable connected QPushButtons based on the threshold value.

        If the current threshold value is valid, the connected buttons will be enabled.
        If the value is invalid or empty, the buttons will be disabled.
        """

        thresh = self.get_threshold(show_warning=False)
        if isinstance(self.connected_buttons, QPushButton):
            cbs = [self.connected_buttons]
        else:
            cbs = self.connected_buttons

        if thresh is None:
            for c in cbs:
                c.setEnabled(False)
        else:
            for c in cbs:
                c.setEnabled(True)

    def set_threshold(self, value):
        """
        Set the input field to the specified threshold value.

        Parameters
        ----------
        value : float or int
                The value to set in the input field.
        """

        try:
            self.setText(str(value).replace(".", ","))
        except:
            print("Please provide a valid threshold value...")

    def get_threshold(self, show_warning=True):
        """
        Retrieve the current threshold value from the input field.

        Converts the value to a float or int based on the `value_type` attribute. If the value
        is invalid and `show_warning` is True, a warning dialog is shown.

        Parameters
        ----------
        show_warning : bool, optional
                If True, show a warning dialog if the value is invalid (default is True).

        Returns
        -------
        float or int or None
                The threshold value as a float or int, or None if the value is invalid.
        """

        try:
            if self.value_type == "float":
                thresh = float(self.text().replace(",", "."))
            else:
                thresh = int(self.text().replace(",", "."))
        except ValueError:
            if show_warning:
                msgBox = QMessageBox()
                msgBox.setWindowTitle("warning")
                msgBox.setIcon(QMessageBox.Critical)
                msgBox.setText("Please set a valid threshold value.")
                msgBox.setWindowTitle("")
                msgBox.setStandardButtons(QMessageBox.Ok)
                returnValue = msgBox.exec()
            thresh = None

        return thresh


def color_from_status(status, recently_modified=False):
    if not recently_modified:
        if status == 0:
            return "tab:blue"
        elif status == 1:
            return "tab:red"
        elif status == 2:
            return "yellow"
        else:
            return "k"
    else:
        if status == 0:
            return "tab:cyan"
        elif status == 1:
            return "tab:orange"
        elif status == 2:
            return "tab:olive"
        else:
            return "k"


def color_from_state(state, recently_modified=False):
    """
    Generate a color map based on unique values in the provided state array.

    This function creates a mapping between the unique values found in the `state` array
    and colors from the `tab10` colormap in Matplotlib. A special condition is applied
    to the value `99`, which is assigned the color black ('k').

    Parameters
    ----------
    state : array-like
            An array or list of state values to be used for generating the color map.

    Returns
    -------
    dict
            A dictionary where the keys are the unique state values from the `state` array,
            and the values are the corresponding colors from Matplotlib's `tab10` colormap.
            The value `99` is mapped to the color black ('k').

    Notes
    -----
    - Matplotlib's `tab10` colormap is used for values other than `99`.
    """

    unique_values = np.unique(state)
    color_map = {}
    for value in unique_values:

        try:
            if np.isnan(value):
                value = "nan"
                color_map[value] = "k"
                continue
        except TypeError:
            pass

        if value == 0:
            color_map[value] = "tab:blue"
        elif value == 1:
            color_map[value] = "tab:red"
        elif value == 99:
            color_map[value] = "k"
        else:
            import matplotlib.pyplot as plt

            if isinstance(value, (int, float, np.number)):
                idx = value
            else:
                idx = hash(str(value)) % 20
            color_map[value] = plt.cm.tab20(idx / 20.0)

    return color_map


def color_from_class(cclass, recently_modified=False):
    if not recently_modified:
        if cclass == 0:
            return "tab:red"
        elif cclass == 1:
            return "tab:blue"
        elif cclass == 2:
            return "yellow"
        else:
            return "k"
    else:
        if cclass == 0:
            return "tab:orange"
        elif cclass == 1:
            return "tab:cyan"
        elif cclass == 2:
            return "tab:olive"
        else:
            return "k"


class ChannelChoice(CelldetectiveWidget):

    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        # self.channel_names = channel_names
        self.setWindowTitle("Choose target channel")
        # Create the QComboBox and add some items
        self.combo_box = QComboBox(self)
        center_window(self)

        channels = parent_window.channel_names

        self.combo_box.addItems(channels)

        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self.add_current_channel)

        # Create the layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.combo_box)
        layout.addWidget(self.add_btn)

    def add_current_channel(self):
        filtername = self.combo_box.currentText()
        self.parent_window.list_widget.addItems([filtername])
        self.close()


def help_generic(tree):
    """
    Interactively traverse a decision tree to provide user guidance based on a nested dictionary structure.

    This function takes a nested dictionary representing a decision tree and guides the user through
    it step-by-step by displaying messages for user input using the `generic_msg()` function.
    At each step, the user selects a key that corresponds to a further step in the tree, until a
    final suggestion (leaf node) is reached.

    Parameters
    ----------
    tree : dict
            A dictionary where keys represent options and values represent either further steps (as dictionaries)
            or a final suggestion (leaf nodes).

    Returns
    -------
    any
            The final suggestion or outcome after traversing the decision tree.

    Example
    -------
    >>> decision_tree = {
    ...     'Start': {
    ...         'Option 1': {
    ...             'Sub-option 1': 'Final suggestion 1',
    ...             'Sub-option 2': 'Final suggestion 2'
    ...         },
    ...         'Option 2': 'Final suggestion 3'
    ...     }
    ... }
    >>> result = help_generic(decision_tree)
    # The function prompts the user to choose between "Option 1" or "Option 2",
    # and then proceeds through the tree based on the user's choices.
    """

    output = generic_msg(list(tree.keys())[0])
    while output is not None:
        tree = tree[list(tree.keys())[0]][output]
        if isinstance(tree, dict):
            output = generic_msg(list(tree.keys())[0])
        else:
            # return the final suggestion
            output = None
    return tree


def generic_msg(text):
    """
    Display a message box with a question and capture the user's response.

    This function creates a message box with a `Yes`, `No`, and `Cancel` option,
    displaying the provided `text` as the question. It returns the user's selection as a string.

    Parameters
    ----------
    text : str
            The message or question to display in the message box.

    Returns
    -------
    str or None
            The user's response: "yes" if Yes is selected, "no" if No is selected,
            and `None` if Cancel is selected or the dialog is closed.

    Example
    -------
    >>> response = generic_msg("Would you like to continue?")
    >>> if response == "yes":
    ...     print("User chose Yes")
    ... elif response == "no":
    ...     print("User chose No")
    ... else:
    ...     print("User cancelled the action")

    Notes
    -----
    - The message box displays a window with three options: Yes, No, and Cancel.
    """

    msgBox = QMessageBox()
    msgBox.setIcon(QMessageBox.Question)
    msgBox.setText(text)
    msgBox.setWindowTitle("Question")
    msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
    returnValue = msgBox.exec()
    if returnValue == QMessageBox.Yes:
        return "yes"
    elif returnValue == QMessageBox.No:
        return "no"
    else:
        return None
