from PyQt5.QtWidgets import (
    QRadioButton,
    QButtonGroup,
    QTableView,
    QAction,
    QMenu,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QComboBox,
    QLabel,
    QCheckBox,
    QMessageBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor

from celldetective.gui.gui_utils import (
    PandasModel,
)
from celldetective.gui.base.figure_canvas import FigureCanvas
from celldetective.gui.base.utils import center_window
from celldetective.gui.table_ops._maths import DifferentiateColWidget, OperationOnColsWidget, CalibrateColWidget, \
    AbsColWidget, LogColWidget
from celldetective.gui.table_ops._merge_one_hot import MergeOneHotWidget
from celldetective.gui.table_ops._query_table import QueryWidget
from celldetective.gui.table_ops._rename_col import RenameColWidget
from celldetective.relative_measurements import expand_pair_table
from celldetective.utils.data_cleaning import collapse_trajectories_by_status
from celldetective.utils.stats import test_2samp_generic
import numpy as np
import os
from celldetective.gui.base.components import (
    CelldetectiveWidget,
    CelldetectiveMainWindow,
    QHSeperationLine,
)
from superqt import QColormapComboBox, QSearchableComboBox
from math import floor
from celldetective import get_logger
from celldetective.utils.types import test_bool_array

logger = get_logger(__name__)


class PivotTableUI(CelldetectiveWidget):

    def __init__(self, data, title="", mode=None, *args, **kwargs):

        CelldetectiveWidget.__init__(self, *args, **kwargs)

        self.data = data
        self.title = title
        self.mode = mode

        self.setWindowTitle(title)
        print("tab to show: ", self.data)

        self.table = QTableView(self)

        self.v_layout = QVBoxLayout()
        self.information_label = QLabel("Information about color code...")
        self.v_layout.addWidget(self.information_label)
        self.v_layout.addWidget(self.table)
        self.setLayout(self.v_layout)

        self.showdata()

        if self.mode == "cliff":
            self.color_cells_cliff()
        elif self.mode == "pvalue":
            self.color_cells_pvalue()

        self.table.resizeColumnsToContents()
        self.setAttribute(Qt.WA_DeleteOnClose)
        center_window(self)

    def showdata(self):
        self.model = PandasModel(self.data)
        self.table.setModel(self.model)

    def set_cell_color(self, row, column, color="red"):
        self.model.change_color(
            row, column, QBrush(QColor(color))
        )  # eval(f"Qt.{color}")

    def color_cells_cliff(self):

        color_codes = {
            "negligible": "#eff3ff",  # Green
            "small": "#bdd7e7",  # Yellow
            "medium": "#6baed6",  # Orange
            "large": "#2171b5",  # Red
        }

        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                value = self.data.iloc[i, j]
                if value < 0.147:
                    self.set_cell_color(i, j, color_codes["negligible"])
                elif value < 0.33:
                    self.set_cell_color(i, j, color_codes["small"])
                elif value < 0.474:
                    self.set_cell_color(i, j, color_codes["medium"])
                elif value >= 0.474:
                    self.set_cell_color(i, j, color_codes["large"])

        # Create the HTML text for the label
        html_caption = f"""
		<p style="background-color:black; padding: 5px; font-weight:bold;">
			<span style="color:{color_codes['negligible']}">Negligible</span>, 
			<span style="color:{color_codes['small']}">Small</span>, 
			<span style="color:{color_codes['medium']}">Medium</span>, 
			<span style="color:{color_codes['large']}">Large</span>
		</p>
		"""
        self.information_label.setText(html_caption)

    def color_cells_pvalue(self):

        color_codes = {
            "ns": "#fee5d9",
            "*": "#fcae91",
            "**": "#fb6a4a",
            "***": "#de2d26",
            "****": "#a50f15",
        }

        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                value = self.data.iloc[i, j]
                if value <= 0.0001:
                    self.set_cell_color(i, j, color_codes["****"])
                elif value <= 0.001:
                    self.set_cell_color(i, j, color_codes["***"])
                elif value <= 0.01:
                    self.set_cell_color(i, j, color_codes["**"])
                elif value <= 0.05:
                    self.set_cell_color(i, j, color_codes["*"])
                elif value > 0.05:
                    self.set_cell_color(i, j, color_codes["ns"])

        html_caption = f"""
		<p style="background-color:black; padding: 5px; font-weight:bold;">
			<span style="color:{color_codes['ns']}">ns</span>, 
			<span style="color:{color_codes['*']}">*</span>, 
			<span style="color:{color_codes['**']}">**</span>, 
			<span style="color:{color_codes['***']}">***</span>,
			<span style="color:{color_codes['****']}">****</span>
		</p>
		"""
        self.information_label.setText(html_caption)


class TableUI(CelldetectiveMainWindow):

    def __init__(
        self,
        data,
        title,
        population="targets",
        plot_mode="plot_track_signals",
        save_inplace_option=False,
        collapse_tracks_option=True,
        *args,
        **kwargs,
    ):

        CelldetectiveMainWindow.__init__(self, *args, **kwargs)

        self.setWindowTitle(title)
        self.setGeometry(100, 100, 1000, 400)
        center_window(self)
        self.title = title
        self.plot_mode = plot_mode
        self.population = population
        self.numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
        self.groupby_cols = ["position", "TRACK_ID"]
        self.tracks = False
        self.save_inplace_option = save_inplace_option
        self.collapse_tracks_option = collapse_tracks_option

        if self.population == "pairs":
            self.groupby_cols = [
                "position",
                "reference_population",
                "neighbor_population",
                "REFERENCE_ID",
                "NEIGHBOR_ID",
            ]
            self.tracks = True  # for now
        else:
            if "TRACK_ID" in data.columns:
                if not np.all(data["TRACK_ID"].isnull()):
                    self.tracks = True

        self.data = data

        self._createMenuBar()
        self._create_actions()

        self.table_view = QTableView(self)
        self.setCentralWidget(self.table_view)

        # Set the model for the table view

        import matplotlib.pyplot as plt

        plt.rcParams["svg.fonttype"] = "none"

        self.model = PandasModel(data)
        self.table_view.setModel(self.model)
        self.table_view.resizeColumnsToContents()

    def resizeEvent(self, event):

        super().resizeEvent(event)

        try:
            self.fig.tight_layout()
        except:
            pass

    def _create_actions(self):

        self.save_as = QAction("&Save as...", self)
        self.save_as.triggered.connect(self.save_as_csv)
        self.save_as.setShortcut("Ctrl+s")
        self.fileMenu.addAction(self.save_as)

        if self.save_inplace_option:
            self.save_inplace = QAction("&Save inplace...", self)
            self.save_inplace.triggered.connect(self.save_as_csv_inplace_per_pos)
            # self.save_inplace.setShortcut("Ctrl+s")
            self.fileMenu.addAction(self.save_inplace)

        self.plot_action = QAction("&Plot...", self)
        self.plot_action.triggered.connect(self.plot)
        self.plot_action.setShortcut("Ctrl+p")
        self.fileMenu.addAction(self.plot_action)

        self.plot_inst_action = QAction("&Plot instantaneous...", self)
        self.plot_inst_action.triggered.connect(self.plot_instantaneous)
        self.plot_inst_action.setShortcut("Ctrl+i")
        self.fileMenu.addAction(self.plot_inst_action)

        self.groupby_action = QAction("&Collapse tracks...", self)
        self.groupby_action.triggered.connect(self.set_projection_mode_tracks)
        self.groupby_action.setShortcut("Ctrl+g")
        self.fileMenu.addAction(self.groupby_action)
        if not self.tracks or not self.collapse_tracks_option:
            self.groupby_action.setEnabled(False)

        if self.population == "pairs":

            self.groupby_pairs_in_neigh_action = QAction(
                "&Collapse pairs in neighborhood...", self
            )
            self.groupby_pairs_in_neigh_action.triggered.connect(
                self.collapse_pairs_in_neigh
            )
            self.fileMenu.addAction(self.groupby_pairs_in_neigh_action)

        if "FRAME" in list(self.data.columns):
            self.groupby_time_action = QAction("&Group by frames...", self)
            self.groupby_time_action.triggered.connect(self.groupby_time_table)
            self.groupby_time_action.setShortcut("Ctrl+t")
            self.fileMenu.addAction(self.groupby_time_action)

        self.query_action = QAction("Query...", self)
        self.query_action.triggered.connect(self.perform_query)
        self.fileMenu.addAction(self.query_action)

        self.delete_action = QAction("&Delete...", self)
        self.delete_action.triggered.connect(self.delete_columns)
        self.delete_action.setShortcut(Qt.Key_Delete)
        self.editMenu.addAction(self.delete_action)

        self.rename_col_action = QAction("&Rename...", self)
        self.rename_col_action.triggered.connect(self.rename_column)
        # self.rename_col_action.setShortcut(Qt.Key_Delete)
        self.editMenu.addAction(self.rename_col_action)

        if self.population == "pairs":
            self.merge_action = QAction("&Merge...", self)
            self.merge_action.triggered.connect(self.merge_tables)
            # self.rename_col_action.setShortcut(Qt.Key_Delete)
            self.editMenu.addAction(self.merge_action)

        self.calibrate_action = QAction("&Calibrate...", self)
        self.calibrate_action.triggered.connect(self.calibrate_selected_feature)
        self.calibrate_action.setShortcut("Ctrl+C")
        self.mathMenu.addAction(self.calibrate_action)

        self.merge_classification_action = QAction("&Merge states...", self)
        self.merge_classification_action.triggered.connect(
            self.merge_classification_features
        )
        self.mathMenu.addAction(self.merge_classification_action)

        self.derivative_action = QAction("&Differentiate...", self)
        self.derivative_action.triggered.connect(self.differenciate_selected_feature)
        self.derivative_action.setShortcut("Ctrl+D")
        self.mathMenu.addAction(self.derivative_action)
        if not self.tracks:
            self.derivative_action.setEnabled(False)

        self.abs_action = QAction("&Absolute value...", self)
        self.abs_action.triggered.connect(self.take_abs_of_selected_feature)
        # self.derivative_action.setShortcut("Ctrl+D")
        self.mathMenu.addAction(self.abs_action)

        self.log_action = QAction("&Log (decimal)...", self)
        self.log_action.triggered.connect(self.take_log_of_selected_feature)
        # self.derivative_action.setShortcut("Ctrl+D")
        self.mathMenu.addAction(self.log_action)

        self.divide_action = QAction("&Divide...", self)
        self.divide_action.triggered.connect(self.divide_signals)
        # self.derivative_action.setShortcut("Ctrl+D")
        self.mathMenu.addAction(self.divide_action)

        self.multiply_action = QAction("&Multiply...", self)
        self.multiply_action.triggered.connect(self.multiply_signals)
        # self.derivative_action.setShortcut("Ctrl+D")
        self.mathMenu.addAction(self.multiply_action)

        self.add_action = QAction("&Add...", self)
        self.add_action.triggered.connect(self.add_signals)
        # self.derivative_action.setShortcut("Ctrl+D")
        self.mathMenu.addAction(self.add_action)

        self.subtract_action = QAction("&Subtract...", self)
        self.subtract_action.triggered.connect(self.subtract_signals)
        # self.derivative_action.setShortcut("Ctrl+D")
        self.mathMenu.addAction(self.subtract_action)

        # self.onehot_action = QAction('&One hot to categorical...', self)
        # self.onehot_action.triggered.connect(self.transform_one_hot_cols_to_categorical)
        # #self.onehot_action.setShortcut("Ctrl+D")
        # self.mathMenu.addAction(self.onehot_action)

    def collapse_pairs_in_neigh(self):

        self.selectNeighWidget = CelldetectiveWidget()
        self.selectNeighWidget.setMinimumWidth(480)
        self.selectNeighWidget.setWindowTitle("Set neighborhood of interest")

        layout = QVBoxLayout()
        self.selectNeighWidget.setLayout(layout)

        self.reference_lbl = QLabel("reference population: ")
        self.reference_pop_cb = QComboBox()
        ref_pops = self.data["reference_population"].unique()
        self.reference_pop_cb.addItems(ref_pops)
        self.reference_pop_cb.currentIndexChanged.connect(self.update_neighborhoods)

        reference_hbox = QHBoxLayout()
        reference_hbox.addWidget(self.reference_lbl, 33)
        reference_hbox.addWidget(self.reference_pop_cb, 66)
        layout.addLayout(reference_hbox)

        self.neigh_lbl = QLabel("neighborhod: ")
        self.neigh_cb = QComboBox()
        neigh_cols = [
            c.replace("status_", "")
            for c in list(
                self.data.loc[
                    self.data["reference_population"]
                    == self.reference_pop_cb.currentText()
                ].columns
            )
            if c.startswith("status_neighborhood")
        ]
        self.neigh_cb.addItems(neigh_cols)

        neigh_hbox = QHBoxLayout()
        neigh_hbox.addWidget(self.neigh_lbl, 33)
        neigh_hbox.addWidget(self.neigh_cb, 66)
        layout.addLayout(neigh_hbox)

        contact_hbox = QHBoxLayout()
        self.contact_only_check = QCheckBox("keep only pairs in contact")
        self.contact_only_check.setChecked(True)
        contact_hbox.addWidget(self.contact_only_check, alignment=Qt.AlignLeft)
        layout.addLayout(contact_hbox)

        self.groupby_pair_rb = QRadioButton("Group by pair")
        self.groupby_reference_rb = QRadioButton("Group by reference")
        self.groupby_pair_rb.setChecked(True)

        groupby_hbox = QHBoxLayout()
        groupby_hbox.addWidget(QLabel("collapse option: "), 33)
        groupby_hbox.addWidget(self.groupby_pair_rb, (100 - 33) // 2)
        groupby_hbox.addWidget(self.groupby_reference_rb, (100 - 33) // 2)
        layout.addLayout(groupby_hbox)

        self.apply_neigh_btn = QPushButton("Set")
        self.apply_neigh_btn.setStyleSheet(self.button_style_sheet)
        self.apply_neigh_btn.clicked.connect(self.prepare_table_at_neighborhood)

        apply_hbox = QHBoxLayout()
        apply_hbox.addWidget(QLabel(""), 33)
        apply_hbox.addWidget(self.apply_neigh_btn, 66)
        layout.addLayout(apply_hbox)

        self.selectNeighWidget.show()
        center_window(self.selectNeighWidget)

    def prepare_table_at_neighborhood(self):

        ref_pop = self.reference_pop_cb.currentText()
        neighborhood = self.neigh_cb.currentText()
        status_neigh = "status_" + neighborhood

        if "self" in neighborhood:
            neighbor_pop = ref_pop

        neigh_col = neighborhood.replace("status_", "")
        if "_(" in neigh_col and ")_" in neigh_col:
            neighbor_pop = neigh_col.split("_(")[-1].split(")_")[0].split("-")[-1]
        else:
            if ref_pop == "targets":
                neighbor_pop = "effectors"
            if ref_pop == "effectors":
                neighbor_pop = "targets"

        from celldetective.neighborhood import extract_neighborhood_in_pair_table

        data = extract_neighborhood_in_pair_table(
            self.data,
            neighborhood_key=neighborhood,
            contact_only=self.contact_only_check.isChecked(),
            reference_population=ref_pop,
        )

        if self.groupby_pair_rb.isChecked():
            self.groupby_cols = ["position", "REFERENCE_ID", "NEIGHBOR_ID"]
        elif self.groupby_reference_rb.isChecked():
            self.groupby_cols = ["position", "REFERENCE_ID"]

        self.current_data = data
        skip_projection = False
        if "reference_tracked" in list(self.current_data.columns):
            print(
                f"{self.current_data['reference_tracked']=} {(self.current_data['reference_tracked']==False)=} {np.all(self.current_data['reference_tracked']==False)=}"
            )
            if np.all(self.current_data["reference_tracked"].astype(bool) == False):
                # reference not tracked
                if self.groupby_reference_rb.isChecked():
                    self.groupby_cols = ["position", "FRAME", "REFERENCE_ID"]
                elif self.groupby_pair_rb.isChecked():
                    print(
                        "The reference cells seem to not be tracked. No collapse can be performed."
                    )
                    skip_projection = True
            else:
                if np.all(self.current_data["neighbors_tracked"].astype(bool) == False):
                    # neighbors not tracked
                    if self.groupby_pair_rb.isChecked():
                        print(
                            "The neighbor cells seem to not be tracked. No collapse can be performed."
                        )
                        skip_projection = True
                    elif self.groupby_reference_rb.isChecked():
                        self.groupby_cols = [
                            "position",
                            "REFERENCE_ID",
                        ]  # think about what would be best

        if not skip_projection:
            self.set_projection_mode_tracks()

    def update_neighborhoods(self):

        neigh_cols = [
            c.replace("status_", "")
            for c in list(
                self.data.loc[
                    self.data["reference_population"]
                    == self.reference_pop_cb.currentText()
                ].columns
            )
            if c.startswith("status_neighborhood")
        ]
        self.neigh_cb.clear()
        self.neigh_cb.addItems(neigh_cols)

    def merge_tables(self):

        df_expanded = expand_pair_table(self.data)
        self.subtable = TableUI(
            df_expanded, "merge", plot_mode="static", population="pairs"
        )
        self.subtable.show()

    def delete_columns(self):

        x = self.table_view.selectedIndexes()
        col_idx = np.unique(np.array([l.column() for l in x]))
        cols = np.array(list(self.data.columns))

        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Question)
        msgBox.setText(
            f"You are about to delete columns {cols[col_idx]}... Do you want to proceed?"
        )
        msgBox.setWindowTitle("Info")
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        return_value = msgBox.exec()
        if return_value == QMessageBox.No:
            return None

        self.data = self.data.drop(list(cols[col_idx]), axis=1)
        self.model = PandasModel(self.data)
        self.table_view.setModel(self.model)

    def rename_column(self):

        x = self.table_view.selectedIndexes()
        col_idx = np.unique(np.array([l.column() for l in x]))

        if len(col_idx) == 0:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Question)
            msg_box.setText(f"Please select a column first.")
            msg_box.setWindowTitle("Warning")
            msg_box.setStandardButtons(QMessageBox.Ok)
            returnValue = msg_box.exec()
            if returnValue == QMessageBox.Ok:
                return None
            else:
                return None

        cols = np.array(list(self.data.columns))
        selected_col = str(cols[col_idx][0])

        self.renameWidget = RenameColWidget(self, selected_col)
        self.renameWidget.show()

    def save_as_csv_inplace_per_pos(self):

        print("Saving each table in its respective position folder...")
        for pos, pos_group in self.data.groupby(["position"]):
            invalid_cols = [
                c for c in list(pos_group.columns) if c.startswith("Unnamed")
            ]
            if len(invalid_cols) > 0:
                pos_group = pos_group.drop(invalid_cols, axis=1)
            pos_group.to_csv(
                pos[0]
                + os.sep.join(
                    ["output", "tables", f"trajectories_{self.population}.csv"]
                ),
                index=False,
            )
        print("Done...")

    def divide_signals(self):

        x = self.table_view.selectedIndexes()
        col_idx = np.unique(np.array([l.column() for l in x]))
        if isinstance(col_idx, (list, np.ndarray)):
            cols = np.array(list(self.data.columns))
            if len(col_idx) > 0:
                selected_col1 = str(cols[col_idx[0]])
                if len(col_idx) > 1:
                    selected_col2 = str(cols[col_idx[1]])
                else:
                    selected_col2 = None
            else:
                selected_col1 = None
                selected_col2 = None
        else:
            selected_col1 = None
            selected_col2 = None

        self.divWidget = OperationOnColsWidget(
            self, column1=selected_col1, column2=selected_col2, operation="divide"
        )
        self.divWidget.show()

    def multiply_signals(self):

        x = self.table_view.selectedIndexes()
        col_idx = np.unique(np.array([l.column() for l in x]))
        if isinstance(col_idx, (list, np.ndarray)):
            cols = np.array(list(self.data.columns))
            if len(col_idx) > 0:
                selected_col1 = str(cols[col_idx[0]])
                if len(col_idx) > 1:
                    selected_col2 = str(cols[col_idx[1]])
                else:
                    selected_col2 = None
            else:
                selected_col1 = None
                selected_col2 = None
        else:
            selected_col1 = None
            selected_col2 = None

        self.mulWidget = OperationOnColsWidget(
            self, column1=selected_col1, column2=selected_col2, operation="multiply"
        )
        self.mulWidget.show()

    def add_signals(self):

        x = self.table_view.selectedIndexes()
        col_idx = np.unique(np.array([l.column() for l in x]))
        if isinstance(col_idx, (list, np.ndarray)):
            cols = np.array(list(self.data.columns))
            if len(col_idx) > 0:
                selected_col1 = str(cols[col_idx[0]])
                if len(col_idx) > 1:
                    selected_col2 = str(cols[col_idx[1]])
                else:
                    selected_col2 = None
            else:
                selected_col1 = None
                selected_col2 = None
        else:
            selected_col1 = None
            selected_col2 = None

        self.addiWidget = OperationOnColsWidget(
            self, column1=selected_col1, column2=selected_col2, operation="add"
        )
        self.addiWidget.show()

    def subtract_signals(self):

        x = self.table_view.selectedIndexes()
        col_idx = np.unique(np.array([l.column() for l in x]))
        if isinstance(col_idx, (list, np.ndarray)):
            cols = np.array(list(self.data.columns))
            if len(col_idx) > 0:
                selected_col1 = str(cols[col_idx[0]])
                if len(col_idx) > 1:
                    selected_col2 = str(cols[col_idx[1]])
                else:
                    selected_col2 = None
            else:
                selected_col1 = None
                selected_col2 = None
        else:
            selected_col1 = None
            selected_col2 = None

        self.subWidget = OperationOnColsWidget(
            self, column1=selected_col1, column2=selected_col2, operation="subtract"
        )
        self.subWidget.show()

    def differenciate_selected_feature(self):

        # check only one col selected and assert is numerical
        # open widget to select window parameters, directionality
        # create new col

        x = self.table_view.selectedIndexes()
        col_idx = np.unique(np.array([l.column() for l in x]))
        if isinstance(col_idx, (list, np.ndarray)):
            cols = np.array(list(self.data.columns))
            if len(col_idx) > 0:
                selected_col = str(cols[col_idx[0]])
            else:
                selected_col = None
        else:
            selected_col = None

        self.diffWidget = DifferentiateColWidget(self, selected_col)
        self.diffWidget.show()

    def take_log_of_selected_feature(self):

        # check only one col selected and assert is numerical
        # open widget to select window parameters, directionality
        # create new col

        x = self.table_view.selectedIndexes()
        col_idx = np.unique(np.array([l.column() for l in x]))
        if isinstance(col_idx, (list, np.ndarray)):
            cols = np.array(list(self.data.columns))
            if len(col_idx) > 0:
                selected_col = str(cols[col_idx[0]])
            else:
                selected_col = None
        else:
            selected_col = None

        self.LogWidget = LogColWidget(self, selected_col)
        self.LogWidget.show()

    def merge_classification_features(self):

        x = self.table_view.selectedIndexes()
        col_idx = np.unique(np.array([l.column() for l in x]))

        col_selection = []
        if isinstance(col_idx, (list, np.ndarray)):
            cols = np.array(list(self.data.columns))
            if len(col_idx) > 0:
                selected_cols = cols[col_idx]
                col_selection.extend(selected_cols)

        # Lazy load MergeGroupWidget
        from celldetective.gui.table_ops._merge_groups import MergeGroupWidget

        self.merge_classification_widget = MergeGroupWidget(self, columns=col_selection)
        self.merge_classification_widget.show()

    def calibrate_selected_feature(self):

        x = self.table_view.selectedIndexes()
        col_idx = np.unique(np.array([l.column() for l in x]))
        if isinstance(col_idx, (list, np.ndarray)):
            cols = np.array(list(self.data.columns))
            if len(col_idx) > 0:
                selected_col = str(cols[col_idx[0]])
            else:
                selected_col = None
        else:
            selected_col = None

        self.calWidget = CalibrateColWidget(self, selected_col)
        self.calWidget.show()

    def take_abs_of_selected_feature(self):

        # check only one col selected and assert is numerical
        # open widget to select window parameters, directionality
        # create new col

        x = self.table_view.selectedIndexes()
        col_idx = np.unique(np.array([l.column() for l in x]))
        if isinstance(col_idx, (list, np.ndarray)):
            cols = np.array(list(self.data.columns))
            if len(col_idx) > 0:
                selected_col = str(cols[col_idx[0]])
            else:
                selected_col = None
        else:
            selected_col = None

        self.absWidget = AbsColWidget(self, selected_col)
        self.absWidget.show()

    def transform_one_hot_cols_to_categorical(self):

        x = self.table_view.selectedIndexes()
        col_idx = np.unique(np.array([l.column() for l in x]))
        selected_cols = None
        if isinstance(col_idx, (list, np.ndarray)):
            cols = np.array(list(self.data.columns))
            if len(col_idx) > 0:
                selected_col = str(cols[col_idx[0]])

        self.mergewidget = MergeOneHotWidget(self, selected_columns=selected_cols)
        self.mergewidget.show()

    def groupby_time_table(self):
        """

        Perform a time average across each track for all features

        """

        num_df = self.data.select_dtypes(include=self.numerics)

        timeseries = num_df.groupby(["FRAME"]).sum().copy()
        timeseries["timeline"] = timeseries.index
        self.subtable = TableUI(
            timeseries, "Group by frames", plot_mode="plot_timeseries"
        )
        self.subtable.show()

    def perform_query(self):
        """

        Perform a time average across each track for all features

        """
        self.query_widget = QueryWidget(self)
        self.query_widget.show()

        # num_df = self.data.select_dtypes(include=self.numerics)

        # timeseries = num_df.groupby("FRAME").mean().copy()
        # timeseries["timeline"] = timeseries.index
        # self.subtable = TableUI(timeseries,"Group by frames", plot_mode="plot_timeseries")
        # self.subtable.show()

    def set_projection_mode_neigh(self):

        self.groupby_cols = [
            "position",
            "reference_population",
            "neighbor_population",
            "NEIGHBOR_ID",
            "FRAME",
        ]
        self.current_data = self.data
        self.set_projection_mode_tracks()

    def set_projection_mode_ref(self):

        self.groupby_cols = [
            "position",
            "reference_population",
            "neighbor_population",
            "REFERENCE_ID",
            "FRAME",
        ]
        self.current_data = self.data
        self.set_projection_mode_tracks()

    def set_projection_mode_tracks(self):

        self.current_data = self.data

        self.projectionWidget = CelldetectiveWidget()
        self.projectionWidget.setMinimumWidth(500)
        self.projectionWidget.setWindowTitle("Set projection mode")

        layout = QVBoxLayout()
        self.projectionWidget.setLayout(layout)

        self.projection_option = QRadioButton("global operation: ")
        self.projection_option.setToolTip(
            "Collapse the cell track measurements with an operation over each track."
        )
        self.projection_option.setChecked(True)
        self.projection_option.toggled.connect(self.enable_projection_options)
        self.projection_op_cb = QComboBox()
        self.projection_op_cb.addItems(
            ["mean", "median", "min", "max", "first", "last", "prod", "sum"]
        )

        projection_layout = QHBoxLayout()
        projection_layout.addWidget(self.projection_option, 33)
        projection_layout.addWidget(self.projection_op_cb, 66)
        layout.addLayout(projection_layout)

        self.event_time_option = QRadioButton("@event time: ")
        self.event_time_option.setToolTip(
            "Pick the measurements at a specific event time."
        )
        self.event_time_option.toggled.connect(self.enable_projection_options)
        self.event_times_cb = QComboBox()
        cols = np.array(self.data.columns)
        time_cols = np.array([c.startswith("t_") for c in cols])
        time_cols = list(cols[time_cols])
        if "t0" in list(self.data.columns):
            time_cols.append("t0")
        self.event_times_cb.addItems(time_cols)
        self.event_times_cb.setEnabled(False)

        event_time_layout = QHBoxLayout()
        event_time_layout.addWidget(self.event_time_option, 33)
        event_time_layout.addWidget(self.event_times_cb, 66)
        layout.addLayout(event_time_layout)

        self.per_status_option = QRadioButton("per status: ")
        self.per_status_option.setToolTip(
            "Collapse the cell track measurements independently for each of the cell state."
        )
        self.per_status_option.toggled.connect(self.enable_projection_options)
        self.per_status_cb = QComboBox()
        self.status_operation = QComboBox()
        self.status_operation.setEnabled(False)
        self.status_operation.addItems(["mean", "median", "min", "max", "prod", "sum"])

        status_cols = np.array(
            [c.startswith("status_") or c.startswith("group_") for c in cols]
        )
        status_cols = list(cols[status_cols])
        if "status" in list(self.data.columns):
            status_cols.append("status")
        self.per_status_cb.addItems(status_cols)
        self.per_status_cb.setEnabled(False)

        per_status_layout = QHBoxLayout()
        per_status_layout.addWidget(self.per_status_option, 33)
        per_status_layout.addWidget(self.per_status_cb, 66)
        layout.addLayout(per_status_layout)

        status_operation_layout = QHBoxLayout()
        status_operation_layout.addWidget(
            QLabel("operation: "), 33, alignment=Qt.AlignRight
        )
        status_operation_layout.addWidget(self.status_operation, 66)
        layout.addLayout(status_operation_layout)

        self.btn_projection_group = QButtonGroup()
        self.btn_projection_group.addButton(self.projection_option)
        self.btn_projection_group.addButton(self.event_time_option)
        self.btn_projection_group.addButton(self.per_status_option)

        apply_layout = QHBoxLayout()

        self.set_projection_btn = QPushButton("Apply")
        self.set_projection_btn.setStyleSheet(self.button_style_sheet)
        self.set_projection_btn.clicked.connect(self.set_proj_mode)
        apply_layout.addWidget(QLabel(""), 33)
        apply_layout.addWidget(self.set_projection_btn, 33)
        apply_layout.addWidget(QLabel(""), 33)
        layout.addLayout(apply_layout)

        self.projectionWidget.show()
        center_window(self.projectionWidget)

    def enable_projection_options(self):

        if self.projection_option.isChecked():
            self.projection_op_cb.setEnabled(True)
            self.event_times_cb.setEnabled(False)
            self.per_status_cb.setEnabled(False)
            self.status_operation.setEnabled(False)
        elif self.event_time_option.isChecked():
            self.projection_op_cb.setEnabled(False)
            self.event_times_cb.setEnabled(True)
            self.per_status_cb.setEnabled(False)
            self.status_operation.setEnabled(False)
        elif self.per_status_option.isChecked():
            self.projection_op_cb.setEnabled(False)
            self.event_times_cb.setEnabled(False)
            self.per_status_cb.setEnabled(True)
            self.status_operation.setEnabled(True)

    def set_1D_plot_params(self):

        self.plot1Dparams = CelldetectiveWidget()
        self.plot1Dparams.setWindowTitle("Set 1D plot parameters")

        layout = QVBoxLayout()
        self.plot1Dparams.setLayout(layout)

        layout.addWidget(QLabel("Representations: "))
        self.hist_check = QCheckBox("histogram")
        self.kde_check = QCheckBox("KDE plot")
        self.count_check = QCheckBox("countplot")
        self.ecdf_check = QCheckBox("ECDF plot")
        self.line_check = QCheckBox("line plot")
        self.scat_check = QCheckBox("scatter plot")
        self.swarm_check = QCheckBox("swarm")
        self.violin_check = QCheckBox("violin")
        self.strip_check = QCheckBox("strip")
        self.box_check = QCheckBox("boxplot")
        self.boxenplot_check = QCheckBox("boxenplot")

        self.sep_line = QHSeperationLine()
        self.pvalue_check = QCheckBox("Compute KS test p-value?")
        self.effect_size_check = QCheckBox("Compute effect size?\n(Cliff's Delta)")

        layout.addWidget(self.hist_check)
        layout.addWidget(self.kde_check)
        layout.addWidget(self.count_check)
        layout.addWidget(self.ecdf_check)
        layout.addWidget(self.line_check)
        layout.addWidget(self.scat_check)
        layout.addWidget(self.swarm_check)
        layout.addWidget(self.violin_check)
        layout.addWidget(self.strip_check)
        layout.addWidget(self.box_check)
        layout.addWidget(self.boxenplot_check)
        layout.addWidget(self.sep_line)
        layout.addWidget(self.pvalue_check)
        layout.addWidget(self.effect_size_check)

        self.x_cb = QSearchableComboBox()
        self.x_cb.addItems(["--"] + list(self.data.columns))

        self.y_cb = QSearchableComboBox()
        self.y_cb.addItems(["--"] + list(self.data.columns))

        self.hue_cb = QSearchableComboBox()
        self.hue_cb.addItems(["--"] + list(self.data.columns))
        idx = self.hue_cb.findText("--")
        self.hue_cb.setCurrentIndex(idx)

        # Set selected column

        try:
            x = self.table_view.selectedIndexes()
            col_idx = np.array([l.column() for l in x])
            row_idx = np.array([l.row() for l in x])
            column_names = self.data.columns
            unique_cols = np.unique(col_idx)[0]
            y = column_names[unique_cols]
            idx = self.y_cb.findText(y)
            self.y_cb.setCurrentIndex(idx)
        except:
            pass

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("x: "), 33)
        hbox.addWidget(self.x_cb, 66)
        layout.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("y: "), 33)
        hbox.addWidget(self.y_cb, 66)
        layout.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("hue: "), 33)
        hbox.addWidget(self.hue_cb, 66)
        layout.addLayout(hbox)

        from matplotlib import colormaps
        import matplotlib.cm

        self.cmap_cb = QColormapComboBox()
        all_cms = list(colormaps)
        for cm in all_cms:
            if hasattr(matplotlib.cm, str(cm).lower()):
                try:
                    self.cmap_cb.addColormap(cm.lower())
                except:
                    pass

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("colormap: "), 33)
        hbox.addWidget(self.cmap_cb, 66)
        layout.addLayout(hbox)

        self.plot1d_btn = QPushButton("set")
        self.plot1d_btn.setStyleSheet(self.button_style_sheet)
        self.plot1d_btn.clicked.connect(self.plot1d)
        layout.addWidget(self.plot1d_btn)

        self.plot1Dparams.show()
        center_window(self.plot1Dparams)

    def plot1d(self):

        self.x_option = False
        if self.x_cb.currentText() != "--":
            self.x_option = True
            self.x = self.x_cb.currentText()

        import matplotlib.pyplot as plt
        import seaborn as sns
        import matplotlib.cm as mcm

        self.fig, self.ax = plt.subplots(1, 1, figsize=(4, 3))
        self.plot1dWindow = FigureCanvas(self.fig, title="scatter")
        self.ax.clear()

        cmap = getattr(mcm, self.cmap_cb.currentText())

        try:
            self.hue_variable = self.hue_cb.currentText()
            colors = [
                cmap(i / len(self.data[self.hue_variable].unique()))
                for i in range(len(self.data[self.hue_variable].unique()))
            ]
        except:
            colors = None

        if self.hue_cb.currentText() == "--":
            self.hue_variable = None

        if self.y_cb.currentText() == "--":
            self.y = None
        else:
            self.y = self.y_cb.currentText()

        if self.x_cb.currentText() == "--":
            self.x = None
        else:
            self.x = self.x_cb.currentText()

        legend = True

        if self.hist_check.isChecked():
            if self.x is not None:
                sns.histplot(
                    data=self.data,
                    x=self.x,
                    hue=self.hue_variable,
                    legend=legend,
                    ax=self.ax,
                    palette=colors,
                    kde=True,
                    common_norm=False,
                    stat="density",
                )
                legend = False
            elif self.x is None and self.y is not None:
                sns.histplot(
                    data=self.data,
                    x=self.y,
                    hue=self.hue_variable,
                    legend=legend,
                    ax=self.ax,
                    palette=colors,
                    kde=True,
                    common_norm=False,
                    stat="density",
                )
                legend = False
            else:
                pass

        if self.kde_check.isChecked():
            if self.x is not None:
                sns.kdeplot(
                    data=self.data,
                    x=self.x,
                    hue=self.hue_variable,
                    legend=legend,
                    ax=self.ax,
                    palette=colors,
                    cut=0,
                )
                legend = False
            elif self.x is None and self.y is not None:
                sns.kdeplot(
                    data=self.data,
                    x=self.y,
                    hue=self.hue_variable,
                    legend=legend,
                    ax=self.ax,
                    palette=colors,
                    cut=0,
                )
                legend = False
            else:
                pass

        if self.count_check.isChecked():
            sns.countplot(
                data=self.data,
                x=self.x,
                hue=self.hue_variable,
                legend=legend,
                ax=self.ax,
                palette=colors,
            )
            legend = False

        if self.ecdf_check.isChecked():
            if self.x is not None:
                sns.ecdfplot(
                    data=self.data,
                    x=self.x,
                    hue=self.hue_variable,
                    legend=legend,
                    ax=self.ax,
                    palette=colors,
                )
                legend = False
            elif self.x is None and self.y is not None:
                sns.ecdfplot(
                    data=self.data,
                    x=self.y,
                    hue=self.hue_variable,
                    legend=legend,
                    ax=self.ax,
                    palette=colors,
                )
                legend = False
            else:
                pass

        if self.line_check.isChecked():
            if self.x_option:
                sns.lineplot(
                    data=self.data,
                    x=self.x,
                    y=self.y,
                    hue=self.hue_variable,
                    legend=legend,
                    ax=self.ax,
                    palette=colors,
                )
                legend = False
            else:
                print("please provide a -x variable...")
                pass

        if self.scat_check.isChecked():
            if self.x_option:
                sns.scatterplot(
                    data=self.data,
                    x=self.x,
                    y=self.y,
                    hue=self.hue_variable,
                    legend=legend,
                    ax=self.ax,
                    palette=colors,
                )
                legend = False
            else:
                print("please provide a -x variable...")
                pass

        if self.swarm_check.isChecked():
            if self.x_option:
                sns.swarmplot(
                    data=self.data,
                    x=self.x,
                    y=self.y,
                    dodge=True,
                    hue=self.hue_variable,
                    legend=legend,
                    ax=self.ax,
                    palette=colors,
                )
                legend = False
            else:
                sns.swarmplot(
                    data=self.data,
                    y=self.y,
                    dodge=True,
                    hue=self.hue_variable,
                    legend=legend,
                    ax=self.ax,
                    palette=colors,
                )
                legend = False

        if self.violin_check.isChecked():
            if self.x_option:
                sns.violinplot(
                    data=self.data,
                    x=self.x,
                    y=self.y,
                    dodge=True,
                    ax=self.ax,
                    hue=self.hue_variable,
                    legend=legend,
                    palette=colors,
                )
                legend = False
            else:
                sns.violinplot(
                    data=self.data,
                    y=self.y,
                    dodge=True,
                    hue=self.hue_variable,
                    legend=legend,
                    ax=self.ax,
                    palette=colors,
                    cut=0,
                )
                legend = False

        if self.box_check.isChecked():
            if self.x_option:
                sns.boxplot(
                    data=self.data,
                    x=self.x,
                    y=self.y,
                    dodge=True,
                    hue=self.hue_variable,
                    legend=legend,
                    ax=self.ax,
                    fill=False,
                    palette=colors,
                    linewidth=2,
                )
                legend = False
            else:
                sns.boxplot(
                    data=self.data,
                    y=self.y,
                    dodge=True,
                    hue=self.hue_variable,
                    legend=legend,
                    ax=self.ax,
                    fill=False,
                    palette=colors,
                    linewidth=2,
                )
                legend = False

        if self.boxenplot_check.isChecked():
            if self.x_option:
                sns.boxenplot(
                    data=self.data,
                    x=self.x,
                    y=self.y,
                    dodge=True,
                    hue=self.hue_variable,
                    legend=legend,
                    ax=self.ax,
                    fill=False,
                    palette=colors,
                    linewidth=2,
                )
                legend = False
            else:
                sns.boxenplot(
                    data=self.data,
                    y=self.y,
                    dodge=True,
                    hue=self.hue_variable,
                    legend=legend,
                    ax=self.ax,
                    fill=False,
                    palette=colors,
                    linewidth=2,
                )
                legend = False

        if self.strip_check.isChecked():
            if self.x_option:
                sns.stripplot(
                    data=self.data,
                    x=self.x,
                    y=self.y,
                    dodge=True,
                    ax=self.ax,
                    hue=self.hue_variable,
                    legend=legend,
                    palette=colors,
                )
                legend = False
            else:
                sns.stripplot(
                    data=self.data,
                    y=self.y,
                    dodge=True,
                    ax=self.ax,
                    hue=self.hue_variable,
                    legend=legend,
                    palette=colors,
                )
                legend = False

        plt.tight_layout()
        self.fig.set_facecolor("none")  # or 'None'
        self.fig.canvas.setStyleSheet("background-color: transparent;")
        self.plot1dWindow.canvas.draw()
        self.plot1dWindow.show()

        if self.effect_size_check.isChecked():
            self.compute_effect_size()
        if self.pvalue_check.isChecked():
            self.compute_pvalue()

    def extract_groupby_cols(self):

        x = self.x
        y = self.y
        hue_variable = self.hue_variable

        if (
            self.hist_check.isChecked()
            or self.ecdf_check.isChecked()
            or self.kde_check.isChecked()
        ):
            y = self.x
            x = None

        groupby_cols = []
        if x is not None:
            groupby_cols.append(x)
        if hue_variable is not None:
            groupby_cols.append(hue_variable)

        return groupby_cols, y

    def compute_effect_size(self):

        if self.count_check.isChecked() or self.scat_check.isChecked():
            print(
                "Please select a valid plot representation to compute effect size (histogram, boxplot, etc.)..."
            )
            return None

        groupby_cols, y = self.extract_groupby_cols()
        pivot = test_2samp_generic(
            self.data, feature=y, groupby_cols=groupby_cols, method="cliffs_delta"
        )
        self.effect_size_table = PivotTableUI(
            pivot, title="Effect size (Cliff's Delta)", mode="cliff"
        )
        self.effect_size_table.show()

    def compute_pvalue(self):

        if self.count_check.isChecked() or self.scat_check.isChecked():
            print(
                "Please select a valid plot representation to compute effect size (histogram, boxplot, etc.)..."
            )
            return None

        groupby_cols, y = self.extract_groupby_cols()
        pivot = test_2samp_generic(
            self.data, feature=y, groupby_cols=groupby_cols, method="ks_2samp"
        )
        self.pval_table = PivotTableUI(
            pivot, title="p-value (1-sided KS test)", mode="pvalue"
        )
        self.pval_table.show()

    def set_proj_mode(self):

        self.static_columns = [
            "well_index",
            "well_name",
            "pos_name",
            "position",
            "well",
            "status",
            "t0",
            "class",
            "cell_type",
            "concentration",
            "antibody",
            "pharmaceutical_agent",
            "TRACK_ID",
            "position",
            "neighbor_population",
            "reference_population",
            "NEIGHBOR_ID",
            "REFERENCE_ID",
            "FRAME",
        ]

        if self.projection_option.isChecked():

            self.projection_mode = self.projection_op_cb.currentText()
            op = getattr(
                self.current_data.groupby(self.groupby_cols), self.projection_mode
            )
            group_table = op(self.current_data.groupby(self.groupby_cols))

            for c in self.static_columns:
                try:
                    group_table[c] = self.current_data.groupby(self.groupby_cols)[
                        c
                    ].apply(lambda x: x.unique()[0])
                except Exception as e:
                    print(e)
                    pass

            if self.population == "pairs":
                for col in reversed(
                    self.groupby_cols
                ):  # ['neighbor_population', 'reference_population', 'NEIGHBOR_ID', 'REFERENCE_ID']
                    if col in group_table:
                        first_column = group_table.pop(col)
                        group_table.insert(0, col, first_column)
            else:
                for col in ["TRACK_ID"]:
                    first_column = group_table.pop(col)
                    group_table.insert(0, col, first_column)
                group_table.pop("FRAME")

        elif self.event_time_option.isChecked():

            time_of_interest = self.event_times_cb.currentText()
            self.projection_mode = f"measurements at {time_of_interest}"
            new_table = []
            for tid, group in self.current_data.groupby(self.groupby_cols):
                time = group[time_of_interest].values[0]
                if time == time:
                    time = floor(time)  # floor for onset
                else:
                    continue
                frames = group["FRAME"].values
                values = group.loc[group["FRAME"] == time, :].to_numpy()
                if len(values) > 0:
                    values = dict(zip(list(self.current_data.columns), values[0]))
                    for k, c in enumerate(self.groupby_cols):
                        values.update({c: tid[k]})
                    new_table.append(values)
            import pandas as pd

            group_table = pd.DataFrame(new_table)
            if self.population == "pairs":
                for col in self.groupby_cols[1:]:
                    first_column = group_table.pop(col)
                    group_table.insert(0, col, first_column)
            else:
                for col in ["TRACK_ID"]:
                    first_column = group_table.pop(col)
                    group_table.insert(0, col, first_column)

            group_table = group_table.sort_values(
                by=self.groupby_cols + ["FRAME"], ignore_index=True
            )
            group_table = group_table.reset_index(drop=True)

        elif self.per_status_option.isChecked():
            self.projection_mode = self.status_operation.currentText()
            group_table = collapse_trajectories_by_status(
                self.current_data,
                status=self.per_status_cb.currentText(),
                population=self.population,
                projection=self.status_operation.currentText(),
                groupby_columns=self.groupby_cols,
            )

        self.subtable = TableUI(
            group_table,
            f"Group by tracks: {self.projection_mode}",
            plot_mode="static",
            collapse_tracks_option=False,
        )
        self.subtable.show()

        self.projectionWidget.close()

    # def groupby_track_table(self):

    # 	"""

    # 	Perform a time average across each track for all features

    # 	"""

    # 	self.subtable = TrajectoryTablePanel(self.data.groupby("TRACK_ID").mean(),"Group by tracks", plot_mode="scatter")
    # 	self.subtable.show()

    def _createMenuBar(self):
        menuBar = self.menuBar()
        self.fileMenu = QMenu("&File", self)
        menuBar.addMenu(self.fileMenu)
        self.editMenu = QMenu("&Edit", self)
        menuBar.addMenu(self.editMenu)
        self.mathMenu = QMenu("&Math", self)
        menuBar.addMenu(self.mathMenu)

    def save_as_csv(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save as .csv",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options,
        )
        if file_name:
            if not file_name.endswith(".csv"):
                file_name += ".csv"
            invalid_cols = [
                c for c in list(self.data.columns) if c.startswith("Unnamed")
            ]
            if len(invalid_cols) > 0:
                self.data = self.data.drop(invalid_cols, axis=1)
            self.data.to_csv(file_name, index=False)

    def plot_instantaneous(self):

        if self.plot_mode == "plot_track_signals":
            self.plot_mode = "static"
            self.plot()
            self.plot_mode = "plot_track_signals"
        elif self.plot_mode == "static":
            self.plot()

    def plot(self):
        import matplotlib.pyplot as plt

        if self.plot_mode == "static":

            x = self.table_view.selectedIndexes()
            col_idx = [l.column() for l in x]
            row_idx = [l.row() for l in x]
            column_names = self.data.columns
            unique_cols = np.unique(col_idx)

            if len(unique_cols) == 1 or len(unique_cols) == 0:
                self.set_1D_plot_params()

            if len(unique_cols) == 2:

                print("two columns, plot mode")
                x1 = test_bool_array(self.data.iloc[row_idx, unique_cols[0]])
                x2 = test_bool_array(self.data.iloc[row_idx, unique_cols[1]])

                self.fig, self.ax = plt.subplots(1, 1, figsize=(4, 3))
                self.scatter_wdw = FigureCanvas(self.fig, title="scatter")
                self.ax.clear()
                self.ax.scatter(x1, x2)
                self.ax.set_xlabel(column_names[unique_cols[0]])
                self.ax.set_ylabel(column_names[unique_cols[1]])
                plt.tight_layout()
                self.fig.set_facecolor("none")  # or 'None'
                self.fig.canvas.setStyleSheet("background-color: transparent;")
                self.scatter_wdw.canvas.draw()
                self.scatter_wdw.show()

            else:
                print("please select less columns")

        elif self.plot_mode == "plot_timeseries":
            print("mode plot frames")
            x = self.table_view.selectedIndexes()
            col_idx = np.array([l.column() for l in x])
            row_idx = np.array([l.row() for l in x])
            column_names = self.data.columns
            unique_cols = np.unique(col_idx)

            self.fig, self.ax = plt.subplots(1, 1, figsize=(4, 3))
            self.plot_wdw = FigureCanvas(self.fig, title="scatter")
            self.ax.clear()
            for k in range(len(unique_cols)):
                row_idx_i = row_idx[np.where(col_idx == unique_cols[k])[0]]
                y = self.data.iloc[row_idx_i, unique_cols[k]]
                self.ax.plot(
                    self.data["timeline"][row_idx_i],
                    y,
                    label=column_names[unique_cols[k]],
                )

            self.ax.legend()
            self.ax.set_xlabel("time [frame]")
            self.ax.set_ylabel(self.title)
            plt.tight_layout()
            self.fig.set_facecolor("none")  # or 'None'
            self.fig.canvas.setStyleSheet("background-color: transparent;")
            self.plot_wdw.canvas.draw()
            plt.show()

        elif self.plot_mode == "plot_track_signals":

            print("mode plot track signals")
            print("we plot here")

            x = self.table_view.selectedIndexes()
            col_idx = np.array([l.column() for l in x])
            row_idx = np.array([l.row() for l in x])
            column_names = self.data.columns
            unique_cols = np.unique(col_idx)

            if len(unique_cols) > 2:
                fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
                for k in range(len(unique_cols)):

                    row_idx_i = row_idx[np.where(col_idx == unique_cols[k])[0]]
                    y = self.data.iloc[row_idx_i, unique_cols[k]]
                    print(unique_cols[k])
                    for w, well_group in self.data.groupby(["well_name"]):
                        for pos, pos_group in well_group.groupby(["pos_name"]):
                            for tid, group_track in pos_group.groupby(
                                self.groupby_cols[1:]
                            ):
                                ax.plot(
                                    group_track["FRAME"],
                                    group_track[column_names[unique_cols[k]]],
                                    label=column_names[unique_cols[k]],
                                )
                    # ax.plot(self.data["FRAME"][row_idx_i], y, label=column_names[unique_cols[k]])
                ax.legend()
                ax.set_xlabel("time [frame]")
                ax.set_ylabel(self.title)
                plt.tight_layout()
                plt.show(block=False)

            if len(unique_cols) == 2:

                self.fig, self.ax = plt.subplots(1, 1, figsize=(4, 3))
                self.scatter_wdw = FigureCanvas(self.fig, title="scatter")
                self.ax.clear()
                for tid, group in self.data.groupby(self.groupby_cols[1:]):
                    self.ax.plot(
                        group[column_names[unique_cols[0]]],
                        group[column_names[unique_cols[1]]],
                        marker="o",
                    )
                self.ax.set_xlabel(column_names[unique_cols[0]])
                self.ax.set_ylabel(column_names[unique_cols[1]])
                plt.tight_layout()
                self.fig.set_facecolor("none")  # or 'None'
                self.fig.canvas.setStyleSheet("background-color: transparent;")
                self.scatter_wdw.canvas.draw()
                self.scatter_wdw.show()

            if len(unique_cols) == 1:

                self.fig, self.ax = plt.subplots(1, 1, figsize=(4, 3))
                self.plot_wdw = FigureCanvas(self.fig, title="scatter")
                self.ax.clear()

                # if 't0' in list(self.data.columns):
                # 	ref_time_col = 't0'
                # else:
                # 	ref_time_col = 'FRAME'

                for w, well_group in self.data.groupby(["well_name"]):
                    for pos, pos_group in well_group.groupby(["pos_name"]):
                        for tid, group_track in pos_group.groupby(
                            self.groupby_cols[1:]
                        ):
                            self.ax.plot(
                                group_track["FRAME"],
                                group_track[column_names[unique_cols[0]]],
                                c="k",
                                alpha=0.1,
                            )
                self.ax.set_xlabel(r"$t$ [frame]")
                self.ax.set_ylabel(column_names[unique_cols[0]])
                plt.tight_layout()
                self.fig.set_facecolor("none")  # or 'None'
                self.fig.canvas.setStyleSheet("background-color: transparent;")
                self.plot_wdw.canvas.draw()
                self.plot_wdw.show()
