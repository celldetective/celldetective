from PyQt5.QtWidgets import (
    QRadioButton,
    QButtonGroup,
    QTableView,
    QAction,
    QMenu,
    QMessageBox,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QComboBox,
    QLabel,
    QCheckBox,
    QMessageBox,
    QApplication,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor
from typing import Optional, Any, List, Tuple
import pandas as pd
from superqt import QSearchableComboBox

from celldetective.gui.gui_utils import (
    PandasModel,
)
from celldetective.gui.base.utils import center_window
from celldetective.relative_measurements import expand_pair_table
import numpy as np
import os
from celldetective.gui.base.components import (
    CelldetectiveWidget,
    CelldetectiveMainWindow,
    QHSeperationLine,
)
from math import floor
import re
from celldetective import get_logger
from celldetective.utils.stats import test_2samp_generic
from celldetective.utils.types import test_bool_array
from celldetective.utils.data_cleaning import collapse_trajectories_by_status

logger = get_logger(__name__)


class PivotTableUI(CelldetectiveWidget):

    def __init__(
        self,
        data: pd.DataFrame,
        title: str = "",
        mode: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the PivotTableUI.

        Parameters
        ----------
        data : pandas.DataFrame
            The pivot table data.
        title : str, optional
            The window title.
        mode : str, optional
            The coloring mode ('cliff', 'pvalue', or None).
        *args, **kwargs
            Additional arguments for CelldetectiveWidget.
        """

        CelldetectiveWidget.__init__(self, *args, **kwargs)

        self.data = data
        self.title = title
        self.mode = mode

        self.setWindowTitle(title)
        logger.debug(f"Pivot table to show: {self.data.shape}")

        self.table = QTableView(self)

        self.v_layout = QVBoxLayout()
        self.information_label = QLabel("Information about color code...")

        # Export button
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self.export_data)

        self.v_layout.addWidget(self.information_label)
        self.v_layout.addWidget(self.table)
        self.v_layout.addWidget(self.export_btn)
        self.setLayout(self.v_layout)

        self.showdata()

        if self.mode == "cliff":
            self.color_cells_cliff()
        elif self.mode == "pvalue":
            self.color_cells_pvalue()

        self.table.resizeColumnsToContents()
        self.adjust_window_size()
        self.setAttribute(Qt.WA_DeleteOnClose)
        center_window(self)

    def showdata(self) -> None:
        """
        Display the data in the table view.
        """
        self.model = PandasModel(self.data)
        self.table.setModel(self.model)
        self.table.horizontalHeader().setSectionsMovable(True)
        self.table.horizontalHeader().setDragEnabled(True)
        self.table.horizontalHeader().setDragDropMode(self.table.InternalMove)

    def export_data(self) -> None:
        """
        Export the pivot table data to a CSV file.
        """
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Pivot Table",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options,
        )

        if file_name:
            if not file_name.endswith(".csv"):
                file_name += ".csv"

            try:
                # Get visual column order from header mapped to the original GUI model
                header = self.table.horizontalHeader()
                visual_cols = [
                    self.model._data.columns[header.logicalIndex(i)]
                    for i in range(header.count())
                ]
                data_sorted = self.data[visual_cols]

                # Save with index because pivot tables usually have meaningful indices
                data_sorted.to_csv(file_name, index=True)
                logger.info(f"Pivot table exported to {file_name}")
            except Exception as e:
                logger.error(f"Failed to export pivot table: {e}")
                QMessageBox.critical(
                    self, "Export Error", f"Failed to export data: {str(e)}"
                )

    def adjust_window_size(self) -> None:
        """
        Auto-adjust the window size to fit the content, capped at 80% of screen size.
        """
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()

        # Calculate content size
        # Width: vertical header width + sum of column widths + padding
        v_header_width = self.table.verticalHeader().width()
        h_header_length = self.table.horizontalHeader().length()
        content_width = (
            v_header_width + h_header_length + 40
        )  # +40 for scrollbar/padding

        # Height: horizontal header height + sum of row heights + padding + extra widgets
        h_header_height = self.table.horizontalHeader().height()
        v_header_length = self.table.verticalHeader().length()

        # Estimate height of other widgets in layout (label + export button + margins)
        # This is an approximation
        extra_widgets_height = 100

        content_height = h_header_height + v_header_length + extra_widgets_height

        # Get screen geometry
        screen = QApplication.primaryScreen().availableGeometry()
        max_width = int(screen.width() * 0.8)
        max_height = int(screen.height() * 0.8)

        # Cap the size
        new_width = min(content_width, max_width)
        new_height = min(content_height, max_height)

        # Ensure minimum size
        new_width = max(new_width, 300)
        new_height = max(new_height, 200)

        self.resize(new_width, new_height)

    def set_cell_color(self, row: int, column: int, color: str = "red") -> None:
        """
        Set the background color of a specific cell.

        Parameters
        ----------
        row : int
            Row index.
        column : int
            Column index.
        color : str, optional
            Color name or hex code. Default is 'red'.
        """
        self.model.change_color(
            row, column, QBrush(QColor(color))
        )  # eval(f"Qt.{color}")

    def color_cells_cliff(self) -> None:
        """
        Color cells based on Cliff's Delta values.
        """

        color_codes = {
            "negligible": "#eff3ff",  # Green
            "small": "#bdd7e7",  # Yellow
            "medium": "#6baed6",  # Orange
            "large": "#2171b5",  # Red
        }

        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                value = self.data.iloc[i, j]
                abs_value = abs(value)
                if abs_value < 0.147:
                    self.set_cell_color(i, j, color_codes["negligible"])
                elif abs_value < 0.33:
                    self.set_cell_color(i, j, color_codes["small"])
                elif abs_value < 0.474:
                    self.set_cell_color(i, j, color_codes["medium"])
                elif abs_value >= 0.474:
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

    def color_cells_pvalue(self) -> None:
        """
        Color cells based on p-values.
        """

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
        data: pd.DataFrame,
        title: str,
        population: str = "targets",
        plot_mode: str = "plot_track_signals",
        save_inplace_option: bool = False,
        collapse_tracks_option: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the TableUI.

        Parameters
        ----------
        data : pandas.DataFrame
            The data to display.
        title : str
            The window title.
        population : str, optional
            The cell population name. Default is 'targets'.
        plot_mode : str, optional
            The plotting mode. Default is 'plot_track_signals'.
        save_inplace_option : bool, optional
            Whether to allow saving inplace. Default is False.
        collapse_tracks_option : bool, optional
            Whether to allow collapsing tracks. Default is True.
        *args, **kwargs
            Additional arguments for CelldetectiveMainWindow.
        """

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
        self.table_view.horizontalHeader().setSectionsMovable(True)
        self.table_view.horizontalHeader().setDragEnabled(True)
        self.table_view.horizontalHeader().setDragDropMode(self.table_view.InternalMove)

    def resizeEvent(self, event: Any) -> None:
        """
        Handle resize event to adjust layout.

        Parameters
        ----------
        event : QResizeEvent
            The resize event.
        """

        super().resizeEvent(event)

        try:
            self.fig.tight_layout()
        except AttributeError:
            # fig not yet created
            pass

    def _get_selected_columns(self, max_cols: Optional[int] = None) -> List[str]:
        """
        Get selected column names from the table view.

        Parameters
        ----------
        max_cols : int, optional
            Maximum number of columns to return. Returns all if None.

        Returns
        -------
        list
            List of selected column names.
        """
        x = self.table_view.selectedIndexes()
        col_idx = np.unique(np.array([l.column() for l in x]))
        cols = np.array(list(self.data.columns))
        result = []
        if len(col_idx) > 0:
            for i in col_idx:
                result.append(str(cols[i]))
        if max_cols is not None:
            return result[:max_cols]
        return result

    def _create_actions(self) -> None:
        """
        Create menu actions.
        """

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

        self.bin_action = QAction("&Bin...", self)
        self.bin_action.triggered.connect(self.bin_selected_feature)
        self.mathMenu.addAction(self.bin_action)

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

    def collapse_pairs_in_neigh(self) -> None:
        """
        Open the widget to collapse pairs in a specific neighborhood.
        """

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

    def prepare_table_at_neighborhood(self) -> None:
        """
        Prepare the data table for the selected neighborhood and collapse options.
        """

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

    def update_neighborhoods(self) -> None:
        """
        Update the available neighborhoods based on the selected reference population.
        """

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

    def merge_tables(self) -> None:
        """
        Merge tables for pairs.
        """

        df_expanded = expand_pair_table(self.data)
        self.subtable = TableUI(
            df_expanded, "merge", plot_mode="static", population="pairs"
        )
        self.subtable.show()

    def delete_columns(self) -> Optional[None]:
        """
        Delete selected columns from the table.
        """

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

    def rename_column(self) -> Optional[None]:
        """
        Rename the selected column.
        """

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

        from celldetective.gui.table_ops._rename_col import RenameColWidget

        self.renameWidget = RenameColWidget(self, selected_col)
        self.renameWidget.show()

    def save_as_csv_inplace_per_pos(self) -> None:
        """
        Save each position's table in its respective folder.

        Splits the current dataframe by 'position' and saves each subset
        to the corresponding 'output/tables' directory.
        """
        logger.info("Saving each table in its respective position folder...")

        # Get visual column order from header mapped to the original GUI model
        header = self.table_view.horizontalHeader()
        visual_cols = [
            self.model._data.columns[header.logicalIndex(i)]
            for i in range(header.count())
        ]

        for pos, pos_group in self.data.groupby(["position"]):
            invalid_cols = [
                c for c in list(pos_group.columns) if c.startswith("Unnamed")
            ]
            if len(invalid_cols) > 0:
                pos_group = pos_group.drop(invalid_cols, axis=1)

            # Filter and reorder by available columns (invalid ones dropped)
            valid_visual_cols = [c for c in visual_cols if c in pos_group.columns]
            pos_group = pos_group[valid_visual_cols]

            pos_group.to_csv(
                pos[0]
                + os.sep.join(
                    ["output", "tables", f"trajectories_{self.population}.csv"]
                ),
                index=False,
            )
        logger.info("Done saving tables.")

    def divide_signals(self) -> None:
        """
        Divide two selected signal columns.
        """
        selected = self._get_selected_columns(max_cols=2)
        selected_col1 = selected[0] if len(selected) > 0 else None
        selected_col2 = selected[1] if len(selected) > 1 else None

        from celldetective.gui.table_ops._maths import OperationOnColsWidget

        self.divWidget = OperationOnColsWidget(
            self, column1=selected_col1, column2=selected_col2, operation="divide"
        )
        self.divWidget.show()

    def multiply_signals(self) -> None:
        """
        Multiply two selected signal columns.
        """
        selected = self._get_selected_columns(max_cols=2)
        selected_col1 = selected[0] if len(selected) > 0 else None
        selected_col2 = selected[1] if len(selected) > 1 else None

        from celldetective.gui.table_ops._maths import OperationOnColsWidget

        self.mulWidget = OperationOnColsWidget(
            self, column1=selected_col1, column2=selected_col2, operation="multiply"
        )
        self.mulWidget.show()

    def bin_selected_feature(self) -> None:
        """Open widget to bin the selected column."""
        selected = self._get_selected_columns(max_cols=1)
        selected_col = selected[0] if selected else None

        from celldetective.gui.table_ops._maths import BinColWidget

        self.binWidget = BinColWidget(self, selected_col)
        self.binWidget.show()

    def add_signals(self) -> None:
        """
        Add two selected signal columns.
        """
        selected = self._get_selected_columns(max_cols=2)
        selected_col1 = selected[0] if len(selected) > 0 else None
        selected_col2 = selected[1] if len(selected) > 1 else None

        from celldetective.gui.table_ops._maths import OperationOnColsWidget

        self.addiWidget = OperationOnColsWidget(
            self, column1=selected_col1, column2=selected_col2, operation="add"
        )
        self.addiWidget.show()

    def subtract_signals(self) -> None:
        """
        Subtract two selected signal columns.
        """
        selected = self._get_selected_columns(max_cols=2)
        selected_col1 = selected[0] if len(selected) > 0 else None
        selected_col2 = selected[1] if len(selected) > 1 else None

        from celldetective.gui.table_ops._maths import OperationOnColsWidget

        self.subWidget = OperationOnColsWidget(
            self, column1=selected_col1, column2=selected_col2, operation="subtract"
        )
        self.subWidget.show()

    def differenciate_selected_feature(self) -> None:
        """Open widget to differentiate the selected column."""
        selected = self._get_selected_columns(max_cols=1)
        selected_col = selected[0] if selected else None

        from celldetective.gui.table_ops._maths import DifferentiateColWidget

        self.diffWidget = DifferentiateColWidget(self, selected_col)
        self.diffWidget.show()

    def take_log_of_selected_feature(self) -> None:
        """Open widget to take log of the selected column."""
        selected = self._get_selected_columns(max_cols=1)
        selected_col = selected[0] if selected else None

        from celldetective.gui.table_ops._maths import LogColWidget

        self.LogWidget = LogColWidget(self, selected_col)
        self.LogWidget.show()

    def merge_classification_features(self) -> None:
        """Open widget to merge selected classification columns."""
        col_selection = self._get_selected_columns()

        # Lazy load MergeGroupWidget
        from celldetective.gui.table_ops._merge_groups import MergeGroupWidget

        self.merge_classification_widget = MergeGroupWidget(self, columns=col_selection)
        self.merge_classification_widget.show()

    def calibrate_selected_feature(self) -> None:
        """Open widget to calibrate the selected column."""
        selected = self._get_selected_columns(max_cols=1)
        selected_col = selected[0] if selected else None

        from celldetective.gui.table_ops._maths import CalibrateColWidget

        self.calWidget = CalibrateColWidget(self, selected_col)
        self.calWidget.show()

    def take_abs_of_selected_feature(self) -> None:
        """Open widget to take absolute value of the selected column."""
        selected = self._get_selected_columns(max_cols=1)
        selected_col = selected[0] if selected else None

        from celldetective.gui.table_ops._maths import AbsColWidget

        self.absWidget = AbsColWidget(self, selected_col)
        self.absWidget.show()

    def transform_one_hot_cols_to_categorical(self) -> None:
        """
        Transform one-hot encoded columns to a single categorical column.
        """

        x = self.table_view.selectedIndexes()
        col_idx = np.unique(np.array([l.column() for l in x]))
        selected_cols = None
        if isinstance(col_idx, (list, np.ndarray)):
            cols = np.array(list(self.data.columns))
            if len(col_idx) > 0:
                selected_col = str(cols[col_idx[0]])

        from celldetective.gui.table_ops._merge_one_hot import MergeOneHotWidget

        self.mergewidget = MergeOneHotWidget(self, selected_columns=selected_cols)
        self.mergewidget.show()

    def groupby_time_table(self) -> None:
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

    def perform_query(self) -> None:
        """

        Perform a time average across each track for all features

        """
        from celldetective.gui.table_ops._query_table import QueryWidget

        self.query_widget = QueryWidget(self)
        self.query_widget.show()

        # num_df = self.data.select_dtypes(include=self.numerics)

        # timeseries = num_df.groupby("FRAME").mean().copy()
        # timeseries["timeline"] = timeseries.index
        # self.subtable = TableUI(timeseries,"Group by frames", plot_mode="plot_timeseries")
        # self.subtable.show()

    def set_projection_mode_neigh(self) -> None:
        """
        Set projection mode for neighbors.
        """

        self.groupby_cols = [
            "position",
            "reference_population",
            "neighbor_population",
            "NEIGHBOR_ID",
            "FRAME",
        ]
        self.current_data = self.data
        self.set_projection_mode_tracks()

    def set_projection_mode_ref(self) -> None:
        """
        Set projection mode for reference cells.
        """

        self.groupby_cols = [
            "position",
            "reference_population",
            "neighbor_population",
            "REFERENCE_ID",
            "FRAME",
        ]
        self.current_data = self.data
        self.set_projection_mode_tracks()

    def set_projection_mode_tracks(self) -> None:
        """
        Set projection mode for tracks.
        """

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

    def enable_projection_options(self) -> None:
        """
        Enable/disable projection options based on the selected mode.
        """

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

    def set_1D_plot_params(self) -> None:
        """
        Open the 1D plot parameter configuration window.
        """
        from celldetective.gui.base.plot_selector import (
            PlotSelectorWidget,
            StatsSelectorWidget,
        )
        from superqt import QColormapComboBox, QSearchableComboBox
        import matplotlib

        self.plot1Dparams = CelldetectiveWidget()
        self.plot1Dparams.setWindowTitle("Set 1D plot parameters")

        layout = QVBoxLayout()
        self.plot1Dparams.setLayout(layout)

        # self.plot1Dparams.resize(400, 600)  # Let it size itself

        layout.addWidget(QLabel("Representations: "))

        # New visual selector
        self.plot_selector = PlotSelectorWidget()
        layout.addWidget(self.plot_selector)

        self.sep_line = QHSeperationLine()

        layout.addWidget(self.sep_line)
        layout.addWidget(QLabel("Statistical Tests:"))
        self.stats_selector = StatsSelectorWidget()
        layout.addWidget(self.stats_selector)

        self.x_cb = QSearchableComboBox()
        self.x_cb.addItems(["--"] + list(self.data.columns))

        self.y_cb = QSearchableComboBox()
        self.y_cb.addItems(["--"] + list(self.data.columns))

        self.hue_cb = QSearchableComboBox()
        self.hue_cb.addItems(["--"] + list(self.data.columns))
        idx = self.hue_cb.findText("--")
        self.hue_cb.setCurrentIndex(idx)

        # Set selected columns
        try:
            x = self.table_view.selectedIndexes()
            col_idx = np.array([item.column() for item in x])
            column_names = self.data.columns
            unique_cols = np.unique(col_idx)

            if len(unique_cols) == 1:
                y_col = column_names[unique_cols[0]]
                idx = self.y_cb.findText(y_col)
                self.y_cb.setCurrentIndex(idx)

            if len(unique_cols) >= 2:
                x_col = column_names[unique_cols[0]]
                idx = self.x_cb.findText(x_col)
                self.x_cb.setCurrentIndex(idx)

                y_col = column_names[unique_cols[1]]
                idx = self.y_cb.findText(y_col)
                self.y_cb.setCurrentIndex(idx)

        except (IndexError, KeyError):
            # No column selected or invalid selection
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

        import warnings

        self.cmap_cb = QColormapComboBox()
        for name in matplotlib.colormaps.keys():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.cmap_cb.addColormap(name)
            except Exception:
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

    def plot1d(self) -> None:
        """
        Generate the 1D plot based on selected parameters.
        """
        import matplotlib.pyplot as plt
        import matplotlib
        import seaborn as sns
        from celldetective.gui.base.figure_canvas import FigureCanvas

        # Parallel coordinates requires its own dialog — x/y/hue don't apply.
        selected_plots = self.plot_selector.get_selection()
        if "parallel coordinates" in selected_plots:
            self.plot1Dparams.close()
            self.set_parallel_coords_params()
            return

        if "correlation matrix" in selected_plots:
            self.plot1Dparams.close()
            self.set_correlation_matrix_params()
            return

        self.fig, self.ax = plt.subplots(1, 1, figsize=(4, 3))
        self.plot1dWindow = FigureCanvas(self.fig, title="scatter", interactive=True)

        # Resolve colormap with a single case-insensitive lookup
        cmap_name = self.cmap_cb.currentText()
        canonical_cmap = next(
            (k for k in matplotlib.colormaps if k.lower() == cmap_name.lower()),
            "viridis",
        )
        cmap = matplotlib.colormaps[canonical_cmap]

        # Build hue palette
        try:
            self.hue_variable = self.hue_cb.currentText()
            unique_hues = self.data[self.hue_variable].dropna().unique()
            n_hues = len(unique_hues)
            colors = (
                sns.color_palette(canonical_cmap, n_colors=n_hues)
                if n_hues > 0
                else None
            )
        except Exception:
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

        self.x_option = self.x is not None

        legend = True

        selected_plots = self.plot_selector.get_selection()

        if "histogram" in selected_plots:

            def _get_binwidth(col_name):
                if col_name and isinstance(col_name, str):
                    match = re.search(r"_binned_([0-9.]+)_", col_name)
                    if match:
                        return float(match.group(1))
                return None

            bw_x = _get_binwidth(self.x)
            bw_y = _get_binwidth(self.y)
            kwargs_1d_x = {"binwidth": bw_x, "shrink": 0.9} if bw_x is not None else {}
            kwargs_1d_y = {"binwidth": bw_y, "shrink": 0.9} if bw_y is not None else {}
            kwargs_2d = (
                {"binwidth": (bw_x, bw_y)}
                if bw_x is not None and bw_y is not None
                else {}
            )

            if self.x is not None and self.y is not None:
                # Use continuous colormap explicitly for 2D histogram density mapping
                sns.histplot(
                    data=self.data,
                    x=self.x,
                    y=self.y,
                    hue=self.hue_variable,
                    cbar=True,
                    cmap=cmap,
                    ax=self.ax,
                    **kwargs_2d,
                )
                legend = False
            elif self.x is not None:
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
                    **kwargs_1d_x,
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
                    **kwargs_1d_y,
                )
                legend = False
            else:
                logger.warning("histogram: no variable selected")

        if "KDE plot" in selected_plots:
            if self.x is not None and self.y is not None:
                sns.kdeplot(
                    data=self.data,
                    x=self.x,
                    y=self.y,
                    hue=self.hue_variable,
                    palette=colors,
                    ax=self.ax,
                    cut=0,
                )
                legend = False
            elif self.x is not None:
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
                logger.warning("KDE plot: no variable selected")

        if "countplot" in selected_plots:
            x_val = self.x if self.x is not None else self.y
            if x_val is not None:
                sns.countplot(
                    data=self.data,
                    x=x_val,
                    hue=self.hue_variable,
                    legend=legend,
                    ax=self.ax,
                    palette=colors,
                )
                legend = False
            else:
                logger.warning("countplot: no variable selected")

        if "ECDF plot" in selected_plots:
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
                logger.warning("ECDF plot: no variable selected")

        if "line plot" in selected_plots:
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
                logger.warning("line plot: please provide an x variable")

        if "scatter plot" in selected_plots:
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
                logger.warning("scatter plot: please provide an x variable")

        if "swarm" in selected_plots:
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

        if "violin" in selected_plots:
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

        if "boxplot" in selected_plots:
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

        if "boxenplot" in selected_plots:
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

        if "strip" in selected_plots:
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

        selected_stats = self.stats_selector.get_selection()
        if "Compute effect size?\n(Cliff's Delta)" in selected_stats:
            self.compute_effect_size()
        if "Compute KS test\np-value?" in selected_stats:
            self.compute_pvalue()

    def extract_groupby_cols(self) -> Tuple[List[str], str]:
        """
        Extract the columns to group by for effect size or p-value computation.

        Returns
        -------
        list
            List of column names to group by.
        str
            The y-axis variable.
        """

        x = self.x
        y = self.y
        hue_variable = self.hue_variable

        selected_plots = self.plot_selector.get_selection()

        # Check if any plot that uses 'x' as the main feature (when vertical) is selected
        # In current logic: hist, ecdf, kde swap x/y logic based on what was passed to seaborn?
        # Actually, let's look at how x/y logic was determined in original code.
        # Original code checked if specific checkboxes were checked to decide if y = self.x

        if (
            "histogram" in selected_plots
            or "ECDF plot" in selected_plots
            or "KDE plot" in selected_plots
        ):
            # For these distributions, the feature being analysed is self.x when x is
            # set, otherwise it falls back to self.y (the y combo-box variable).
            y = self.x if self.x is not None else self.y
            x = None

        groupby_cols = []
        if x is not None:
            groupby_cols.append(x)
        if hue_variable is not None:
            groupby_cols.append(hue_variable)

        return groupby_cols, y

    def compute_effect_size(self) -> Optional[None]:
        """
        Compute Cliff's Delta effect size for the current selection.
        """
        selected_plots = self.plot_selector.get_selection()

        if "countplot" in selected_plots or "scatter plot" in selected_plots:
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

    def compute_pvalue(self) -> Optional[None]:
        """
        Compute the p-value using the KS test for the current selection.
        """
        selected_plots = self.plot_selector.get_selection()

        if "countplot" in selected_plots or "scatter plot" in selected_plots:
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

    def set_proj_mode(self) -> None:
        """
        Apply the selected projection mode to the data and show the result.
        """

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
            group_table = getattr(
                self.current_data.groupby(self.groupby_cols), self.projection_mode
            )(numeric_only=True)

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

    def _createMenuBar(self) -> None:
        """
        Create the menu bar for the main window.
        """
        menuBar = self.menuBar()
        self.fileMenu = QMenu("&File", self)
        menuBar.addMenu(self.fileMenu)
        self.editMenu = QMenu("&Edit", self)
        menuBar.addMenu(self.editMenu)
        self.mathMenu = QMenu("&Math", self)
        menuBar.addMenu(self.mathMenu)

    def save_as_csv(self) -> None:
        """
        Save the current table data as a CSV file.
        """
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

            # Get visual column order from header mapped to the original GUI model
            header = self.table_view.horizontalHeader()
            visual_cols = [
                self.model._data.columns[header.logicalIndex(i)]
                for i in range(header.count())
            ]

            # Export with preserved visual order and dropped unnamed cols
            valid_visual_cols = [c for c in visual_cols if c in self.data.columns]
            data_sorted = self.data[valid_visual_cols]

            data_sorted.to_csv(file_name, index=False)

    def plot_instantaneous(self) -> None:
        """
        Open the 1D/2D plotting menu (Plot Instantaneous).
        Always reachable regardless of selection.
        """
        self.set_1D_plot_params()

    def plot(self) -> None:
        """
        Plot the data based on the current mode.
        """

        if self.plot_mode == "static":

            x = self.table_view.selectedIndexes()
            col_idx = [item.column() for item in x]
            row_idx = [item.row() for item in x]
            column_names = self.data.columns
            unique_cols = np.unique(col_idx)

            if len(unique_cols) == 0:
                return

            if len(unique_cols) == 1:
                self.set_1D_plot_params()

            if len(unique_cols) == 2:
                import matplotlib.pyplot as plt
                from celldetective.gui.base.figure_canvas import FigureCanvas

                x1 = test_bool_array(self.data.iloc[row_idx, unique_cols[0]])
                x2 = test_bool_array(self.data.iloc[row_idx, unique_cols[1]])

                self.fig, self.ax = plt.subplots(1, 1, figsize=(4, 3))
                self.scatter_wdw = FigureCanvas(
                    self.fig, title="scatter", interactive=True
                )
                self.ax.scatter(x1, x2)
                self.ax.set_xlabel(column_names[unique_cols[0]])
                self.ax.set_ylabel(column_names[unique_cols[1]])
                plt.tight_layout()
                self.fig.set_facecolor("none")
                self.fig.canvas.setStyleSheet("background-color: transparent;")
                self.scatter_wdw.canvas.draw()
                self.scatter_wdw.show()

            if len(unique_cols) > 2:
                selected_col_names = [str(column_names[i]) for i in unique_cols]
                self.set_parallel_coords_params(selected_col_names)

        elif self.plot_mode == "plot_timeseries":
            import matplotlib.pyplot as plt
            from celldetective.gui.base.figure_canvas import FigureCanvas

            x = self.table_view.selectedIndexes()
            col_idx = np.array([item.column() for item in x])
            row_idx = np.array([item.row() for item in x])
            column_names = self.data.columns
            unique_cols = np.unique(col_idx)

            self.fig, self.ax = plt.subplots(1, 1, figsize=(4, 3))
            self.plot_wdw = FigureCanvas(self.fig, title="scatter", interactive=True)
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
            self.plot_wdw.show()

        elif self.plot_mode == "plot_track_signals":
            import matplotlib.pyplot as plt
            from celldetective.gui.base.figure_canvas import FigureCanvas

            x = self.table_view.selectedIndexes()
            col_idx = np.array([item.column() for item in x])
            row_idx = np.array([item.row() for item in x])
            column_names = self.data.columns
            unique_cols = np.unique(col_idx)

            if len(unique_cols) > 2:
                self.fig, self.ax = plt.subplots(1, 1, figsize=(7, 5.5))
                self.plot_wdw = FigureCanvas(
                    self.fig, title="track signals", interactive=True
                )
                for k in range(len(unique_cols)):

                    row_idx_i = row_idx[np.where(col_idx == unique_cols[k])[0]]
                    for w, well_group in self.data.groupby(["well_name"]):
                        for pos, pos_group in well_group.groupby(["pos_name"]):
                            for tid, group_track in pos_group.groupby(
                                self.groupby_cols[1:]
                            ):
                                self.ax.plot(
                                    group_track["FRAME"],
                                    group_track[column_names[unique_cols[k]]],
                                    label=column_names[unique_cols[k]],
                                )
                self.ax.legend()
                self.ax.set_xlabel("time [frame]")
                self.ax.set_ylabel(self.title)
                plt.tight_layout()
                self.fig.set_facecolor("none")
                self.fig.canvas.setStyleSheet("background-color: transparent;")
                self.plot_wdw.canvas.draw()
                self.plot_wdw.show()

            if len(unique_cols) == 2:

                self.fig, self.ax = plt.subplots(1, 1, figsize=(4, 3))
                self.scatter_wdw = FigureCanvas(
                    self.fig, title="scatter", interactive=True
                )
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
                self.plot_wdw = FigureCanvas(
                    self.fig, title="scatter", interactive=True
                )

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

    def set_parallel_coords_params(
        self, preselected_cols: Optional[List[str]] = None
    ) -> None:
        """
        Open the parallel coordinates plot parameter configuration window.

        Parameters
        ----------
        preselected_cols : list of str, optional
            Column names to pre-select as axes. Defaults to all numeric columns.
        """
        from PyQt5.QtWidgets import (
            QListWidget,
            QListWidgetItem,
            QDoubleSpinBox,
            QSpinBox,
        )

        self.parallelCoordsParams = CelldetectiveWidget()
        self.parallelCoordsParams.setWindowTitle("Parallel Coordinates Parameters")
        self.parallelCoordsParams.setMinimumWidth(420)

        layout = QVBoxLayout()
        self.parallelCoordsParams.setLayout(layout)

        # --- Axis columns (multi-select list) ---
        layout.addWidget(QLabel("Axes (select columns):"))
        self._pc_col_list = QListWidget()
        self._pc_col_list.setSelectionMode(QListWidget.MultiSelection)
        self._pc_col_list.setMaximumHeight(160)

        numeric_cols = list(
            self.data.select_dtypes(
                include=["int16", "int32", "int64", "float16", "float32", "float64"]
            ).columns
        )
        for col in numeric_cols:
            item = QListWidgetItem(col)
            self._pc_col_list.addItem(item)
            if preselected_cols and col in preselected_cols:
                item.setSelected(True)

        layout.addWidget(self._pc_col_list)

        from superqt import QSearchableComboBox

        # --- Color by ---
        hbox_hue = QHBoxLayout()
        hbox_hue.addWidget(QLabel("Color by: "), 33)
        self._pc_hue_cb = QSearchableComboBox()
        self._pc_hue_cb.addItems(["--"] + list(self.data.columns))
        hbox_hue.addWidget(self._pc_hue_cb, 66)
        layout.addLayout(hbox_hue)

        # --- Colormap (Plotly-native names only) ---
        hbox_cmap = QHBoxLayout()
        hbox_cmap.addWidget(QLabel("Colormap: "), 33)
        self._pc_cmap_cb = QComboBox()
        try:
            import plotly.colors as pc_colors

            plotly_scales = sorted(pc_colors.named_colorscales())
        except Exception:
            plotly_scales = ["Viridis", "Plasma", "Inferno", "Cividis", "Jet"]
        self._pc_cmap_cb.addItems(plotly_scales)
        # Default to viridis (plotly names are lowercase)
        idx = self._pc_cmap_cb.findText("viridis")
        if idx >= 0:
            self._pc_cmap_cb.setCurrentIndex(idx)
        hbox_cmap.addWidget(self._pc_cmap_cb, 66)
        layout.addLayout(hbox_cmap)

        # --- Alpha ---
        hbox_alpha = QHBoxLayout()
        hbox_alpha.addWidget(QLabel("Alpha: "), 33)
        self._pc_alpha_sb = QDoubleSpinBox()
        self._pc_alpha_sb.setRange(0.01, 1.0)
        self._pc_alpha_sb.setSingleStep(0.05)
        self._pc_alpha_sb.setValue(0.3)
        hbox_alpha.addWidget(self._pc_alpha_sb, 66)
        layout.addLayout(hbox_alpha)

        # --- Normalization ---
        hbox_norm = QHBoxLayout()
        hbox_norm.addWidget(QLabel("Normalize axes: "), 33)
        self._pc_norm_cb = QComboBox()
        self._pc_norm_cb.addItems(["min-max", "z-score", "none"])
        hbox_norm.addWidget(self._pc_norm_cb, 66)
        layout.addLayout(hbox_norm)

        # --- Plot button ---
        plot_btn = QPushButton("Plot")
        plot_btn.setStyleSheet(self.button_style_sheet)
        plot_btn.clicked.connect(self.plot_parallel_coords)
        layout.addWidget(plot_btn)

        self.parallelCoordsParams.show()
        center_window(self.parallelCoordsParams)

    def plot_parallel_coords(self) -> None:
        """
        Render an interactive parallel coordinates plot using Plotly.

        Opens the chart as a self-contained HTML file in the system's default
        browser, giving the user interactive axis reordering and per-axis
        brushing/filtering out of the box.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.error(
                "plotly is required for parallel coordinates plots. "
                "Install it with: pip install plotly"
            )
            QMessageBox.critical(
                self,
                "Missing dependency",
                "plotly is required for parallel coordinates plots.\n"
                "Install it with: pip install plotly",
            )
            return

        import tempfile
        import webbrowser

        # --- Collect parameters from dialog ---
        selected_items = self._pc_col_list.selectedItems()
        cols = [item.text() for item in selected_items]
        if len(cols) < 2:
            logger.warning(
                "parallel coordinates: please select at least 2 axis columns."
            )
            return

        hue_col = self._pc_hue_cb.currentText()
        if hue_col == "--":
            hue_col = None

        cmap_name = self._pc_cmap_cb.currentText()
        alpha = self._pc_alpha_sb.value()
        norm_mode = self._pc_norm_cb.currentText()  # "min-max", "z-score", "none"

        # --- Prepare data ---
        df = self.data[cols].copy().dropna()
        if hue_col is not None and hue_col in self.data.columns:
            hue_series = self.data.loc[df.index, hue_col]
        else:
            hue_series = None

        # --- Build per-axis dimension specs ---
        dimensions = []

        for col in cols:
            values = df[col]
            vmin, vmax = float(values.min()), float(values.max())
            span = vmax - vmin

            if norm_mode == "min-max":
                scaled = (values - vmin) / span if span != 0 else values * 0
                range_ = [0.0, 1.0]
            elif norm_mode == "z-score":
                std = values.std()
                mean = values.mean()
                scaled = (values - mean) / std if std != 0 else values * 0
                r = max(abs(float(scaled.min())), abs(float(scaled.max())))
                range_ = [-r, r]
            else:
                scaled = values
                range_ = [vmin, vmax]

            dimensions.append(
                dict(
                    label=col,
                    values=scaled.tolist(),
                    range=range_,
                    # Show the original tick values even when normalized
                    tickvals=(
                        [0.0, 0.25, 0.5, 0.75, 1.0] if norm_mode == "min-max" else None
                    ),
                    ticktext=(
                        [f"{vmin + i * span / 4:.3g}" for i in range(5)]
                        if norm_mode == "min-max" and span != 0
                        else None
                    ),
                )
            )

        # Silence Plotly alias warnings by mapping common aliases to fully namespaced names
        alias_map = {
            "rainbow": "gnuplot:rainbow",
            "prgn": "colorbrewer:PRGn",
            "rdbu": "colorbrewer:RdBu",
            "ylorbr": "colorbrewer:YlOrBr",
            "copper": "matlab:copper",
            "ocean": "gnuplot:ocean",
        }
        colorscale = alias_map.get(cmap_name, cmap_name)

        # --- Color line array ---
        if hue_series is not None:
            try:
                color_values = pd.to_numeric(
                    hue_series.reindex(df.index), errors="raise"
                ).tolist()
                colorbar_title = hue_col
            except (ValueError, TypeError):
                # Categorical: encode as integers
                cats = hue_series.reindex(df.index)
                unique_vals = list(cats.unique())
                color_values = [unique_vals.index(v) for v in cats]
                colorbar_title = hue_col
        else:
            color_values = list(range(len(df)))
            colorbar_title = None

        line_dict = dict(
            color=color_values,
            colorscale=colorscale,
            showscale=True,
            colorbar=(
                dict(title=colorbar_title, thickness=15, len=0.75)
                if colorbar_title
                else dict(showticklabels=False, thickness=0)
            ),
        )

        # --- Build figure ---
        fig = go.Figure(
            data=go.Parcoords(
                line=line_dict,
                dimensions=dimensions,
            )
        )
        fig.update_layout(
            # title="Parallel Coordinates",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=16),
            margin=dict(l=60, r=30, t=60, b=30),
        )

        # --- Open in window via QWebEngineView ---
        tmp = tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, prefix="parallel_coords_"
        )
        tmp_path = tmp.name
        tmp.close()
        fig.write_html(tmp_path, include_plotlyjs="cdn")

        try:
            from PyQt5.QtWebEngineWidgets import QWebEngineView
            from PyQt5.QtWidgets import QMainWindow
            from PyQt5.QtCore import QUrl

            self.pc_window = CelldetectiveMainWindow()
            self.pc_window.setWindowTitle("Parallel Coordinates")
            self.pc_window.resize(900, 600)

            browser = QWebEngineView()
            browser.load(QUrl.fromLocalFile(tmp_path))
            self.pc_window.setCentralWidget(browser)
            self.pc_window.show()
            center_window(self.pc_window)

            logger.info(
                f"Parallel coordinates plot opened in native window from {tmp_path}"
            )

        except ImportError:
            logger.warning(
                "PyQtWebEngine not found. Falling back to system web browser."
            )
            webbrowser.open(f"file:///{tmp_path}")
            logger.info(f"Parallel coordinates plot saved to {tmp_path}")

    def set_correlation_matrix_params(
        self, preselected_cols: Optional[List[str]] = None
    ) -> None:
        """
        Open the parameters configuration window for a Correlation Matrix plot.
        """
        from PyQt5.QtWidgets import (
            QVBoxLayout,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QListWidget,
            QListWidgetItem,
            QComboBox,
        )

        self.corrMatrixParams = CelldetectiveWidget()
        self.corrMatrixParams.setWindowTitle("Correlation Matrix Parameters")
        self.corrMatrixParams.setMinimumWidth(350)

        layout = QVBoxLayout()
        self.corrMatrixParams.setLayout(layout)

        # --- Axis columns (multi-select list) ---
        layout.addWidget(QLabel("Features (select columns):"))
        self._cm_col_list = QListWidget()
        self._cm_col_list.setSelectionMode(QListWidget.MultiSelection)
        self._cm_col_list.setMaximumHeight(200)

        numeric_cols = list(
            self.data.select_dtypes(
                include=["int16", "int32", "int64", "float16", "float32", "float64"]
            ).columns
        )
        for col in numeric_cols:
            item = QListWidgetItem(col)
            self._cm_col_list.addItem(item)
            if preselected_cols and col in preselected_cols:
                item.setSelected(True)

        layout.addWidget(self._cm_col_list)

        # --- Colormap (Plotly-native names only) ---
        hbox_cmap = QHBoxLayout()
        hbox_cmap.addWidget(QLabel("Colormap: "), 50)
        self._cm_cmap_cb = QComboBox()
        try:
            import plotly.colors as pc_colors

            plotly_scales = sorted(pc_colors.named_colorscales())
        except Exception:
            plotly_scales = ["rdbu", "viridis", "plasma", "inferno", "cividis", "jet"]
        self._cm_cmap_cb.addItems(plotly_scales)
        # Default to continuous diverging colormap for correlations (rdbu)
        idx = self._cm_cmap_cb.findText("rdbu")
        if idx >= 0:
            self._cm_cmap_cb.setCurrentIndex(idx)
        hbox_cmap.addWidget(self._cm_cmap_cb, 50)
        layout.addLayout(hbox_cmap)

        # --- Correlation Method ---
        hbox_method = QHBoxLayout()
        hbox_method.addWidget(QLabel("Method: "), 50)
        self._cm_method_cb = QComboBox()
        self._cm_method_cb.addItems(["pearson", "spearman", "kendall"])
        hbox_method.addWidget(self._cm_method_cb, 50)
        layout.addLayout(hbox_method)

        # --- Plot button ---
        plot_btn = QPushButton("Plot")
        plot_btn.setStyleSheet(self.button_style_sheet)
        plot_btn.clicked.connect(self.plot_correlation_matrix)
        layout.addWidget(plot_btn)

        self.corrMatrixParams.show()
        center_window(self.corrMatrixParams)

    def plot_correlation_matrix(self) -> None:
        """
        Render an interactive correlation matrix plot using Plotly heatmap.
        """
        try:
            import plotly.express as px
        except ImportError:
            logger.error("plotly is required. Install it with: pip install plotly")
            QMessageBox.critical(
                self,
                "Missing dependency",
                "plotly is required. Install it with: pip install plotly",
            )
            return

        import tempfile
        import webbrowser

        # --- Collect parameters from dialog ---
        selected_items = self._cm_col_list.selectedItems()
        cols = [item.text() for item in selected_items]
        if len(cols) < 2:
            logger.warning("correlation matrix: please select at least 2 features.")

            QMessageBox.warning(
                self,
                "Invalid Selection",
                "Please select at least 2 features to compute correlation.",
            )
            return

        cmap_name = self._cm_cmap_cb.currentText()
        method = self._cm_method_cb.currentText()

        # --- Calculate Correlation ---
        df = self.data[cols].copy().dropna()
        corr_matrix = df.corr(method=method)

        # --- Build figure ---
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale=cmap_name,
                color_continuous_midpoint=(
                    0
                    if "rdbu" in cmap_name.lower() or "prgn" in cmap_name.lower()
                    else None
                ),
                title=f"Correlation Matrix ({method.capitalize()})",
                labels=dict(color="Correlation"),
            )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=16),
            margin=dict(l=60, r=40, t=60, b=80),
        )
        # fig.update_xaxes(tickangle=-45)

        # --- Open in window via QWebEngineView ---
        tmp = tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, prefix="corr_matrix_"
        )
        tmp_path = tmp.name
        tmp.close()
        fig.write_html(tmp_path, include_plotlyjs="cdn")

        try:
            from PyQt5.QtWebEngineWidgets import QWebEngineView
            from PyQt5.QtWidgets import QMainWindow
            from PyQt5.QtCore import QUrl

            self.cm_window = QMainWindow()
            self.cm_window.setWindowTitle("Correlation Matrix")
            self.cm_window.resize(700, 700)

            browser = QWebEngineView()
            browser.load(QUrl.fromLocalFile(tmp_path))
            self.cm_window.setCentralWidget(browser)
            self.cm_window.show()
            center_window(self.cm_window)

            logger.info(
                f"Correlation matrix plot opened in native window from {tmp_path}"
            )

        except ImportError:
            logger.warning(
                "PyQtWebEngine not found. Falling back to system web browser."
            )
            webbrowser.open(f"file:///{tmp_path}")
            logger.info(f"Correlation matrix plot saved to {tmp_path}")
