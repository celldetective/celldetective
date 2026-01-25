from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QMessageBox,
    QSlider,
    QComboBox,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIntValidator, QKeySequence
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.cm import tab10
from superqt import QLabeledDoubleSlider
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6

from celldetective.gui.base_annotator import BaseAnnotator
from celldetective.gui.viewers.contour_viewer import CellEdgeVisualizer
from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.gui.gui_utils import color_from_state, color_from_class
from celldetective.utils.image_loaders import locate_labels, load_frames
from celldetective.utils.masks import contour_of_instance_segmentation
from celldetective.gui.base.figure_canvas import FigureCanvas
from celldetective.gui.base.utils import center_window
from celldetective import get_logger

logger = get_logger(__name__)


class AnnotatorStackVisualizer(CellEdgeVisualizer):
    def __init__(self, *args, **kwargs):
        self.scat_markers = None
        super().__init__(*args, **kwargs)
        self.compact_layout()

    def generate_figure_canvas(self):
        super().generate_figure_canvas()
        self.generate_custom_overlays()
        # Force layout update
        self.compact_layout()

    def generate_custom_overlays(self):
        """Initialize scatter artists."""
        self.scat_markers = self.ax.scatter([], [], color="tab:red", picker=True)
        # CellEdgeVisualizer handles self.im_mask

    def update_overlays(self, positions, colors):
        # Update Scatter
        self.scat_markers.set_offsets(positions)
        if len(positions) > 0 and len(colors) > 0:
            # colors should be an array of colors
            self.scat_markers.set_edgecolors(colors)
            self.scat_markers.set_facecolors("none")
            self.scat_markers.set_picker(10)
            self.scat_markers.set_linewidths(2)
            self.scat_markers.set_sizes([200] * len(positions))

        self.canvas.canvas.draw_idle()

    def generate_edge_slider(self):
        # Override to hide edge slider
        pass

    def generate_opacity_slider(self):
        # Compact opacity slider
        self.opacity_slider = QLabeledDoubleSlider()
        self.opacity_slider.setOrientation(Qt.Horizontal)
        self.opacity_slider.setRange(0, 1)
        self.opacity_slider.setValue(self.mask_alpha)
        self.opacity_slider.setDecimals(2)
        self.opacity_slider.valueChanged.connect(self.change_mask_opacity)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(QLabel("Opacity:"), 15)
        layout.addWidget(self.opacity_slider, 85)
        self.canvas.layout.addLayout(layout)

    def compact_layout(self):
        # Reduce margins/spacing for all slider layouts in canvas
        self.canvas.layout.setSpacing(0)
        self.canvas.layout.setContentsMargins(0, 5, 0, 5)
        for i in range(self.canvas.layout.count()):
            item = self.canvas.layout.itemAt(i)
            if item.layout():
                item.layout().setContentsMargins(0, 0, 0, 0)
                item.layout().setSpacing(0)
            elif item.widget():
                # If there are direct widgets, ensure they are compact too if possible
                pass


class MeasureAnnotator(BaseAnnotator):

    def __init__(self, *args, **kwargs):

        self.status_name = "group"
        super().__init__(read_config=False, *args, **kwargs)

        self.setWindowTitle("Static annotator")

        self.int_validator = QIntValidator()
        self.current_alpha = 0.5
        self.value_magnitude = 1

        epsilon = 0.01
        self.observed_min_intensity = 0
        self.observed_max_intensity = 0 + epsilon

        self.current_frame = 0
        self.show_fliers = False

        if self.proceed:

            from celldetective.utils.image_loaders import fix_missing_labels
            from celldetective.tracking import write_first_detection_class

            # Ensure labels match stack length
            if self.len_movie > 0:
                temp_labels = locate_labels(self.pos, population=self.mode)
                if temp_labels is None or len(temp_labels) < self.len_movie:
                    fix_missing_labels(
                        self.pos,
                        population=self.mode,
                        prefix=self.parent_window.movie_prefix,
                    )
                    self.labels = locate_labels(self.pos, population=self.mode)
                elif len(temp_labels) > self.len_movie:
                    self.labels = temp_labels[: self.len_movie]
                else:
                    self.labels = temp_labels
            else:
                self.labels = locate_labels(self.pos, population=self.mode)

            self.current_channel = 0
            self.frame_lbl = QLabel("position: ")

            # self.static_image() # Replaced by StackVisualizer initialization in populate_window

            self.populate_window()
            self.changed_class()

            self.previous_index = None

        else:
            self.close()

    def locate_tracks(self):
        """
        Locate the tracks.
        """

        if not os.path.exists(self.trajectories_path):

            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("The trajectories cannot be detected.")
            msgBox.setWindowTitle("Warning")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Yes:
                self.close()
        else:

            # Load and prep tracks
            self.df_tracks = pd.read_csv(self.trajectories_path)
            if "TRACK_ID" in self.df_tracks.columns:
                self.df_tracks = self.df_tracks.sort_values(by=["TRACK_ID", "FRAME"])
            else:
                self.df_tracks = self.df_tracks.sort_values(by=["ID", "FRAME"])

            cols = np.array(self.df_tracks.columns)
            self.class_cols = np.array(
                [
                    c.startswith("group")
                    or c.startswith("class")
                    or c.startswith("status")
                    for c in list(self.df_tracks.columns)
                ]
            )
            self.class_cols = list(cols[self.class_cols])

            to_remove = [
                "class_id",
                "group_color",
                "class_color",
                "group_id",
                "status_color",
                "status_id",
            ]
            for col in to_remove:
                try:
                    self.class_cols.remove(col)
                except:
                    pass

            # Generate missing status columns from class columns
            for c in self.class_cols:
                if c.startswith("class_"):
                    status_col = c.replace("class_", "status_")
                    if status_col not in self.df_tracks.columns:
                        if (
                            status_col == "status_firstdetection"
                            or c == "class_firstdetection"
                        ):
                            try:
                                from celldetective.tracking import (
                                    write_first_detection_class,
                                )

                                self.df_tracks = write_first_detection_class(
                                    self.df_tracks
                                )
                            except Exception as e:
                                logger.error(
                                    f"Could not generate status_firstdetection: {e}"
                                )
                                self.df_tracks[status_col] = self.df_tracks[c]
                        else:
                            self.df_tracks[status_col] = self.df_tracks[c]

            # Re-evaluate class_cols after generation
            cols = np.array(self.df_tracks.columns)
            self.class_cols = np.array(
                [
                    c.startswith("group") or c.startswith("status")
                    for c in list(self.df_tracks.columns)
                ]
            )
            self.class_cols = list(cols[self.class_cols])
            for col in to_remove:
                try:
                    self.class_cols.remove(col)
                except:
                    pass

            if len(self.class_cols) > 0:
                if self.status_name not in self.class_cols:
                    self.status_name = self.class_cols[0]
            else:
                self.status_name = "group"

            if self.status_name not in self.df_tracks.columns:
                # only create the status column if it does not exist to not erase static classification results
                self.make_status_column()
            else:
                # all good, do nothing
                pass

            all_states = self.df_tracks.loc[:, self.status_name].tolist()
            all_states = np.array(all_states)
            self.state_color_map = color_from_state(all_states, recently_modified=False)
            self.df_tracks["group_color"] = self.df_tracks[self.status_name].apply(
                self.assign_color_state
            )

            self.df_tracks = self.df_tracks.dropna(subset=["POSITION_X", "POSITION_Y"])
            self.df_tracks["x_anim"] = self.df_tracks["POSITION_X"]
            self.df_tracks["y_anim"] = self.df_tracks["POSITION_Y"]
            self.df_tracks["x_anim"] = self.df_tracks["x_anim"].astype(int)
            self.df_tracks["y_anim"] = self.df_tracks["y_anim"].astype(int)

            self.extract_scatter_from_trajectories()
            if "TRACK_ID" in self.df_tracks.columns:
                self.track_of_interest = self.df_tracks.dropna(subset="TRACK_ID")[
                    "TRACK_ID"
                ].min()
            else:
                self.track_of_interest = self.df_tracks.dropna(subset="ID")["ID"].min()

            self.loc_t = []
            self.loc_idx = []
            for t in range(len(self.tracks)):
                indices = np.where(self.tracks[t] == self.track_of_interest)[0]
                if len(indices) > 0:
                    self.loc_t.append(t)
                    self.loc_idx.append(indices[0])

            from sklearn.preprocessing import MinMaxScaler

            self.MinMaxScaler = MinMaxScaler()
            self.columns_to_rescale = list(self.df_tracks.columns)

            cols_to_remove = [
                "group",
                "group_color",
                "status",
                "status_color",
                "class_color",
                "TRACK_ID",
                "FRAME",
                "x_anim",
                "y_anim",
                "t",
                "dummy",
                "group_color",
                "state",
                "generation",
                "root",
                "parent",
                "class_id",
                "class",
                "t0",
                "POSITION_X",
                "POSITION_Y",
                "position",
                "well",
                "well_index",
                "well_name",
                "pos_name",
                "index",
                "concentration",
                "cell_type",
                "antibody",
                "pharmaceutical_agent",
                "ID",
            ] + self.class_cols

            from celldetective.utils.experiment import (
                get_experiment_metadata,
                get_experiment_labels,
            )

            meta = get_experiment_metadata(self.exp_dir)
            if meta is not None:
                keys = list(meta.keys())
                cols_to_remove.extend(keys)

            labels = get_experiment_labels(self.exp_dir)
            if labels is not None:
                keys = list(labels.keys())
                cols_to_remove.extend(labels)

            for tr in cols_to_remove:
                try:
                    self.columns_to_rescale.remove(tr)
                except:
                    pass

            x = self.df_tracks[self.columns_to_rescale].values
            self.MinMaxScaler.fit(x)

    def populate_options_layout(self):
        # clear options hbox
        for i in reversed(range(self.options_hbox.count())):
            self.options_hbox.itemAt(i).widget().setParent(None)

        time_option_hbox = QHBoxLayout()
        time_option_hbox.setContentsMargins(100, 0, 100, 0)
        time_option_hbox.setSpacing(0)

        self.time_of_interest_label = QLabel("phenotype: ")
        time_option_hbox.addWidget(self.time_of_interest_label, 30)

        self.time_of_interest_le = QLineEdit()
        self.time_of_interest_le.setValidator(self.int_validator)
        time_option_hbox.addWidget(self.time_of_interest_le)

        self.suppr_btn = QPushButton("")
        self.suppr_btn.setStyleSheet(self.button_select_all)
        self.suppr_btn.setIcon(icon(MDI6.delete, color="black"))
        self.suppr_btn.setToolTip("Delete cell")
        self.suppr_btn.setIconSize(QSize(20, 20))
        self.suppr_btn.clicked.connect(self.del_cell)
        time_option_hbox.addWidget(self.suppr_btn)

        self.options_hbox.addLayout(time_option_hbox)

    def update_widgets(self):

        self.class_label.setText("characteristic \n group: ")
        self.update_class_cb()
        self.add_class_btn.setToolTip("Add a new characteristic group")
        self.del_class_btn.setToolTip("Delete a characteristic group")

        self.export_btn.disconnect()
        self.export_btn.clicked.connect(self.export_measurements)

    def update_class_cb(self):

        self.class_choice_cb.disconnect()
        self.class_choice_cb.clear()
        cols = np.array(self.df_tracks.columns)
        self.class_cols = np.array(
            [
                c.startswith("group")
                or c.startswith("status")
                or (
                    c.startswith("class")
                    and not c.endswith("_id")
                    and not c.endswith("_color")
                )
                for c in list(self.df_tracks.columns)
            ]
        )
        self.class_cols = list(cols[self.class_cols])

        to_remove = [
            "group_id",
            "group_color",
            "class_id",
            "class_color",
            "status_color",
            "status_id",
        ]
        for col in to_remove:
            while col in self.class_cols:
                self.class_cols.remove(col)

        # Filter to keep only group_* and status_* as requested by user, but allow 'group' if it exists
        final_cols = []
        for c in self.class_cols:
            if c == "group" or c.startswith("group_") or c.startswith("status_"):
                final_cols.append(c)

        self.class_cols = final_cols

        self.class_choice_cb.addItems(self.class_cols)
        if self.status_name in self.class_cols:
            self.class_choice_cb.setCurrentText(self.status_name)
        self.class_choice_cb.currentIndexChanged.connect(self.changed_class)

    def populate_window(self):

        super().populate_window()
        # Left panel updates
        self.populate_options_layout()
        self.update_widgets()

        self.annotation_btns_to_hide = [
            self.time_of_interest_label,
            self.time_of_interest_le,
            self.suppr_btn,
        ]
        self.hide_annotation_buttons()

        # Right panel - Initialize StackVisualizer
        self.viewer = AnnotatorStackVisualizer(
            stack_path=self.stack_path,
            labels=self.labels,
            frame_slider=True,
            channel_cb=True,
            channel_names=self.channel_names,
            n_channels=self.nbr_channels,
            target_channel=0,
            window_title="Stack Viewer",
        )

        # Connect viewer signals
        self.viewer.frame_slider.valueChanged.connect(self.sync_frame)
        self.viewer.channel_cb.currentIndexChanged.connect(self.plot_signals)

        # Connect mpl event
        self.cid_pick = self.viewer.fig.canvas.mpl_connect(
            "pick_event", self.on_scatter_pick
        )

        self.right_panel.addWidget(self.viewer.canvas)

        # Force start at frame 0
        self.viewer.frame_slider.setValue(0)

        self.plot_signals()
        self.compact_layout_main()

    def compact_layout_main(self):
        # Attempt to compact the viewer layout one more time from the main window side
        if hasattr(self, "viewer"):
            self.viewer.compact_layout()

    def sync_frame(self, value):
        """Callback when StackVisualizer frame changes"""

        self.current_frame = value
        self.update_frame_logic()

    def plot_signals(self):
        """Delegate signal plotting but check for viewer availability"""
        if not hasattr(self, "viewer"):
            return

        # Call the original plot_signals logic or adapt it.
        # Since plot_signals uses self.cell_ax (left panel), it should be fine.
        # However, we need to ensure it uses the correct current_frame.

        current_frame = self.current_frame

        yvalues = []
        all_yvalues = []
        current_yvalues = []
        labels = []
        range_values = []

        for i in range(len(self.signal_choice_cb)):
            signal_choice = self.signal_choice_cb[i].currentText()

            if signal_choice != "--":
                if "TRACK_ID" in self.df_tracks.columns:
                    ydata = self.df_tracks.loc[
                        (self.df_tracks["TRACK_ID"] == self.track_of_interest)
                        & (self.df_tracks["FRAME"] == current_frame),
                        signal_choice,
                    ].to_numpy()
                else:
                    ydata = self.df_tracks.loc[
                        (self.df_tracks["ID"] == self.track_of_interest), signal_choice
                    ].to_numpy()
                all_ydata = self.df_tracks.loc[:, signal_choice].to_numpy()
                ydataNaN = ydata
                ydata = ydata[ydata == ydata]  # remove nan

                current_ydata = self.df_tracks.loc[
                    (self.df_tracks["FRAME"] == current_frame), signal_choice
                ].to_numpy()
                current_ydata = current_ydata[current_ydata == current_ydata]
                all_ydata = all_ydata[all_ydata == all_ydata]
                yvalues.extend(ydataNaN)
                current_yvalues.append(current_ydata)
                all_yvalues.append(all_ydata)
                range_values.extend(all_ydata)
                labels.append(signal_choice)

        self.cell_ax.clear()
        if self.log_scale:
            self.cell_ax.set_yscale("log")
        else:
            self.cell_ax.set_yscale("linear")

        if len(yvalues) > 0:
            try:
                self.cell_ax.boxplot(all_yvalues, showfliers=self.show_fliers)
            except Exception as e:
                logger.error(f"{e=}")

            x_pos = np.arange(len(all_yvalues)) + 1
            for index, feature in enumerate(current_yvalues):
                x_values_strip = (index + 1) + np.random.normal(
                    0, 0.04, size=len(feature)
                )
                self.cell_ax.plot(
                    x_values_strip,
                    feature,
                    marker="o",
                    linestyle="None",
                    color=tab10.colors[0],
                    alpha=0.1,
                )
            self.cell_ax.plot(
                x_pos,
                yvalues,
                marker="H",
                linestyle="None",
                color=tab10.colors[3],
                alpha=1,
            )
            range_values = np.array(range_values)
            range_values = range_values[range_values == range_values]

            # Filter out non-positive values if log scale is active to prevent warnings
            if self.log_scale:
                range_values = range_values[range_values > 0]

            if len(range_values) > 0:
                self.value_magnitude = np.nanmin(range_values) - 0.03 * (
                    np.nanmax(range_values) - np.nanmin(range_values)
                )
            else:
                self.value_magnitude = 1

            self.non_log_ymin = np.nanmin(range_values) - 0.03 * (
                np.nanmax(range_values) - np.nanmin(range_values)
            )
            self.non_log_ymax = np.nanmax(range_values) + 0.03 * (
                np.nanmax(range_values) - np.nanmin(range_values)
            )
            if self.cell_ax.get_yscale() == "linear":
                self.cell_ax.set_ylim(self.non_log_ymin, self.non_log_ymax)
            else:
                self.cell_ax.set_ylim(self.value_magnitude, self.non_log_ymax)
        else:
            self.cell_ax.text(
                0.5,
                0.5,
                "No data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=self.cell_ax.transAxes,
            )

        self.cell_fcanvas.canvas.draw()

    def plot_red_points(self, ax):
        yvalues = []
        current_frame = self.current_frame
        for i in range(len(self.signal_choice_cb)):
            signal_choice = self.signal_choice_cb[i].currentText()
            if signal_choice != "--":
                if "TRACK_ID" in self.df_tracks.columns:
                    ydata = self.df_tracks.loc[
                        (self.df_tracks["TRACK_ID"] == self.track_of_interest)
                        & (self.df_tracks["FRAME"] == current_frame),
                        signal_choice,
                    ].to_numpy()
                else:
                    ydata = self.df_tracks.loc[
                        (self.df_tracks["ID"] == self.track_of_interest)
                        & (self.df_tracks["FRAME"] == current_frame),
                        signal_choice,
                    ].to_numpy()
                ydata = ydata[ydata == ydata]  # remove nan
                yvalues.extend(ydata)
        x_pos = np.arange(len(yvalues)) + 1
        ax.plot(
            x_pos, yvalues, marker="H", linestyle="None", color=tab10.colors[3], alpha=1
        )  # Plot red points representing cells
        self.cell_fcanvas.canvas.draw()

    def select_single_cell(self, index, timepoint):

        self.correct_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)
        self.del_shortcut.setEnabled(True)

        self.track_of_interest = self.tracks[timepoint][index]
        logger.info(f"You selected cell #{self.track_of_interest}...")
        self.give_cell_information()

        if len(self.cell_ax.lines) > 0:
            self.cell_ax.lines[
                -1
            ].remove()  # Remove the last line (red points) from the plot
            self.plot_red_points(self.cell_ax)
        else:
            self.plot_signals()

        self.loc_t = []
        self.loc_idx = []
        for t in range(len(self.tracks)):
            indices = np.where(self.tracks[t] == self.track_of_interest)[0]
            if len(indices) > 0:
                self.loc_t.append(t)
                self.loc_idx.append(indices[0])

        self.previous_color = []
        for t, idx in zip(self.loc_t, self.loc_idx):
            self.previous_color.append(self.colors[t][idx].copy())
            self.colors[t][idx] = "lime"

        self.draw_frame(self.current_frame)

    def cancel_selection(self):
        super().cancel_selection()
        self.event = None
        self.draw_frame(self.current_frame)

    def export_measurements(self):
        logger.info("User interactions: Exporting measurements...")
        # Implementation same as before
        auto_dataset_name = (
            self.pos.split(os.sep)[-4]
            + "_"
            + self.pos.split(os.sep)[-2]
            + f"_{str(self.current_frame).zfill(3)}"
            + f"_{self.status_name}.npy"
        )

        if self.normalized_signals:
            self.normalize_features_btn.click()

        subdf = self.df_tracks.loc[self.df_tracks["FRAME"] == self.current_frame, :]
        subdf["class"] = subdf[self.status_name]
        dico = subdf.to_dict("records")

        pathsave = QFileDialog.getSaveFileName(
            self, "Select file name", self.exp_dir + auto_dataset_name, ".npy"
        )[0]
        if pathsave != "":
            if not pathsave.endswith(".npy"):
                pathsave += ".npy"
            try:
                np.save(pathsave, dico)
                logger.info(f"File successfully written in {pathsave}.")
            except Exception as e:
                logger.error(f"Error {e}...")

    def write_new_event_class(self):

        if self.class_name_le.text() == "":
            self.target_class = "group"
        else:
            self.target_class = "group_" + self.class_name_le.text()

        logger.info(
            f"User interactions: Creating new characteristic group '{self.target_class}'"
        )

        if self.target_class in list(self.df_tracks.columns):
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText(
                "This characteristic group name already exists. If you proceed,\nall annotated data will be rewritten. Do you wish to continue?"
            )
            msgBox.setWindowTitle("Warning")
            msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.No:
                return None
            else:
                pass

        self.df_tracks.loc[:, self.target_class] = 0

        self.update_class_cb()

        idx = self.class_choice_cb.findText(self.target_class)
        self.status_name = self.target_class
        self.class_choice_cb.setCurrentIndex(idx)
        self.newClassWidget.close()

    def hide_annotation_buttons(self):

        for a in self.annotation_btns_to_hide:
            a.hide()
        self.time_of_interest_label.setEnabled(False)
        self.time_of_interest_le.setText("")
        self.time_of_interest_le.setEnabled(False)

    def show_annotation_buttons(self):

        for a in self.annotation_btns_to_hide:
            a.show()

        self.time_of_interest_label.setEnabled(True)
        self.time_of_interest_le.setEnabled(True)
        self.correct_btn.setText("submit")

        self.correct_btn.disconnect()
        self.correct_btn.clicked.connect(self.apply_modification)

    def give_cell_information(self):

        try:
            cell_selected = f"cell: {self.track_of_interest}\n"
            if self.status_name in self.df_tracks.columns:
                if "TRACK_ID" in self.df_tracks.columns:
                    val = self.df_tracks.loc[
                        (self.df_tracks["FRAME"] == self.current_frame)
                        & (self.df_tracks["TRACK_ID"] == self.track_of_interest),
                        self.status_name,
                    ].to_numpy()
                    if len(val) > 0:
                        cell_status = f"phenotype: {val[0]}\n"
                    else:
                        cell_status = "phenotype: N/A\n"
                else:
                    val = self.df_tracks.loc[
                        self.df_tracks["ID"] == self.track_of_interest, self.status_name
                    ].to_numpy()
                    if len(val) > 0:
                        cell_status = f"phenotype: {val[0]}\n"
                    else:
                        cell_status = "phenotype: N/A\n"
            else:
                cell_status = f"phenotype: N/A (col '{self.status_name}' missing)\n"
            self.cell_info.setText(cell_selected + cell_status)
        except Exception as e:
            logger.error(f"Error in give_cell_information: {e}")

    def create_new_event_class(self):

        # display qwidget to name the event
        self.newClassWidget = CelldetectiveWidget()
        self.newClassWidget.setWindowTitle("Create new characteristic group")

        layout = QVBoxLayout()
        self.newClassWidget.setLayout(layout)
        name_hbox = QHBoxLayout()
        name_hbox.addWidget(QLabel("group name: "), 25)
        self.class_name_le = QLineEdit("group")
        name_hbox.addWidget(self.class_name_le, 75)
        layout.addLayout(name_hbox)

        btn_hbox = QHBoxLayout()
        submit_btn = QPushButton("submit")
        cancel_btn = QPushButton("cancel")
        btn_hbox.addWidget(cancel_btn, 50)
        btn_hbox.addWidget(submit_btn, 50)
        layout.addLayout(btn_hbox)

        submit_btn.clicked.connect(self.write_new_event_class)
        cancel_btn.clicked.connect(self.close_without_new_class)

        self.newClassWidget.show()
        center_window(self.newClassWidget)

    def apply_modification(self):
        if self.time_of_interest_le.text() != "":
            status = int(self.time_of_interest_le.text())
        else:
            status = 0

        logger.info(
            f"User interactions: Reclassifying cell #{self.track_of_interest} at frame {self.current_frame} to status {status}"
        )
        if "TRACK_ID" in self.df_tracks.columns:
            self.df_tracks.loc[
                (self.df_tracks["TRACK_ID"] == self.track_of_interest)
                & (self.df_tracks["FRAME"] == self.current_frame),
                self.status_name,
            ] = status

            indices = self.df_tracks.index[
                (self.df_tracks["TRACK_ID"] == self.track_of_interest)
                & (self.df_tracks["FRAME"] == self.current_frame)
            ]
        else:
            self.df_tracks.loc[
                (self.df_tracks["ID"] == self.track_of_interest)
                & (self.df_tracks["FRAME"] == self.current_frame),
                self.status_name,
            ] = status

            indices = self.df_tracks.index[
                (self.df_tracks["ID"] == self.track_of_interest)
                & (self.df_tracks["FRAME"] == self.current_frame)
            ]

        self.df_tracks.loc[indices, self.status_name] = status
        all_states = self.df_tracks.loc[:, self.status_name].tolist()
        all_states = np.array(all_states)
        self.state_color_map = color_from_state(all_states, recently_modified=False)

        self.df_tracks["group_color"] = self.df_tracks[self.status_name].apply(
            self.assign_color_state
        )
        self.extract_scatter_from_trajectories()
        self.give_cell_information()

        self.correct_btn.disconnect()
        self.correct_btn.clicked.connect(self.show_annotation_buttons)

        self.hide_annotation_buttons()
        self.correct_btn.setEnabled(False)
        self.correct_btn.setText("correct")
        self.cancel_btn.setEnabled(False)
        self.del_shortcut.setEnabled(False)

        if len(self.selection) > 0:
            self.selection.pop(0)

        self.draw_frame(self.current_frame)

    def assign_color_state(self, state):

        try:
            if np.isnan(state):
                state = "nan"
        except TypeError:
            pass
        return self.state_color_map[state]

    def on_scatter_pick(self, event):
        """Handle pick event on scatter plot."""
        self.event = event
        ind = event.ind

        if len(ind) > 1:
            # disambiguate based on distance to mouse click
            datax, datay = [self.positions[self.current_frame][i, 0] for i in ind], [
                self.positions[self.current_frame][i, 1] for i in ind
            ]
            msx, msy = event.mouseevent.xdata, event.mouseevent.ydata
            dist = np.sqrt((np.array(datax) - msx) ** 2 + (np.array(datay) - msy) ** 2)
            ind = [ind[np.argmin(dist)]]

        if len(ind) > 0:
            # We have a single point
            idx = ind[0]

            # Enforce single selection / Toggle
            if len(self.selection) > 0:
                # Check if we clicked the same cell
                prev_idx, prev_frame = self.selection[0]
                if (
                    prev_idx == idx
                ):  # and prev_frame == self.current_frame (implicit since we pick on current frame)

                    self.cancel_selection()
                    return

                self.cancel_selection()

            self.selection = [[idx, self.current_frame]]
            self.select_single_cell(idx, self.current_frame)

    def draw_frame(self, framedata):
        """
        Update plot elements at each timestep of the loop.
        Using StackVisualizer overlay update
        """
        self.framedata = framedata

        # Prepare overlays
        if self.framedata < len(self.positions):
            pos = self.positions[self.framedata]
            cols = self.colors[self.framedata][:, 0]
        else:
            pos = []
            cols = []

        # No need to manage contour cache or labels here
        # CellEdgeVisualizer handles it.

        # Update Viewer Scatter Only
        # Note: Mask is updated automatically by CellEdgeVisualizer's change_frame
        self.viewer.update_overlays(
            positions=pos,
            colors=cols,
        )

    def make_status_column(self):
        if self.status_name == "state_firstdetection":
            pass
        else:
            self.df_tracks.loc[:, self.status_name] = 0
            all_states = self.df_tracks.loc[:, self.status_name].tolist()
            all_states = np.array(all_states)
            self.state_color_map = color_from_state(all_states, recently_modified=False)
            self.df_tracks["group_color"] = self.df_tracks[self.status_name].apply(
                self.assign_color_state
            )

    def extract_scatter_from_trajectories(self):

        self.positions = []
        self.colors = []
        self.tracks = []

        for t in np.arange(self.len_movie):
            self.positions.append(
                self.df_tracks.loc[
                    self.df_tracks["FRAME"] == t, ["POSITION_X", "POSITION_Y"]
                ].to_numpy()
            )
            self.colors.append(
                self.df_tracks.loc[self.df_tracks["FRAME"] == t, ["group_color"]]
                .to_numpy()
                .copy()
            )
            if "TRACK_ID" in self.df_tracks.columns:
                self.tracks.append(
                    self.df_tracks.loc[
                        self.df_tracks["FRAME"] == t, "TRACK_ID"
                    ].to_numpy()
                )
            else:
                self.tracks.append(
                    self.df_tracks.loc[self.df_tracks["FRAME"] == t, "ID"].to_numpy()
                )

    def compute_status_and_colors(self, index=0):
        self.changed_class()

    def changed_class(self):
        self.status_name = self.class_choice_cb.currentText()
        if self.status_name != "":
            # self.compute_status_and_colors()
            self.modify()
            self.draw_frame(self.current_frame)

    def update_frame_logic(self):
        """
        Logic to execute when frame changes.
        """
        # Auto-switch track of interest if ID mode
        if "TRACK_ID" in list(self.df_tracks.columns):
            pass
        elif "ID" in list(self.df_tracks.columns):
            # print("ID in cols... change class of interest... ")
            candidates = self.df_tracks[self.df_tracks["FRAME"] == self.current_frame][
                "ID"
            ]
            if not candidates.empty:
                self.track_of_interest = candidates.min()
            self.modify()

        self.draw_frame(self.current_frame)
        self.give_cell_information()
        self.plot_signals()

    def changed_channel(self):
        """Handled by StackViewer mostly, but we might need to refresh plotting if things depend on channel"""
        pass  # StackViewer handles image reload

    def save_trajectories(self):
        logger.info(f"Saving trajectories...")
        if self.normalized_signals:
            self.normalize_features_btn.click()
        if self.selection:
            self.cancel_selection()

        # Avoid crash if status doesn't exist or is special
        # self.df_tracks = self.df_tracks.drop(
        #     self.df_tracks[self.df_tracks[self.status_name] == 99].index
        # )

        try:
            self.df_tracks.drop(columns="", inplace=True)
        except:
            pass
        try:
            self.df_tracks.drop(columns="group_color", inplace=True)
        except:
            pass
        try:
            self.df_tracks.drop(columns="x_anim", inplace=True)
        except:
            pass
        try:
            self.df_tracks.drop(columns="y_anim", inplace=True)
        except:
            pass

        self.df_tracks.to_csv(self.trajectories_path, index=False)
        logger.info("Table successfully exported...")

        self.locate_tracks()
        self.changed_class()

    def modify(self):
        if self.status_name not in self.df_tracks.columns:
            logger.warning(
                f"Column '{self.status_name}' not found in df_tracks. Skipping modify."
            )
            return

        all_states = self.df_tracks.loc[:, self.status_name].tolist()
        all_states = np.array(all_states)
        self.state_color_map = color_from_state(all_states, recently_modified=False)

        self.df_tracks["group_color"] = self.df_tracks[self.status_name].apply(
            self.assign_color_state
        )

        self.extract_scatter_from_trajectories()
        self.give_cell_information()

        self.correct_btn.disconnect()
        self.correct_btn.clicked.connect(self.show_annotation_buttons)

    def del_cell(self):
        logger.info(
            f"User interactions: Deleting cell #{self.track_of_interest} (setting status to 99)"
        )
        self.time_of_interest_le.setEnabled(False)
        self.time_of_interest_le.setText("99")
        self.apply_modification()
