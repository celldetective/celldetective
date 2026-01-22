import json
import os
from glob import glob

import numpy as np
from PyQt5.QtCore import Qt, QSize, QThread
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QAction,
    QMenu,
    QMessageBox,
    QLabel,
    QFileDialog,
    QHBoxLayout,
    QGridLayout,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QComboBox,
    QPushButton,
    QApplication,
    QRadioButton,
    QButtonGroup,
)
from fonticon_mdi6 import MDI6

from superqt import QLabeledSlider, QLabeledDoubleRangeSlider
from superqt.fonticon import icon

from celldetective.gui.gui_utils import PreprocessingLayout
from celldetective.gui.base.components import (
    generic_message,
    CelldetectiveMainWindow,
    CelldetectiveWidget,
)
from celldetective.gui.gui_utils import color_from_class, help_generic
from celldetective.gui.base.figure_canvas import FigureCanvas
from celldetective.gui.viewers.threshold_viewer import ThresholdedStackVisualizer
from celldetective.utils.image_loaders import load_frames

from celldetective import (
    get_software_location,
)
from celldetective.utils.data_cleaning import rename_intensity_column
from celldetective.utils.experiment import extract_experiment_channels
import logging

logger = logging.getLogger(__name__)


class BackgroundLoader(QThread):
    def run(self):
        logger.info("Loading background packages...")
        try:
            from celldetective.segmentation import (
                identify_markers_from_binary,
                apply_watershed,
            )
            from scipy.ndimage._measurements import label
            import pandas as pd
            from celldetective.regionprops._regionprops import regionprops_table
        except Exception:
            logger.error("Background packages not loaded...")
        logger.info("Background packages loaded...")


class ThresholdConfigWizard(CelldetectiveMainWindow):
    """
    UI to create a threshold pipeline for segmentation.

    """

    def __init__(self, parent_window=None):

        super().__init__()
        self.parent_window = parent_window
        self.screen_height = (
            self.parent_window.parent_window.parent_window.parent_window.screen_height
        )
        self.screen_width = (
            self.parent_window.parent_window.parent_window.parent_window.screen_width
        )
        self.setMinimumWidth(int(0.8 * self.screen_width))
        self.setMinimumHeight(int(0.8 * self.screen_height))
        self.setWindowTitle("Threshold configuration wizard")

        self._createActions()
        self._create_menu_bar()

        self.mode = self.parent_window.mode
        self.pos = self.parent_window.parent_window.parent_window.pos
        self.exp_dir = self.parent_window.parent_window.exp_dir
        self.soft_path = get_software_location()
        self.footprint = 30
        self.min_dist = 30
        self.onlyFloat = QDoubleValidator()
        self.onlyInt = QIntValidator()
        self.cell_properties = [
            "centroid",
            "area",
            "perimeter",
            "eccentricity",
            "intensity_mean",
            "solidity",
        ]
        self.edge = None
        self.filters = []
        self.fill_holes = True

        self.locate_stack()
        self.generate_viewer()
        self.img = self.viewer.init_frame

        self.config_out_name = f"threshold_{self.mode}.json"
        if self.img is not None:
            self.threshold_slider = QLabeledDoubleRangeSlider()
            self.initialize_histogram()
            # self.show_image()
            self.initalize_props_scatter()
            self.prep_cell_properties()
            self.populate_widget()
            self.setAttribute(Qt.WA_DeleteOnClose)

        self.bg_loader = BackgroundLoader()
        self.bg_loader.start()

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        # Creating menus using a QMenu object
        file_menu = QMenu("&File", self)
        file_menu.addAction(self.openAction)
        menu_bar.addMenu(file_menu)

    # Creating menus using a title
    # editMenu = menuBar.addMenu("&Edit")
    # helpMenu = menuBar.addMenu("&Help")

    def _createActions(self):
        # Creating action using the first constructor
        # self.newAction = QAction(self)
        # self.newAction.setText("&New")
        # Creating actions using the second constructor
        self.openAction = QAction(icon(MDI6.folder), "&Open...", self)
        self.openAction.triggered.connect(self.load_previous_config)

    def populate_widget(self):
        """
        Create the multibox design.

        """
        self.button_widget = CelldetectiveWidget()
        main_layout = QHBoxLayout()
        self.button_widget.setLayout(main_layout)

        main_layout.setContentsMargins(30, 30, 30, 30)

        self.scroll_area = QScrollArea()
        self.scroll_container = CelldetectiveWidget()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_container)

        self.left_panel = QVBoxLayout(self.scroll_container)
        self.left_panel.setContentsMargins(30, 30, 30, 30)
        self.left_panel.setSpacing(10)
        self.populate_left_panel()

        # Right panel
        self.right_panel = QVBoxLayout()
        self.populate_right_panel()

        main_layout.addWidget(self.scroll_area, 35)
        main_layout.addLayout(self.right_panel, 65)
        self.button_widget.adjustSize()

        self.setCentralWidget(self.button_widget)
        self.show()

        QApplication.processEvents()

    def populate_left_panel(self):

        self.preprocessing = PreprocessingLayout(self)
        self.left_panel.addLayout(self.preprocessing)

        ###################
        # THRESHOLD SECTION
        ###################

        grid_threshold = QGridLayout()
        grid_threshold.setContentsMargins(20, 20, 20, 20)
        idx = 0

        threshold_title_grid = QHBoxLayout()
        section_threshold = QLabel("Threshold")
        section_threshold.setStyleSheet("font-weight: bold;")
        threshold_title_grid.addWidget(section_threshold, 80, alignment=Qt.AlignCenter)

        self.fill_holes_btn = QPushButton("")
        self.fill_holes_btn.setIcon(icon(MDI6.format_color_fill, color="white"))
        self.fill_holes_btn.setIconSize(QSize(20, 20))
        self.fill_holes_btn.setStyleSheet(self.button_select_all)
        self.fill_holes_btn.setToolTip("Fill holes in binary mask")
        self.fill_holes_btn.setCheckable(True)
        self.fill_holes_btn.setChecked(True)
        self.fill_holes_btn.clicked.connect(self.toggle_fill_holes)
        threshold_title_grid.addWidget(self.fill_holes_btn, 5)

        self.ylog_check = QPushButton("")
        self.ylog_check.setIcon(icon(MDI6.math_log, color="black"))
        self.ylog_check.setStyleSheet(self.button_select_all)
        self.ylog_check.setCheckable(True)
        self.ylog_check.clicked.connect(self.switch_to_log)
        threshold_title_grid.addWidget(self.ylog_check, 5)

        self.equalize_option_btn = QPushButton("")
        self.equalize_option_btn.setIcon(icon(MDI6.equalizer, color="black"))
        self.equalize_option_btn.setIconSize(QSize(20, 20))
        self.equalize_option_btn.setStyleSheet(self.button_select_all)
        self.equalize_option_btn.setToolTip("Enable histogram matching")
        self.equalize_option_btn.clicked.connect(self.activate_histogram_equalizer)
        self.equalize_option = False
        threshold_title_grid.addWidget(self.equalize_option_btn, 5)

        grid_threshold.addLayout(threshold_title_grid, idx, 0, 1, 2)

        idx += 1

        # Slider to set vmin & vmax
        self.threshold_slider.setSingleStep(0.00001)
        self.threshold_slider.setTickInterval(0.00001)
        self.threshold_slider.setOrientation(Qt.Horizontal)
        self.threshold_slider.setDecimals(5)
        self.threshold_slider.setRange(
            np.amin(self.img[self.img == self.img]),
            np.amax(self.img[self.img == self.img]),
        )
        self.threshold_slider.setValue(
            [np.percentile(self.img.ravel(), 90), np.amax(self.img)]
        )
        self.threshold_slider.valueChanged.connect(self.threshold_changed)

        # self.initialize_histogram()
        grid_threshold.addWidget(self.canvas_hist, idx, 0, 1, 3)

        idx += 1

        grid_threshold.addWidget(self.threshold_slider, idx, 1, 1, 1)
        self.canvas_hist.setMinimumHeight(self.screen_height // 6)
        self.left_panel.addLayout(grid_threshold)

        self.generate_marker_contents()
        self.generate_props_contents()

        #################
        # FINAL SAVE BTN#
        #################

        self.save_btn = QPushButton("Save")
        self.save_btn.setStyleSheet(self.button_style_sheet)
        self.save_btn.clicked.connect(self.write_instructions)
        self.left_panel.addWidget(self.save_btn)

        self.properties_box_widgets = [
            self.propscanvas,
            *self.features_cb,
            self.property_query_le,
            self.submit_query_btn,
            self.save_btn,
        ]
        for p in self.properties_box_widgets:
            p.setEnabled(False)

        # Force initial update after all UI elements are created
        self.threshold_changed(self.threshold_slider.value())

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
            message_box = QMessageBox()
            message_box.setIcon(QMessageBox.Information)
            message_box.setTextFormat(Qt.RichText)
            message_box.setText(
                f"The suggested technique is to {suggestion}.\nSee a tutorial <a "
                f"href='https://celldetective.readthedocs.io/en/latest/segment.html'>here</a>."
            )
            message_box.setWindowTitle("Info")
            message_box.setStandardButtons(QMessageBox.Ok)
            return_value = message_box.exec()
            if return_value == QMessageBox.Ok:
                return None

    def generate_marker_contents(self):

        marker_box = QVBoxLayout()
        marker_box.setContentsMargins(30, 30, 30, 30)

        marker_lbl = QLabel("Objects")
        marker_lbl.setStyleSheet("font-weight: bold;")
        marker_box.addWidget(marker_lbl, alignment=Qt.AlignCenter)

        object_option_hbox = QHBoxLayout()
        self.marker_option = QRadioButton("markers")
        self.all_objects_option = QRadioButton("all non-contiguous objects")
        self.marker_option_group = QButtonGroup()
        self.marker_option_group.addButton(self.marker_option)
        self.marker_option_group.addButton(self.all_objects_option)
        object_option_hbox.addWidget(self.marker_option, 50, alignment=Qt.AlignCenter)
        object_option_hbox.addWidget(
            self.all_objects_option, 50, alignment=Qt.AlignCenter
        )
        marker_box.addLayout(object_option_hbox)

        hbox_footprint = QHBoxLayout()
        hbox_footprint.addWidget(QLabel("Footprint: "), 20)
        self.footprint_slider = QLabeledSlider()
        self.footprint_slider.setSingleStep(1)
        self.footprint_slider.setOrientation(Qt.Horizontal)
        self.footprint_slider.setRange(1, self.img.shape[0] // 4)
        self.footprint_slider.setValue(self.footprint)
        self.footprint_slider.valueChanged.connect(self.set_footprint)
        hbox_footprint.addWidget(self.footprint_slider, 30)
        hbox_footprint.addWidget(QLabel(""), 50)
        marker_box.addLayout(hbox_footprint)

        hbox_distance = QHBoxLayout()
        hbox_distance.addWidget(QLabel("Min distance: "), 20)
        self.min_dist_slider = QLabeledSlider()
        self.min_dist_slider.setSingleStep(1)
        self.min_dist_slider.setOrientation(Qt.Horizontal)
        self.min_dist_slider.setRange(0, self.img.shape[0] // 4)
        self.min_dist_slider.setValue(self.min_dist)
        self.min_dist_slider.valueChanged.connect(self.set_min_dist)
        hbox_distance.addWidget(self.min_dist_slider, 30)
        hbox_distance.addWidget(QLabel(""), 50)
        marker_box.addLayout(hbox_distance)

        hbox_marker_btns = QHBoxLayout()

        self.markers_btn = QPushButton("Run")
        self.markers_btn.clicked.connect(self.detect_markers)
        self.markers_btn.setStyleSheet(self.button_style_sheet)
        hbox_marker_btns.addWidget(self.markers_btn)

        self.watershed_btn = QPushButton("Watershed")
        self.watershed_btn.setIcon(icon(MDI6.waves_arrow_up, color="white"))
        self.watershed_btn.setIconSize(QSize(20, 20))
        self.watershed_btn.clicked.connect(self.apply_watershed_to_selection)
        self.watershed_btn.setStyleSheet(self.button_style_sheet)
        self.watershed_btn.setEnabled(False)
        hbox_marker_btns.addWidget(self.watershed_btn)
        marker_box.addLayout(hbox_marker_btns)

        self.marker_option.clicked.connect(self.enable_marker_options)
        self.all_objects_option.clicked.connect(self.enable_marker_options)
        self.marker_option.click()

        self.left_panel.addLayout(marker_box)

    def enable_marker_options(self):
        if self.marker_option.isChecked():
            self.footprint_slider.setEnabled(True)
            self.min_dist_slider.setEnabled(True)
            self.markers_btn.setEnabled(True)
        else:
            self.footprint_slider.setEnabled(False)
            self.min_dist_slider.setEnabled(False)
            self.markers_btn.setEnabled(False)
            self.watershed_btn.setEnabled(True)

    def generate_props_contents(self):

        properties_box = QVBoxLayout()
        properties_box.setContentsMargins(30, 30, 30, 30)

        properties_lbl = QLabel("Filter on properties")
        properties_lbl.setStyleSheet("font-weight: bold;")
        properties_box.addWidget(properties_lbl, alignment=Qt.AlignCenter)

        properties_box.addWidget(self.propscanvas)

        self.features_cb = [QComboBox() for _ in range(2)]
        for i in range(2):
            hbox_feat = QHBoxLayout()
            hbox_feat.addWidget(QLabel(f"feature {i}: "), 20)
            hbox_feat.addWidget(self.features_cb[i], 80)
            properties_box.addLayout(hbox_feat)

        hbox_classify = QHBoxLayout()
        hbox_classify.addWidget(QLabel("remove: "), 10)
        self.property_query_le = QLineEdit()
        self.property_query_le.setPlaceholderText(
            "eliminate points using a query such as: area > 100 or eccentricity > 0.95"
        )
        hbox_classify.addWidget(self.property_query_le, 70)
        self.submit_query_btn = QPushButton("Submit...")
        self.submit_query_btn.setStyleSheet(self.button_style_sheet)
        self.submit_query_btn.clicked.connect(self.apply_property_query)
        hbox_classify.addWidget(self.submit_query_btn, 20)
        properties_box.addLayout(hbox_classify)

        self.left_panel.addLayout(properties_box)

    def generate_viewer(self):
        self.viewer = ThresholdedStackVisualizer(
            preprocessing=self.filters,
            show_opacity_slider=False,
            show_threshold_slider=False,
            stack_path=self.stack_path,
            frame_slider=True,
            contrast_slider=True,
            channel_cb=True,
            channel_names=self.channel_names,
            n_channels=self.nbr_channels,
            target_channel=0,
            PxToUm=None,
            initial_threshold=None,
            fill_holes=self.fill_holes,
        )

    def populate_right_panel(self):
        self.right_panel.addWidget(self.viewer.canvas)

    def locate_stack(self):
        """
        Locate the target movie.

        """

        if isinstance(self.pos, str):
            movies = glob(
                self.pos
                + f"movie/{self.parent_window.parent_window.parent_window.movie_prefix}*.tif"
            )

        else:
            generic_message(
                "Please select a unique position before launching the wizard..."
            )
            self.img = None
            self.close()
            return None

        if len(movies) == 0:
            generic_message(
                "No movies are detected in the experiment folder. Cannot load an image to test Haralick."
            )
            self.img = None
            self.close()
        else:
            self.stack_path = movies[0]
            self.channel_names, _ = extract_experiment_channels(self.exp_dir)
            self.channel_names = np.array(self.channel_names)
            self.nbr_channels = len(self.channel_names)

    def initalize_props_scatter(self):
        """
        Define properties scatter.
        """

        from matplotlib.figure import Figure

        self.fig_props = Figure(tight_layout=True)
        self.ax_props = self.fig_props.add_subplot(111)
        self.propscanvas = FigureCanvas(self.fig_props, interactive=True)
        self.fig_props.set_facecolor("none")
        self.fig_props.canvas.setStyleSheet("background-color: transparent;")
        self.scat_props = self.ax_props.scatter([], [], color="k", alpha=0.75)
        self.propscanvas.canvas.draw_idle()
        self.propscanvas.canvas.setMinimumHeight(self.screen_height // 5)

    def initialize_histogram(self):

        self.img = self.viewer.init_frame

        from matplotlib.figure import Figure

        self.fig_hist = Figure(tight_layout=True)
        self.ax_hist = self.fig_hist.add_subplot(111)
        self.canvas_hist = FigureCanvas(self.fig_hist, interactive=False)
        self.fig_hist.set_facecolor("none")
        self.fig_hist.canvas.setStyleSheet("background-color: transparent;")

        # self.ax_hist.clear()
        # self.ax_hist.cla()
        self.ax_hist.patch.set_facecolor("none")
        self.hist_y, x, _ = self.ax_hist.hist(
            self.img.ravel(), density=True, bins=300, color="k"
        )
        # self.ax_hist.set_xlim(np.amin(self.img),np.amax(self.img))
        self.ax_hist.set_xlabel("intensity [a.u.]")
        self.ax_hist.spines["top"].set_visible(False)
        self.ax_hist.spines["right"].set_visible(False)
        # self.ax_hist.set_yticks([])
        self.ax_hist.set_xlim(
            np.amin(self.img[self.img == self.img]),
            np.amax(self.img[self.img == self.img]),
        )
        self.ax_hist.set_ylim(0, self.hist_y.max())

        self.threshold_slider.setRange(
            np.amin(self.img[self.img == self.img]),
            np.amax(self.img[self.img == self.img]),
        )
        self.threshold_slider.setValue(
            [np.nanpercentile(self.img.ravel(), 90), np.amax(self.img)]
        )
        self.add_hist_threshold()

        self.canvas_hist.canvas.draw_idle()
        self.canvas_hist.canvas.setMinimumHeight(self.screen_height // 8)

    def update_histogram(self):
        """
        Redraw the histogram after an update on the image.
        Move the threshold slider accordingly.

        """

        self.ax_hist.clear()
        self.ax_hist.patch.set_facecolor("none")
        self.hist_y, x, _ = self.ax_hist.hist(
            self.img.ravel(), density=True, bins=300, color="k"
        )
        self.ax_hist.set_xlabel("intensity [a.u.]")
        self.ax_hist.spines["top"].set_visible(False)
        self.ax_hist.spines["right"].set_visible(False)
        # self.ax_hist.set_yticks([])
        self.ax_hist.set_xlim(
            np.amin(self.img[self.img == self.img]),
            np.amax(self.img[self.img == self.img]),
        )
        self.ax_hist.set_ylim(0, self.hist_y.max())
        self.add_hist_threshold()
        self.canvas_hist.canvas.draw()

        self.threshold_slider.setRange(
            np.amin(self.img[self.img == self.img]),
            np.amax(self.img[self.img == self.img]),
        )
        self.threshold_slider.setValue(
            [np.nanpercentile(self.img.ravel(), 90), np.amax(self.img)]
        )
        self.threshold_changed(self.threshold_slider.value())

    def add_hist_threshold(self):

        ymin, ymax = self.ax_hist.get_ylim()
        (self.min_intensity_line,) = self.ax_hist.plot(
            [self.threshold_slider.value()[0], self.threshold_slider.value()[0]],
            [0, ymax],
            c="tab:purple",
        )
        (self.max_intensity_line,) = self.ax_hist.plot(
            [self.threshold_slider.value()[1], self.threshold_slider.value()[1]],
            [0, ymax],
            c="tab:purple",
        )

    # self.canvas_hist.canvas.draw_idle()

    def reload_frame(self):
        """
        Load the frame from the current channel and time choice. Show imshow, update histogram.
        """

        self.clear_post_threshold_options()
        self.viewer.set_preprocessing(self.filters)
        self.img = self.viewer.processed_image
        self.update_histogram()

    def preprocess_image(self):
        """
        Reload the frame, apply the filters, update imshow and histogram.

        """

        self.filters = self.preprocessing.list.items
        self.reload_frame()
        self.update_histogram()

    def threshold_changed(self, value):
        """
        Move the threshold values on histogram, when slider is moved.
        """
        self.clear_post_threshold_options()
        self.viewer.change_threshold(value)

        ymin, ymax = self.ax_hist.get_ylim()
        self.min_intensity_line.set_data([value[0], value[0]], [0, ymax])
        self.max_intensity_line.set_data([value[1], value[1]], [0, ymax])
        self.canvas_hist.canvas.draw_idle()

    def switch_to_log(self):
        """
        Switch threshold histogram to log scale. Auto adjust.
        """

        if self.ax_hist.get_yscale() == "linear":
            self.ax_hist.set_yscale("log")
            self.ylog_check.setIcon(icon(MDI6.math_log, color="white"))
        else:
            self.ax_hist.set_yscale("linear")
            self.ylog_check.setIcon(icon(MDI6.math_log, color="black"))

        # self.ax_hist.autoscale()
        ymin = 0 if self.ax_hist.get_yscale() == "linear" else 1e-1
        self.ax_hist.set_ylim(ymin, self.hist_y.max())
        self.canvas_hist.canvas.draw_idle()

    def toggle_fill_holes(self):
        self.fill_holes = self.fill_holes_btn.isChecked()
        self.viewer.fill_holes = self.fill_holes
        self.viewer.change_threshold(self.threshold_slider.value())
        color = "white" if self.fill_holes else "black"
        self.fill_holes_btn.setIcon(icon(MDI6.format_color_fill, color=color))

    def set_footprint(self):
        self.footprint = self.footprint_slider.value()

    # print(f"Setting footprint to {self.footprint}")

    def set_min_dist(self):
        self.min_dist = self.min_dist_slider.value()

    # print(f"Setting min distance to {self.min_dist}")

    def detect_markers(self):

        self.clear_post_threshold_options()

        if self.viewer.mask.ndim == 3:
            self.viewer.mask = np.squeeze(self.viewer.mask)

        from celldetective.segmentation import identify_markers_from_binary

        self.coords, self.edt_map = identify_markers_from_binary(
            self.viewer.mask,
            self.min_dist,
            footprint_size=self.footprint,
            footprint=None,
            return_edt=True,
        )
        if len(self.coords) > 0:
            self.viewer.scat_markers.set_offsets(self.coords[:, [1, 0]])
            self.viewer.scat_markers.set_visible(True)
            self.viewer.canvas.draw()
            self.scat_props.set_visible(True)
            self.watershed_btn.setEnabled(True)
        else:
            self.watershed_btn.setEnabled(False)

    def apply_watershed_to_selection(self):

        import scipy.ndimage as ndi
        from celldetective.segmentation import apply_watershed

        if self.marker_option.isChecked():
            self.labels = apply_watershed(
                self.viewer.mask, self.coords, self.edt_map, fill_holes=self.fill_holes
            )
        else:
            from scipy.ndimage._measurements import label

            self.labels, _ = label(self.viewer.mask.astype(int))

        self.viewer.channel_trigger = True
        self.viewer.change_frame_from_channel_switch(self.viewer.frame_slider.value())
        self.viewer.im_mask.set_cmap("tab20c")
        self.viewer.im_mask.set_data(
            np.ma.masked_where(self.labels == 0.0, self.labels)
        )
        self.viewer.im_mask.autoscale()
        self.viewer.canvas.canvas.draw_idle()

        self.compute_features()
        for p in self.properties_box_widgets:
            p.setEnabled(True)

        for i in range(2):
            self.features_cb[i].currentTextChanged.connect(self.update_props_scatter)

    def compute_features(self):

        import pandas as pd
        from skimage.measure import regionprops_table

        # Run regionprops to have properties for filtering
        intensity_image_idx = [self.nbr_channels * self.viewer.frame_slider.value()]
        for i in range(self.nbr_channels - 1):
            intensity_image_idx += [intensity_image_idx[-1] + 1]

        # Load channels at time t
        multichannel = load_frames(
            intensity_image_idx, self.stack_path, normalize_input=False
        )
        self.props = pd.DataFrame(
            regionprops_table(
                self.labels,
                intensity_image=multichannel,
                properties=self.cell_properties,
            )
        )
        self.props = rename_intensity_column(self.props, self.channel_names)
        self.props["radial_distance"] = np.sqrt(
            (self.props["centroid-1"] - self.img.shape[0] / 2) ** 2
            + (self.props["centroid-0"] - self.img.shape[1] / 2) ** 2
        )

        self.props["class"] = 1

        for i in range(2):
            self.features_cb[i].clear()
            self.features_cb[i].addItems(list(self.props.columns))
            self.features_cb[i].setCurrentIndex(i)

        self.update_props_scatter()

    def update_props_scatter(self):

        feat1 = self.features_cb[1].currentText()
        feat0 = self.features_cb[0].currentText()

        if feat1 == "" or feat0 == "":
            return

        self.scat_props.set_offsets(self.props[[feat1, feat0]].to_numpy())
        self.scat_props.set_facecolor(
            [color_from_class(c) for c in self.props["class"].to_numpy()]
        )
        self.ax_props.set_xlabel(feat1)
        self.ax_props.set_ylabel(feat0)

        self.viewer.scat_markers.set_offsets(
            self.props[["centroid-1", "centroid-0"]].to_numpy()
        )
        self.viewer.scat_markers.set_color(["k"] * len(self.props))
        self.viewer.scat_markers.set_facecolor(
            [color_from_class(c) for c in self.props["class"].to_numpy()]
        )
        self.viewer.scat_markers.set_visible(True)

        if not self.props.empty:
            min_f1, max_f1 = self.props[feat1].min(), self.props[feat1].max()
            min_f0, max_f0 = self.props[feat0].min(), self.props[feat0].max()

            if np.isfinite([min_f1, max_f1, min_f0, max_f0]).all():
                self.ax_props.set_xlim(
                    0.75 * min_f1,
                    1.05 * max_f1,
                )
                self.ax_props.set_ylim(
                    0.75 * min_f0,
                    1.05 * max_f0,
                )
        self.propscanvas.canvas.draw_idle()
        self.viewer.canvas.canvas.draw()
        logger.info(f"Update markers for {len(self.props)} objects.")

    def prep_cell_properties(self):

        self.cell_properties_options = list(np.copy(self.cell_properties))
        self.cell_properties_options.remove("centroid")
        for k in range(self.nbr_channels):
            self.cell_properties_options.append(f"intensity_mean-{k}")
        self.cell_properties_options.remove("intensity_mean")

    def apply_property_query(self):
        query = self.property_query_le.text()
        self.props["class"] = 1

        if query == "":
            logger.warning("empty query")
        else:
            try:
                self.selection = self.props.query(query).index
                logger.info(f"{self.selection}")
                self.props.loc[self.selection, "class"] = 0
            except Exception as e:
                generic_message(
                    f"The query could not be understood. No filtering was applied. {e}"
                )
                return None

        self.update_props_scatter()

    def clear_post_threshold_options(self):

        self.watershed_btn.setEnabled(False)
        for p in self.properties_box_widgets:
            p.setEnabled(False)

        for i in range(2):
            try:
                self.features_cb[i].disconnect()
            except Exception as _:
                pass
            self.features_cb[i].clear()

        self.property_query_le.setText("")

        self.viewer.change_threshold(self.threshold_slider.value())
        self.viewer.scat_markers.set_color("tab:red")
        self.viewer.scat_markers.set_visible(False)

    def write_instructions(self):

        instructions = {
            "target_channel": self.viewer.channel_cb.currentText(),
            # for now index but would be more universal to use name
            "thresholds": self.threshold_slider.value(),
            "filters": self.preprocessing.list.items,
            "marker_min_distance": self.min_dist,
            "marker_footprint_size": self.footprint,
            "feature_queries": [self.property_query_le.text()],
            "equalize_reference": [
                self.equalize_option,
                self.viewer.frame_slider.value(),
            ],
            "do_watershed": self.marker_option.isChecked(),
            "fill_holes": self.fill_holes,
        }

        logger.info(f"The following instructions will be written: {instructions}")
        self.instruction_file = QFileDialog.getSaveFileName(
            self,
            "Save File",
            self.exp_dir + f"configs/threshold_config_{self.mode}.json",
            ".json",
        )[0]
        if self.instruction_file != "":
            json_object = json.dumps(instructions, indent=4)
            with open(self.instruction_file, "w") as outfile:
                outfile.write(json_object)
            logger.info(
                f"Configuration successfully written in {self.instruction_file}"
            )

            self.parent_window.filename = self.instruction_file
            self.parent_window.file_label.setText(self.instruction_file[:16] + "...")
            self.parent_window.file_label.setToolTip(self.instruction_file)

            self.close()
        else:
            logger.error("The instruction file could not be written...")

    def activate_histogram_equalizer(self):

        if not self.equalize_option:
            self.equalize_option = True
            self.equalize_option_btn.setIcon(icon(MDI6.equalizer, color="#1f77b4"))
            self.equalize_option_btn.setIconSize(QSize(20, 20))
        else:
            self.equalize_option = False
            self.equalize_option_btn.setIcon(icon(MDI6.equalizer, color="black"))
            self.equalize_option_btn.setIconSize(QSize(20, 20))

    def load_previous_config(self):
        self.previous_instruction_file = QFileDialog.getOpenFileName(
            self,
            "Load config",
            self.exp_dir + f"configs/threshold_config_{self.mode}.json",
            "JSON (*.json)",
        )[0]
        with open(self.previous_instruction_file, "r") as f:
            threshold_instructions = json.load(f)

        target_channel = threshold_instructions["target_channel"]
        index = self.viewer.channels_cb.findText(target_channel)
        self.viewer.channels_cb.setCurrentIndex(index)

        filters = threshold_instructions["filters"]
        items_to_add = [f[0] + "_filter" for f in filters]
        self.preprocessing.list.list_widget.clear()
        self.preprocessing.list.list_widget.addItems(items_to_add)
        self.preprocessing.list.items = filters
        self.preprocessing.apply_btn.click()

        thresholds = threshold_instructions["thresholds"]
        self.threshold_slider.setValue(thresholds)

        marker_footprint_size = threshold_instructions["marker_footprint_size"]
        self.footprint_slider.setValue(marker_footprint_size)

        marker_min_dist = threshold_instructions["marker_min_distance"]
        self.min_dist_slider.setValue(marker_min_dist)

        self.markers_btn.click()
        self.watershed_btn.click()

        feature_queries = threshold_instructions["feature_queries"]
        self.property_query_le.setText(feature_queries[0])
        self.submit_query_btn.click()

        if "do_watershed" in threshold_instructions:
            do_watershed = threshold_instructions["do_watershed"]
            if do_watershed:
                self.marker_option.click()
            else:
                self.all_objects_option.click()
