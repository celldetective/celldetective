"""
The annotator windows driven end to end on a small synthetic experiment.

The event, interaction (pair event) and phenotype annotators are opened on a
generated position (movie, labels, trajectories, pairs), and their workflow is
followed as a user would: select a cell or a pair, correct its annotation, save
and export. The tables written are then checked.
"""

import logging
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from PyQt5.QtCore import Qt
from tifffile import imwrite

N_FRAMES = 6
SIZE = 64
NEIGHBORHOOD = "neighborhood_self_circle_20_px"
# Tracks of the targets: (TRACK_ID, x, y, class_lysis, t_lysis).
TRACKS = [(0, 12, 12, 0, 2.0), (1, 48, 14, 1, -1.0), (2, 14, 50, 1, -1.0), (3, 50, 50, 2, -1.0)]


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable logging to avoid Windows OSError with pytest capture."""
    try:
        logging.disable(logging.CRITICAL)
        yield
    finally:
        logging.disable(logging.NOTSET)


@pytest.fixture
def position(tmp_path):
    """
    A position of a synthetic experiment: a two channel movie of six frames,
    the labels and trajectories of four target cells, and their pairs.
    """

    exp_dir = tmp_path / "Exp"
    pos = exp_dir / "W1" / "100"
    for folder in (
        pos / "movie",
        pos / "labels_targets",
        pos / "output" / "tables",
        exp_dir / "configs",
    ):
        folder.mkdir(parents=True)

    (exp_dir / "config.ini").write_text(
        "[Populations]\npopulations = targets\n\n"
        "[MovieSettings]\npxtoum = 1.0\nframetomin = 1.0\n"
        f"len_movie = {N_FRAMES}\nshape_x = {SIZE}\nshape_y = {SIZE}\nmovie_prefix = \n\n"
        "[Channels]\nch_a = 0\nch_b = 1\n\n"
        "[Labels]\ncell_types = 0\nantibodies = 0\nconcentrations = 0\npharmaceutical_agents = 0\n\n"
        "[Metadata]\nconcentration_units = pM\n"
    )

    rng = np.random.default_rng(0)
    movie = rng.integers(100, 1000, size=(N_FRAMES, 2, SIZE, SIZE)).astype(np.uint16)
    imwrite(pos / "movie" / "stack.tif", movie, imagej=True, metadata={"axes": "TCYX"})

    yy, xx = np.mgrid[:SIZE, :SIZE]
    rows = []
    for t in range(N_FRAMES):
        labels = np.zeros((SIZE, SIZE), dtype=np.uint16)
        for track, x, y, cclass, t_event in TRACKS:
            labels[(xx - x) ** 2 + (yy - y) ** 2 <= 16] = track + 1
            rows.append(
                {
                    "TRACK_ID": track,
                    "FRAME": t,
                    "POSITION_X": float(x),
                    "POSITION_Y": float(y),
                    "class_id": track + 1,
                    "area": 50.0 + track + t,
                    "intensity_mean": 100.0 * (track + 1) + t,
                    "class_lysis": cclass,
                    "t_lysis": t_event,
                    NEIGHBORHOOD: 1.0,
                }
            )
        imwrite(pos / "labels_targets" / f"{t:04d}.tif", labels)
    pd.DataFrame(rows).to_csv(
        pos / "output" / "tables" / "trajectories_targets.csv", index=False
    )

    pairs = []
    for reference, neighbor in ((0, 1), (1, 0), (0, 2), (2, 0)):
        for t in range(N_FRAMES):
            pairs.append(
                {
                    "REFERENCE_ID": reference,
                    "NEIGHBOR_ID": neighbor,
                    "reference_population": "targets",
                    "neighbor_population": "targets",
                    "FRAME": t,
                    "distance": 10.0 * (reference + neighbor) + t,
                    f"status_{NEIGHBORHOOD}": 1.0,
                    f"class_{NEIGHBORHOOD}": 1,
                    f"t0_{NEIGHBORHOOD}": -1.0,
                }
            )
    pd.DataFrame(pairs).to_csv(
        pos / "output" / "tables" / "trajectories_pairs.csv", index=False
    )

    return str(exp_dir) + os.sep, str(pos) + os.sep


def parent_chain(position, mode="targets"):
    """The windows above an annotator: the process block, the control panel, the app."""

    exp_dir, pos = position
    app = SimpleNamespace(screen_height=800, screen_width=1200)
    control_panel = SimpleNamespace(
        pos=pos,
        PxToUm=1.0,
        movie_prefix="",
        len_movie=N_FRAMES,
        populations=["targets"],
        parent_window=app,
    )
    return SimpleNamespace(
        mode=mode, exp_dir=exp_dir, parent_window=control_panel, movie_prefix=""
    )


def pick(artist, index, positions):
    """A pick event on a point of a scatter."""

    x, y = positions[index]
    return SimpleNamespace(
        ind=[index],
        artist=artist,
        mouseevent=SimpleNamespace(xdata=x, ydata=y),
    )


def read_table(position, name):
    _, pos = position
    return pd.read_csv(os.path.join(pos, "output", "tables", f"trajectories_{name}.csv"))


@pytest.fixture
def event_annotator(qtbot, position):
    from celldetective.gui.event_annotator import EventAnnotator

    annotator = EventAnnotator(parent_chain(position), lazy_load=True)
    qtbot.addWidget(annotator)
    annotator.prepare_stack()
    annotator.finalize_init()
    annotator.show()
    annotator.stop()
    yield annotator
    annotator.close()


@pytest.fixture
def pair_annotator(qtbot, position):
    from celldetective.gui.pair_event_annotator import PairEventAnnotator

    annotator = PairEventAnnotator(parent_chain(position, mode="pairs"))
    qtbot.addWidget(annotator)
    annotator.stop()
    yield annotator
    annotator.close()


@pytest.fixture
def measure_annotator(qtbot, position):
    from celldetective.gui.measure_annotator import MeasureAnnotator

    annotator = MeasureAnnotator(parent_chain(position))
    qtbot.addWidget(annotator)
    annotator.show()
    yield annotator
    annotator.close()


class TestEventAnnotatorWorkflow:

    def test_opens_on_the_position(self, event_annotator):
        """The movie is loaded at a quarter of its size, the event class chosen."""

        assert event_annotator.stack.shape == (N_FRAMES, SIZE // 4, SIZE // 4)
        assert event_annotator.class_name == "class_lysis"
        assert event_annotator.status_name == "status_lysis"
        df = event_annotator.df_tracks
        track0 = df[df["TRACK_ID"] == 0].sort_values("FRAME")
        assert list(track0["status_lysis"]) == [0, 0, 1, 1, 1, 1]
        assert event_annotator.contrast_slider.value()[0] < event_annotator.contrast_slider.value()[1]

    def test_correct_an_event_and_save(self, event_annotator, position):
        """A cell picked and annotated as an event at frame 3 is saved so."""

        annotator = event_annotator
        index = list(annotator.tracks[0]).index(1)
        annotator.on_scatter_pick(
            pick(annotator.status_scatter, index, annotator.positions[0])
        )
        assert annotator.track_of_interest == 1
        assert annotator.correct_btn.isEnabled()

        annotator.correct_btn.click()
        assert annotator.no_event_btn.isChecked()
        annotator.event_btn.click()
        annotator.time_of_interest_le.setText("3")
        annotator.correct_btn.click()

        assert not annotator.correct_btn.isEnabled()
        assert annotator.correct_btn.text() == "correct"
        assert annotator.selection == []

        annotator.save_btn.click()
        saved = read_table(position, "targets")
        track1 = saved[saved["TRACK_ID"] == 1].sort_values("FRAME")
        assert set(track1["class_lysis"]) == {0}
        assert set(track1["t_lysis"]) == {3.0}
        assert list(track1["status_lysis"]) == [0, 0, 0, 1, 1, 1]

    def test_remove_a_cell(self, event_annotator, position):
        """A cell marked for deletion is left out of the saved table."""

        annotator = event_annotator
        index = list(annotator.tracks[0]).index(2)
        annotator.on_scatter_pick(
            pick(annotator.status_scatter, index, annotator.positions[0])
        )
        annotator.shortcut_suppr()
        annotator.save_btn.click()

        saved = read_table(position, "targets")
        assert 2 not in set(saved["TRACK_ID"])
        assert {0, 1, 3} <= set(saved["TRACK_ID"])

    def test_browse_the_movie(self, event_annotator):
        """The movie loops on the frame buttons and pauses on the first and last frames."""

        annotator = event_annotator
        annotator.set_first_frame()
        assert annotator.framedata == 0
        annotator.prev_frame()
        assert annotator.framedata == N_FRAMES - 1
        annotator.next_frame()
        assert annotator.framedata == 0
        annotator.set_last_frame()
        assert annotator.framedata == N_FRAMES - 1
        assert annotator.frame_lbl.text() == f"frame: {N_FRAMES - 1}"

        annotator.speed_slider.setValue(10)
        assert annotator.anim_interval == 100

        annotator.start()
        assert annotator.stop_btn.isVisibleTo(annotator)
        assert not annotator.prev_frame_btn.isEnabled()
        annotator.toggle_animation()
        assert annotator.start_btn.isVisibleTo(annotator)
        assert annotator.prev_frame_btn.isEnabled()

    def test_export_signals(self, event_annotator, tmp_path):
        """The export holds a sample per track, with its class and time of interest."""

        path = str(tmp_path / "export.npy")
        with patch(
            "PyQt5.QtWidgets.QFileDialog.getSaveFileName", return_value=(path, ".npy")
        ):
            event_annotator.export_btn.click()

        data = np.load(path, allow_pickle=True)
        assert len(data) == len(TRACKS)
        assert [sample["class"] for sample in data] == [t[3] for t in TRACKS]
        assert [sample["time_of_interest"] for sample in data] == [t[4] for t in TRACKS]
        assert list(data[0]["FRAME"]) == list(range(N_FRAMES))

    def test_normalize_and_log(self, event_annotator):
        """Normalization rescales the features and back; log toggles the scale."""

        annotator = event_annotator
        original = annotator.df_tracks["area"].to_numpy().copy()
        annotator.normalize_features_btn.click()
        assert annotator.df_tracks["area"].max() == pytest.approx(1.0)
        annotator.normalize_features_btn.click()
        np.testing.assert_allclose(annotator.df_tracks["area"].to_numpy(), original)

        annotator.log_btn.click()
        assert annotator.cell_ax.get_yscale() == "log"
        annotator.log_btn.click()
        assert annotator.cell_ax.get_yscale() == "linear"


class TestEventAnnotatorFixes:

    def test_cancel_resets_the_correction(self, event_annotator):
        """
        Canceling a correction disables the correct button and rewires it.

        Bug prevented: the button stayed enabled on "submit", which applied the
        options to a stale cell and failed on the empty selection.
        """

        annotator = event_annotator
        annotator.on_scatter_pick(
            pick(annotator.status_scatter, 0, annotator.positions[0])
        )
        annotator.correct_btn.click()
        assert annotator.correct_btn.text() == "submit"

        annotator.cancel_btn.click()
        assert not annotator.correct_btn.isEnabled()
        assert annotator.correct_btn.text() == "correct"
        assert annotator.selection == []


    def test_submit_without_option_keeps_the_cell(self, event_annotator):
        """
        Submitting with no option checked changes nothing.

        Bug prevented: a cell whose class the options do not show (NaN) was
        marked for deletion on submit.
        """

        annotator = event_annotator
        annotator.on_scatter_pick(
            pick(annotator.status_scatter, 0, annotator.positions[0])
        )
        track = annotator.track_of_interest
        before = annotator.df_tracks.loc[
            annotator.df_tracks["TRACK_ID"] == track, "class_lysis"
        ].to_numpy()

        annotator.correct_btn.click()
        for button in annotator.class_options:
            button.setAutoExclusive(False)
            button.setChecked(False)
        annotator.correct_btn.click()

        after = annotator.df_tracks.loc[
            annotator.df_tracks["TRACK_ID"] == track, "class_lysis"
        ].to_numpy()
        assert list(after) == list(before)


class TestPairEventAnnotatorWorkflow:

    def test_opens_on_the_position(self, pair_annotator):
        """The neighborhood and its interaction event are found."""

        annotator = pair_annotator
        assert annotator.neighborhood_cols == [f"targets_ref_{NEIGHBORHOOD}"]
        assert annotator.current_neighborhood == NEIGHBORHOOD
        assert annotator.reference_population == annotator.neighbor_population == "targets"
        assert annotator.pair_class_name == f"class_{NEIGHBORHOOD}"
        assert annotator.pair_time_name == f"t0_{NEIGHBORHOOD}"
        assert annotator.stack.shape == (N_FRAMES, SIZE // 4, SIZE // 4)

    def test_select_a_pair_and_annotate_it(self, pair_annotator, position):
        """A pair picked and annotated as an event at frame 2 is saved so."""

        annotator = pair_annotator
        scatter = annotator.status_scatter["targets"]
        positions = annotator.positions["targets"][0]
        tracks = list(annotator.tracks["targets"][0])

        annotator.on_scatter_pick(pick(scatter, tracks.index(0), positions))
        assert annotator.reference_track_of_interest == 0
        assert set(annotator.neighbors) == {1, 2}
        assert "Ref #0" in annotator.reference_cell_info.text()

        annotator.on_scatter_pick(pick(scatter, tracks.index(2), positions))
        assert annotator.neighbor_track_of_interest == 2
        assert annotator.correct_btn.isEnabled()
        assert "Pair: (0,2)" in annotator.pair_info.text()

        annotator.correct_btn.click()
        assert annotator.no_event_btn.isChecked()
        annotator.event_btn.click()
        annotator.time_of_interest_le.setText("2")
        annotator.correct_btn.click()
        assert annotator.neighbor_track_of_interest is None
        assert annotator.reference_track_of_interest == 0

        annotator.save_btn.click()
        saved = read_table(position, "pairs")
        pair = saved[(saved["REFERENCE_ID"] == 0) & (saved["NEIGHBOR_ID"] == 2)]
        assert set(pair[f"class_{NEIGHBORHOOD}"]) == {0}
        assert set(pair[f"t0_{NEIGHBORHOOD}"]) == {2.0}
        assert list(pair.sort_values("FRAME")[f"status_{NEIGHBORHOOD}"]) == [0, 0, 1, 1, 1, 1]
        other = saved[(saved["REFERENCE_ID"] == 0) & (saved["NEIGHBOR_ID"] == 1)]
        assert set(other[f"class_{NEIGHBORHOOD}"]) == {1}

    def test_draw_the_pair_lines(self, pair_annotator):
        """With a reference selected, each frame shows a line per neighbor."""

        annotator = pair_annotator
        scatter = annotator.status_scatter["targets"]
        positions = annotator.positions["targets"][0]
        tracks = list(annotator.tracks["targets"][0])
        annotator.on_scatter_pick(pick(scatter, tracks.index(0), positions))
        annotator.set_first_frame()
        for _ in range(1, N_FRAMES):
            annotator.next_frame()
            assert len(annotator.lines_list) == 2

        # Picking the reference again unselects it.
        annotator.set_first_frame()
        annotator.on_scatter_pick(pick(scatter, tracks.index(0), positions))
        assert annotator.reference_track_of_interest is None
        assert not annotator.correct_btn.isEnabled()

    def test_export_signals(self, pair_annotator, tmp_path):
        """The export holds a sample per pair, with the signals of both cells."""

        path = str(tmp_path / "pairs.npy")
        with patch(
            "PyQt5.QtWidgets.QFileDialog.getSaveFileName", return_value=(path, ".npy")
        ):
            pair_annotator.export_btn.click()

        data = np.load(path, allow_pickle=True)
        assert len(data) == 4
        sample = data[0]
        assert sample["class"] == 1
        assert sample["neighborhood_of_interest"] == NEIGHBORHOOD
        assert list(sample["pair_distance"]) == [10.0 + t for t in range(N_FRAMES)]
        assert "reference_area" in sample and "neighbor_area" in sample


class TestPairEventAnnotatorFixes:

    def test_first_and_last_frame_buttons_keep_working(self, pair_annotator):
        """
        The first and last frame buttons can be used more than once.

        Bug prevented: they disabled themselves on their first use.
        """

        annotator = pair_annotator
        for _ in range(2):
            annotator.last_frame_btn.click()
            assert annotator.framedata == N_FRAMES - 1
            annotator.first_frame_btn.click()
            assert annotator.framedata == 0
        assert annotator.first_frame_btn.isEnabled()
        assert annotator.last_frame_btn.isEnabled()

    def test_lines_do_not_pile_up(self, pair_annotator):
        """
        The lines to the neighbors of the previous frame leave the axes.

        Bug prevented: a new line was added per neighbor and frame, forever.
        """

        annotator = pair_annotator
        scatter = annotator.status_scatter["targets"]
        tracks = list(annotator.tracks["targets"][0])
        annotator.on_scatter_pick(
            pick(scatter, tracks.index(0), annotator.positions["targets"][0])
        )
        n_lines = len(annotator.ax.lines)
        for _ in range(3 * N_FRAMES):
            annotator.next_frame()
        assert len(annotator.ax.lines) <= n_lines + 2

    def test_shortcuts_follow_the_pair_selection(self, pair_annotator):
        """
        The Delete and N shortcuts are on while a pair is selected.

        Bug prevented: they were never enabled.
        """

        annotator = pair_annotator
        scatter = annotator.status_scatter["targets"]
        positions = annotator.positions["targets"][0]
        tracks = list(annotator.tracks["targets"][0])
        annotator.on_scatter_pick(pick(scatter, tracks.index(0), positions))
        annotator.on_scatter_pick(pick(scatter, tracks.index(1), positions))
        assert annotator.no_event_shortcut.isEnabled()

        annotator.shortcut_no_event()
        assert not annotator.no_event_shortcut.isEnabled()
        assert not annotator.correct_btn.isEnabled()

    def test_next_pair_in_one_click(self, pair_annotator):
        """
        After a pair is annotated, one click selects the next neighbor.

        Bug prevented: the annotation left a pair with no neighbor selected,
        so the first click only canceled it.
        """

        annotator = pair_annotator
        scatter = annotator.status_scatter["targets"]
        positions = annotator.positions["targets"][0]
        tracks = list(annotator.tracks["targets"][0])
        annotator.on_scatter_pick(pick(scatter, tracks.index(0), positions))
        annotator.on_scatter_pick(pick(scatter, tracks.index(1), positions))
        annotator.shortcut_no_event()
        assert annotator.pair_selection == []

        annotator.on_scatter_pick(pick(scatter, tracks.index(2), positions))
        assert annotator.neighbor_track_of_interest == 2
        assert annotator.correct_btn.isEnabled()


class TestMeasureAnnotatorWorkflow:

    def test_annotate_a_phenotype_and_save(self, measure_annotator, position):
        """A phenotype set on a cell at the current frame is saved."""

        annotator = measure_annotator
        assert annotator.status_name in annotator.df_tracks.columns
        index = list(annotator.tracks[0]).index(3)
        annotator.on_scatter_pick(
            pick(annotator.viewer.scat_markers, index, annotator.positions[0])
        )
        assert annotator.track_of_interest == 3
        assert "cell: 3" in annotator.cell_info.text()

        annotator.correct_btn.click()
        annotator.time_of_interest_le.setText("2")
        annotator.correct_btn.click()
        assert not annotator.correct_btn.isEnabled()

        status_name = annotator.status_name
        annotator.save_btn.click()
        saved = read_table(position, "targets")
        cell = saved[(saved["TRACK_ID"] == 3) & (saved["FRAME"] == 0)]
        assert list(cell[status_name]) == [2]
        # The other cells keep the status made from their class.
        others = saved[(saved["TRACK_ID"] != 3) | (saved["FRAME"] != 0)]
        assert list(others[status_name]) == list(others["class_lysis"])

    def test_new_group(self, measure_annotator):
        """A new characteristic group starts at 0 and is selected."""

        annotator = measure_annotator
        annotator.add_class_btn.click()
        annotator.class_name_le.setText("shape")
        annotator.write_new_event_class()
        assert annotator.class_choice_cb.currentText() == "group_shape"
        assert set(annotator.df_tracks["group_shape"]) == {0}

    def test_browse_frames(self, measure_annotator):
        """The markers follow the frame of the viewer."""

        annotator = measure_annotator
        annotator.viewer.frame_slider.setValue(3)
        assert annotator.current_frame == 3
        offsets = annotator.viewer.scat_markers.get_offsets()
        assert len(offsets) == len(TRACKS)


class TestInteractiveEventViewerWorkflow:

    @pytest.fixture
    def viewer(self, qtbot, position):
        from celldetective.gui.interactive_timeseries_viewer import (
            InteractiveEventViewer,
        )

        _, pos = position
        path = os.path.join(pos, "output", "tables", "trajectories_targets.csv")
        updates = []
        viewer = InteractiveEventViewer(
            path, event_label="lysis", callback=lambda: updates.append(1)
        )
        viewer.updates = updates
        qtbot.addWidget(viewer)
        viewer.signal_combo.setCurrentText("area")
        yield viewer
        viewer.close()

    def test_plots_the_tracks_centered_on_their_event(self, viewer):
        """A curve per track, its time centered on the event, colored by class."""

        assert viewer.class_col == "class_lysis"
        assert viewer.time_col == "t_lysis"
        assert viewer.signal_name == "area"
        assert "area" in viewer.signal_columns()
        assert "t_lysis" not in viewer.signal_columns()
        assert sorted(viewer.lines.values()) == [0, 1, 2, 3]
        colors = {tid: line.get_color() for line, tid in viewer.lines.items()}
        assert colors == {0: "tab:red", 1: "tab:blue", 2: "tab:blue", 3: "yellow"}
        line0 = next(line for line, tid in viewer.lines.items() if tid == 0)
        assert list(line0.get_xdata()) == [t - 2.0 for t in range(N_FRAMES)]

        viewer.event_filter_combo.setCurrentText("No Events (1)")
        assert sorted(viewer.lines.values()) == [1, 2]

    def test_select_shift_and_classify(self, viewer, position):
        """Tracks selected in a box are shifted with the arrows, then classified."""

        from PyQt5.QtGui import QKeyEvent
        from PyQt5.QtCore import QEvent

        def click(x, y):
            return SimpleNamespace(xdata=x, ydata=y)

        # Track 0 is the only one with a negative centered time.
        viewer.on_select_rect(click(-2.5, 0), click(-0.5, 1000))
        assert viewer.selected_tracks == {0}

        viewer.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Left, Qt.NoModifier))
        track0 = viewer.df[viewer.df["TRACK_ID"] == 0]
        assert set(track0["t_lysis"]) == {2.5}
        assert viewer.updates

        # The others reach later centered times than track 0 now does.
        viewer.on_select_rect(click(3, 0), click(N_FRAMES + 2, 1000))
        assert viewer.selected_tracks == {1, 2, 3}
        viewer.set_class(0)
        assert viewer.selected_tracks == set()
        assert set(viewer.df["class_lysis"]) == {0}

        with patch(
            "celldetective.gui.interactive_timeseries_viewer.QMessageBox.information"
        ):
            viewer.save_changes()
        saved = read_table(position, "targets")
        assert set(saved["class_lysis"]) == {0}
