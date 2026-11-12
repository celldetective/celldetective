"""Tests of the comparison of survival curves and synchronised signals
(celldetective.gui.comparison_ui), and of the band of the signal plot."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from lifelines import KaplanMeierFitter

from celldetective.gui.comparison_ui import SignalComparisonWidget, show_survival_comparison
from celldetective.gui.generic_signal_plot import BAND_CI, BAND_SPREAD, SignalPlotWidget, signal_band

TIMELINE = np.arange(-20, 21, dtype=float)


def _infos(rng, wells=("W1", "W2"), positions=3):
    """Position and well tables as the survival and signal windows build them."""
    pos_rows, well_rows = [], []
    for w, well in enumerate(wells):
        t = rng.exponential(50 * (w + 1), 120)
        well_rows.append(dict(well_name=well, select=True,
                              survival_fit=KaplanMeierFitter().fit(np.minimum(t, 100), (t <= 100).astype(int))))
        for p in range(positions):
            m = rng.normal(0, 1, (10, len(TIMELINE)))
            m[:, TIMELINE >= 0] += 2.0 * w
            pos_rows.append(dict(well_name=well, pos_name=str(p), select=True,
                                 signal=dict(timeline=TIMELINE, matrix_event=m, matrix_all=m, matrix_no_event=m)))
    return pd.DataFrame(pos_rows), pd.DataFrame(well_rows)


def _plot_window(rng, target_class=(0,), absolute=False, **kwargs):
    pos_info, well_info = _infos(rng, **kwargs)
    window = SimpleNamespace(df_pos_info=pos_info, df_well_info=well_info,
                             parent_window=SimpleNamespace(FrameToMin=1.0),
                             target_class=list(target_class), absolute_time=absolute)
    window.signal_keys = lambda: SignalPlotWidget.signal_keys(window)
    return window


@pytest.fixture
def windows():
    """The result windows, created but not shown."""
    with patch("celldetective.gui.comparison_ui.generic_message") as message,             patch("celldetective.gui.tableUI.PivotTableUI") as pivot:
        yield SimpleNamespace(message=message, pivot=pivot)


def test_survival_comparison_shows_log_rank(windows):
    shown = show_survival_comparison(_plot_window(np.random.default_rng(0)))
    assert shown is windows.pivot.return_value
    pivot = windows.pivot.call_args[0][0]
    assert list(pivot.index) == ["W1", "W2"] and pivot.loc["W1", "W2"] < 0.05
    assert windows.pivot.call_args[1]["mode"] == "pvalue"


def test_survival_comparison_needs_two_wells(windows):
    assert show_survival_comparison(_plot_window(np.random.default_rng(1), wells=("W1",))) is None
    windows.message.assert_called_once()


def test_signal_comparison_reuses_the_table_explorer_tests(qtbot, windows):
    widget = SignalComparisonWidget(_plot_window(np.random.default_rng(2)))
    qtbot.addWidget(widget)
    assert [f.text() for f in widget.windows["baseline"]] == ["-10", "-1"]
    widget.run()
    means = widget.cells.groupby("well_name")["response"].mean()
    assert means["W2"] - means["W1"] == pytest.approx(2.0, abs=0.4)
    modes = [c[1]["mode"] for c in windows.pivot.call_args_list]
    assert modes == ["pvalue", "cliff"]


def test_signal_comparison_refuses_cells_without_event_time(qtbot, windows):
    widget = SignalComparisonWidget(_plot_window(np.random.default_rng(3), target_class=(1,)))
    qtbot.addWidget(widget)
    widget.run()
    assert "no event time" in windows.message.call_args[0][0]
    assert not hasattr(widget, "cells")


def test_signal_comparison_accepts_all_cells_on_absolute_time(qtbot, windows):
    widget = SignalComparisonWidget(_plot_window(np.random.default_rng(4), target_class=(0, 1), absolute=True))
    qtbot.addWidget(widget)
    widget.run()
    assert hasattr(widget, "cells")


def test_signal_window_ending_before_it_starts_is_refused(qtbot, windows):
    widget = SignalComparisonWidget(_plot_window(np.random.default_rng(5)))
    qtbot.addWidget(widget)
    widget.windows["response"][1].setText("-5")
    widget.run()
    assert "ends before it starts" in windows.message.call_args[0][0]


# ---------------------------------------------------------------------------
# Band of the signal plot
# ---------------------------------------------------------------------------


def test_signal_keys_follow_the_class_shown():
    keys = lambda c: SignalPlotWidget.signal_keys(SimpleNamespace(target_class=c))
    assert keys([0, 1]) == ("mean_all", "std_all", "matrix_all")
    assert keys([0]) == ("mean_event", "std_event", "matrix_event")
    assert keys([1]) == ("mean_no_event", "std_no_event", "matrix_no_event")


def test_spread_band_is_the_spread_of_cells():
    lower, upper = signal_band([1.0, 3.0], [1.0, 1.0], None, BAND_SPREAD)
    assert np.allclose(lower, [0.0, 2.0]) and np.allclose(upper, [2.0, 4.0])


def test_ci_band_is_the_interval_of_the_mean():
    rng = np.random.default_rng(7)
    m = rng.normal(0, 1, (12, 1))
    lower, upper = signal_band(m.mean(0), m.std(0), np.array([12]), BAND_CI)
    half = 2.201 * m.std(ddof=1) / np.sqrt(12)  # t(11) = 2.201
    assert upper[0] - m.mean() == pytest.approx(half, rel=1e-3)


def test_ci_band_needs_two_cells():
    lower, upper = signal_band([1.0, 2.0], [0.5, 0.0], np.array([5, 1]), BAND_CI)
    assert np.isfinite(lower[0]) and np.isnan(lower[1])



def test_signal_windows_keep_their_end_frames_with_uneven_calibration(qtbot, windows):
    # 10 s frames: -1 frame shows as -0.166667 min, which read back falls short of -1.
    window = _plot_window(np.random.default_rng(6))
    window.parent_window.FrameToMin = 1 / 6
    widget = SignalComparisonWidget(window)
    qtbot.addWidget(widget)
    assert widget.window_in_frames("baseline") == (-10.0, -1.0)
    assert widget.window_in_frames("response") == (0.0, 9.0)
