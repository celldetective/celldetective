"""Tests between the curves of the survival and signal plots.

The computations are in :mod:`celldetective.comparisons`; the results are
shown like the tests of the table explorer, as pairwise matrices
(:class:`celldetective.gui.tableUI.PivotTableUI`).
"""

from typing import Any, Optional

import pandas as pd
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from celldetective.gui.base.components import CelldetectiveWidget, generic_message, hint_label
from celldetective.gui.base.utils import center_window
from celldetective.gui.gui_utils import NumberLineEdit

WELLS_NOTE = (
    "The cells of a well are pooled, so these tests compare wells. With one well per "
    "condition, a difference may come from the well rather than its condition: repeat the "
    "experiment to conclude on a condition."
)
LOGRANK_GUIDE = (
    "<b>How to read:</b> each cell tests whether the <b>row</b> and <b>column</b> wells have "
    "the same survival (log-rank test on their cells pooled, Holm-adjusted for the number of "
    "pairs). Read the direction on the curves."
)


def show_survival_comparison(plot_window: Any) -> Optional[CelldetectiveWidget]:
    """Compare the survival of the wells shown with a matrix of log-rank
    p-values. Returns the window, or None if there is nothing to compare."""
    from celldetective.comparisons import pairwise_logrank
    from celldetective.gui.tableUI import PivotTableUI

    info = plot_window.df_well_info
    info = info.loc[info["select"].astype(bool)] if "select" in info.columns else info
    fits = {
        name: fit
        for name, fit in zip(info["well_name"], info.get("survival_fit", []))
        if hasattr(fit, "durations")
    }
    if len(fits) < 2:
        generic_message("Show at least two wells with a survival curve to compare them.")
        return None

    pvalues = PivotTableUI(pairwise_logrank(fits), title="Log-rank p-value (Holm-adjusted)", mode="pvalue")
    pvalues.set_reading_guide(LOGRANK_GUIDE + "<br>" + WELLS_NOTE)
    # Sized in the constructor for the KS guide this one replaces.
    pvalues.adjust_window_size()
    pvalues.show()
    return pvalues


class SignalComparisonWidget(CelldetectiveWidget):
    """Compare synchronised signals by the response of each cell: the mean
    over a window after the synchronising time minus the mean over a baseline
    window. The responses are tested between wells as in the table explorer
    (KS test and Cliff's delta)."""

    def __init__(self, plot_window: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.plot_window = plot_window
        self.frame_to_min = plot_window.parent_window.FrameToMin
        self.setWindowTitle("Compare signals")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)

        self.windows = {}
        for name, (lo, hi) in (("baseline", (-10, -1)), ("response", (0, 9))):
            hbox = QHBoxLayout()
            hbox.addWidget(QLabel(f"{name} [min]: "), 30)
            fields = []
            for value in (lo, hi):
                field = NumberLineEdit(
                    f"{value * self.frame_to_min:g}",
                    tooltip=f"The {name} window relative to the synchronising time, both ends included.",
                    invalid=f"The {name} window needs a number at both ends.",
                )
                field.OPTIONAL = False
                hbox.addWidget(field, 35)
                fields.append(field)
            self.windows[name] = fields
            layout.addLayout(hbox)

        layout.addWidget(hint_label(WELLS_NOTE))

        self.compare_btn = QPushButton("compare")
        self.compare_btn.setStyleSheet(self.button_style_sheet)
        self.compare_btn.clicked.connect(self.run)
        layout.addWidget(self.compare_btn)
        center_window(self)

    def window_in_frames(self, name: str) -> tuple:
        lo, hi = (self.to_frames(f.value()) for f in self.windows[name])
        if hi < lo:
            raise ValueError(f"The {name} window ends before it starts.")
        return lo, hi

    def to_frames(self, minutes: float) -> float:
        """Minutes to frames, snapped to a whole frame when within rounding:
        the fields show minutes to six significant digits, so a frame end read
        back (e.g. -0.166667 min with 10 s frames) would otherwise fall just
        short of its frame and leave it out of the window."""
        frames = minutes / self.frame_to_min
        nearest = round(frames)
        return float(nearest) if abs(frames - nearest) < 1e-3 else frames

    def alignment_problem(self) -> str:
        """Cells without the event are drawn from the start of the movie, not
        from an event: a window around the synchronising time means nothing
        for them unless all cells share an absolute time."""
        if not self.plot_window.absolute_time and list(self.plot_window.target_class) != [0]:
            return ("Cells without the event have no event time and are aligned on the start of the "
                    "movie. Show the class 'event', or synchronise on an absolute time, to compare responses.")
        return ""

    def responses(self) -> pd.DataFrame:
        """One row per cell of the positions shown, with its response."""
        from celldetective.comparisons import signal_responses

        _, _, key = self.plot_window.signal_keys()
        info = self.plot_window.df_pos_info
        return signal_responses(info.loc[info["select"].astype(bool)], key,
                                self.window_in_frames("baseline"), self.window_in_frames("response"))

    def run(self) -> None:
        from celldetective.comparisons import compare_responses
        from celldetective.gui.tableUI import TWO_SAMPLE_TESTS, PivotTableUI

        problem = self.alignment_problem()
        if problem:
            generic_message(problem)
            return None
        try:
            cells = self.responses()
        except ValueError as e:
            generic_message(str(e))
            return None
        if cells.empty or cells["well_name"].nunique() < 2:
            generic_message("Not enough cells observed over both windows in at least two wells.")
            return None

        self.cells = cells
        self.result_windows = [
            PivotTableUI(pivot, title=TWO_SAMPLE_TESTS[method][0], mode=TWO_SAMPLE_TESTS[method][1])
            for method, pivot in compare_responses(cells).items()
        ]
        for window in self.result_windows:
            window.show()
