"""Tests for the shared event schema and the headless threshold event-detector
runtime (the non-DL event detector, made reusable via a saved config)."""

import numpy as np
import pandas as pd
import pytest

from celldetective.utils.event_schema import (
    event_column_names,
    status_from_event,
    EVENT,
    NO_EVENT,
    ELSE,
)
from celldetective.measure import classify_from_threshold_config


class TestSchema:
    def test_normalize_population_aliases(self):
        from celldetective.utils.schema import normalize_population

        assert normalize_population("target") == "targets"
        assert normalize_population("targets") == "targets"
        assert normalize_population("effector") == "effectors"
        assert normalize_population("Effectors") == "effectors"
        # custom/unknown passes through unchanged
        assert normalize_population("pairs") == "pairs"
        assert normalize_population("my_pop") == "my_pop"

    def test_trajectory_table_name(self):
        from celldetective.utils.schema import trajectory_table_name

        assert trajectory_table_name("target") == "trajectories_targets.csv"
        assert trajectory_table_name("effectors") == "trajectories_effectors.csv"
        assert trajectory_table_name("pairs") == "trajectories_pairs.csv"

    def test_trajectory_table_path(self):
        import os
        from celldetective.utils.schema import trajectory_table_path

        p = trajectory_table_path("/exp/W1/100", "targets")
        assert p == os.path.join(
            "/exp/W1/100", "output", "tables", "trajectories_targets.csv"
        )

    def test_reexports_event_encoding(self):
        from celldetective.utils import schema

        assert (schema.EVENT, schema.NO_EVENT, schema.ELSE) == (0, 1, 2)
        assert schema.event_column_names("x") == ("class_x", "t_x", "status_x")


class TestEventSchema:
    def test_column_names_default(self):
        assert event_column_names(None) == ("class", "t0", "status")
        assert event_column_names("") == ("class", "t0", "status")

    def test_column_names_labelled(self):
        assert event_column_names("lysis") == (
            "class_lysis",
            "t_lysis",
            "status_lysis",
        )

    def test_status_event(self):
        s = status_from_event(np.array([0, 1, 2, 3, 4]), EVENT, 2)
        np.testing.assert_array_equal(s, [0, 0, 1, 1, 1])

    def test_status_no_event(self):
        s = status_from_event(np.array([0, 1, 2]), NO_EVENT, -1)
        np.testing.assert_array_equal(s, [0, 0, 0])

    def test_status_else(self):
        s = status_from_event(np.array([0, 1, 2]), ELSE, -1)
        np.testing.assert_array_equal(s, [2, 2, 2])

    def test_status_nan_t0(self):
        s = status_from_event(np.array([0, 1, 2]), EVENT, np.nan)
        np.testing.assert_array_equal(s, [0, 0, 0])


def _tracked_df():
    """Two tracks: one with a rising signal (event), one flat (no event)."""
    rows = []
    for tid, signal in [(1, [0, 0, 1, 1, 1]), (2, [0, 0, 0, 0, 0])]:
        for f, s in enumerate(signal):
            rows.append(
                {
                    "position": "/exp/W1/100/",
                    "TRACK_ID": tid,
                    "FRAME": f,
                    "signal": float(s),
                    "t_firstdetection": 0,
                }
            )
    return pd.DataFrame(rows)


class TestClassifyFromThresholdConfig:
    def test_requires_name_and_query(self):
        with pytest.raises(KeyError):
            classify_from_threshold_config(_tracked_df(), {"name": "x"})

    def test_static_group(self):
        out = classify_from_threshold_config(
            _tracked_df(),
            {"name": "hi", "query": "signal > 0.5", "time_correlated": False},
        )
        assert "group_hi" in out.columns
        assert "class_hi" not in out.columns
        assert set(out["group_hi"].dropna().unique()) <= {0.0, 1.0}

    def test_time_correlated_emits_event_columns(self):
        out = classify_from_threshold_config(
            _tracked_df(),
            {
                "name": "ev",
                "query": "signal > 0.5",
                "time_correlated": True,
                "event_type": "irreversible",
                "r2_threshold": 0.4,
            },
        )
        for col in ["class_ev", "t_ev", "status_ev"]:
            assert col in out.columns
        c1 = out.loc[out["TRACK_ID"] == 1, "class_ev"].iloc[0]
        c2 = out.loc[out["TRACK_ID"] == 2, "class_ev"].iloc[0]
        assert c1 == EVENT  # rising track -> event
        assert c2 == NO_EVENT  # flat track -> no event

    def test_missing_t_firstdetection_raises_clearly(self):
        df = _tracked_df().drop(columns=["t_firstdetection"])
        with pytest.raises(KeyError):
            classify_from_threshold_config(
                df,
                {
                    "name": "ev",
                    "query": "signal > 0.5",
                    "time_correlated": True,
                    "event_type": "irreversible",
                },
            )


class TestClassifyPositionFromConfig:
    def _write_position(self, tmp_path):
        from celldetective import signals  # noqa: F401 (ensure importable)

        pos = tmp_path / "exp" / "W1" / "100"
        (pos / "output" / "tables").mkdir(parents=True)
        df = _tracked_df()
        df["position"] = str(pos)
        df.to_csv(pos / "output" / "tables" / "trajectories_targets.csv", index=False)
        return pos

    def test_event_config_writes_columns_and_colors(self, tmp_path):
        from celldetective.signals import classify_position_from_config

        pos = self._write_position(tmp_path)
        config = {
            "name": "death",
            "query": "signal > 0.5",
            "time_correlated": True,
            "event_type": "irreversible",
            "r2_threshold": 0.4,
        }
        out = classify_position_from_config(str(pos), config, mode="targets")
        for col in ["class_death", "t_death", "status_death", "status_color"]:
            assert col in out.columns

        reread = pd.read_csv(
            pos / "output" / "tables" / "trajectories_targets.csv"
        )
        assert "class_death" in reread.columns

    def test_static_config_writes_group(self, tmp_path):
        from celldetective.signals import classify_position_from_config

        pos = self._write_position(tmp_path)
        config = {"name": "bright", "query": "signal > 0.5", "time_correlated": False}
        out = classify_position_from_config(str(pos), config, mode="targets")
        assert "group_bright" in out.columns
        assert "status_color" not in out.columns  # static config sets no event colors

    def test_pairs_mode_static(self, tmp_path):
        from celldetective.signals import classify_position_from_config

        pos = tmp_path / "exp" / "W1" / "100"
        (pos / "output" / "tables").mkdir(parents=True)
        # Pair table: no TRACK_ID, keyed on REFERENCE_ID/NEIGHBOR_ID.
        df = pd.DataFrame(
            {
                "position": [str(pos)] * 4,
                "REFERENCE_ID": [1, 1, 2, 2],
                "NEIGHBOR_ID": [10, 10, 20, 20],
                "pair_FRAME": [0, 1, 0, 1],
                "distance": [5.0, 50.0, 5.0, 50.0],
            }
        )
        df.to_csv(pos / "output" / "tables" / "trajectories_pairs.csv", index=False)

        out = classify_position_from_config(
            str(pos),
            {"name": "close", "query": "distance < 10", "time_correlated": False},
            mode="pairs",
        )
        assert "group_close" in out.columns
        reread = pd.read_csv(pos / "output" / "tables" / "trajectories_pairs.csv")
        assert "group_close" in reread.columns

    def test_missing_table_raises(self, tmp_path):
        from celldetective.signals import classify_position_from_config

        with pytest.raises(FileNotFoundError):
            classify_position_from_config(
                str(tmp_path / "nope"), {"name": "x", "query": "signal > 0"}
            )

    def test_back_to_back_configs_accumulate_columns(self, tmp_path):
        from celldetective.signals import classify_position_from_config

        pos = self._write_position(tmp_path)
        # Apply two static configs in sequence (as the CLASSIFY step does).
        classify_position_from_config(
            str(pos),
            {"name": "hi", "query": "signal > 0.5", "time_correlated": False},
            mode="targets",
        )
        classify_position_from_config(
            str(pos),
            {"name": "lo", "query": "signal < 0.5", "time_correlated": False},
            mode="targets",
        )
        reread = pd.read_csv(pos / "output" / "tables" / "trajectories_targets.csv")
        assert "group_hi" in reread.columns
        assert "group_lo" in reread.columns

    def test_batch_skips_missing_and_counts(self, tmp_path):
        from celldetective.signals import classify_positions_from_config

        pos = self._write_position(tmp_path)
        config = {"name": "bright", "query": "signal > 0.5", "time_correlated": False}
        # one valid position + one missing path
        done = classify_positions_from_config(
            [str(pos), str(tmp_path / "missing")], config, mode="targets"
        )
        assert done == 1
        reread = pd.read_csv(pos / "output" / "tables" / "trajectories_targets.csv")
        assert "group_bright" in reread.columns
