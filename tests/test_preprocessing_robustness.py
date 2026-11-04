"""What the corrections do when a run goes wrong, and when a movie is not what it seems."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

import celldetective.preprocessing as preprocessing
from celldetective.extra_properties import intensity_radial_gradient
from celldetective.measure import measure_features
from celldetective.preprocessing import (
    UnfittableBackgroundError,
    already_corrected_movies,
    first_step_outputs,
    correct_background_model,
    correct_background_model_free,
    correct_channel_offset,
    field_correction,
    fit_background_model,
    fit_paraboloid,
    fit_plane,
    register_stacks,
)
from celldetective.utils.experiment import get_position_movie_path
from tests.preprocessing_helpers import FIELD, write_experiment

QUIET = dict(show_progress_per_well=False, show_progress_per_pos=False)


def _offset(exp_dir, **kwargs):
    correct_channel_offset(
        exp_dir,
        target_channel="Channel1",
        correction_horizontal=1,
        export=True,
        **QUIET,
        **kwargs,
    )


def _fail_on_call(monkeypatch, call):
    """Make the n-th frame read fail, as a lost disk would mid-export."""
    calls = [0]
    load_frames = preprocessing.load_frames

    def failing(*args, **kwargs):
        calls[0] += 1
        if calls[0] == call:
            raise OSError("read failed")
        return load_frames(*args, **kwargs)

    monkeypatch.setattr(preprocessing, "load_frames", failing)


# ------------------------------------------------------------------ interrupted exports


def test_failed_export_leaves_no_truncated_stack(tmp_path, monkeypatch):
    exp_dir, (movie_dir,) = write_experiment(tmp_path, [1])
    _fail_on_call(monkeypatch, 2)

    with pytest.raises(OSError):
        _offset(exp_dir)

    assert sorted(f.name for f in movie_dir.iterdir()) == ["sample.tif"]


def test_failed_export_keeps_the_previous_stack(tmp_path, monkeypatch):
    exp_dir, (movie_dir,) = write_experiment(tmp_path, [1])
    _offset(exp_dir)
    previous = tifffile.imread(movie_dir / "Corrected_sample.tif")
    _fail_on_call(monkeypatch, 2)

    with pytest.raises(OSError):
        correct_channel_offset(
            exp_dir, target_channel="Channel1", correction_horizontal=5, export=True, **QUIET
        )

    np.testing.assert_array_equal(
        tifffile.imread(movie_dir / "Corrected_sample.tif"), previous
    )
    assert not list(movie_dir.glob("*.tmp"))


def test_temporary_file_cannot_be_selected_as_a_movie(tmp_path):
    # Left behind by a killed process, which cannot clean up.
    _, (movie_dir,) = write_experiment(tmp_path, [1])
    (movie_dir / "Corrected_sample.tif.tmp").write_bytes(b"")

    assert get_position_movie_path(str(movie_dir.parent), "Corrected") is None


# ------------------------------------------------------------------ re-runs


def test_empty_prefix_reports_the_corrected_stack_and_its_new_name(tmp_path):
    exp_dir, (movie_dir,) = write_experiment(tmp_path, [1], prefix="")
    assert already_corrected_movies(exp_dir) == []

    _offset(exp_dir)

    assert already_corrected_movies(exp_dir) == [
        (str(movie_dir / "Corrected_sample.tif"), "Corrected_Corrected_sample.tif")
    ]


def test_prefix_set_as_the_guides_ask_reports_the_corrected_stack(tmp_path):
    exp_dir, (movie_dir,) = write_experiment(tmp_path, [1])
    _offset(exp_dir)
    config = tmp_path / "Experiment" / "config.ini"
    config.write_text(
        config.read_text().replace("movie_prefix = sample", "movie_prefix = Corrected_")
    )

    ((source, output),) = already_corrected_movies(exp_dir)

    assert source.endswith("Corrected_sample.tif")
    assert output == "Corrected_Corrected_sample.tif"


def test_prefix_of_the_raw_movie_reports_nothing_after_a_run(tmp_path):
    exp_dir, _ = write_experiment(tmp_path, [1])
    _offset(exp_dir)

    assert already_corrected_movies(exp_dir) == []


def test_chain_step_reads_what_the_first_step_wrote(tmp_path):
    # A previous run left a deeper corrected stack: "Corrected_sample" does not select it.
    exp_dir, (movie_dir,) = write_experiment(tmp_path, [1])
    _offset(exp_dir)
    (movie_dir / "Corrected_Corrected_sample.tif").write_bytes(
        (movie_dir / "sample.tif").read_bytes()
    )

    source = get_position_movie_path(str(movie_dir.parent), "Corrected_sample")

    assert source == str(movie_dir / "Corrected_sample.tif")


def test_chain_reads_the_exact_stack_of_the_first_step_with_an_empty_prefix(tmp_path):
    # "Corrected_*" would select the stale stack, which sorts first.
    exp_dir, (movie_dir,) = write_experiment(tmp_path, [1], prefix="")
    (movie_dir / "sample.tif").rename(movie_dir / "Alpha.tif")
    sources = first_step_outputs(exp_dir)
    assert [Path(s).name for s in sources.values()] == ["Corrected_Alpha.tif"]
    _offset(exp_dir)
    stale = movie_dir / "Corrected_Corrected_Alpha.tif"
    stale.write_bytes((movie_dir / "Alpha.tif").read_bytes())
    stale_bytes = stale.read_bytes()

    _offset(exp_dir, movie_prefix="Corrected", export_prefix=None, source_stacks=sources)

    assert stale.read_bytes() == stale_bytes
    assert sorted(f.name for f in movie_dir.iterdir() if f.suffix == ".tif") == [
        "Alpha.tif",
        "Corrected_Alpha.tif",
        "Corrected_Corrected_Alpha.tif",
    ]


def test_chained_model_free_ignores_the_stale_stack_of_another_position(tmp_path):
    exp_dir, (kept, stale) = write_experiment(tmp_path, [2])
    for movie_dir, level in ((kept, 7.0), (stale, 1000.0)):
        tifffile.imwrite(
            movie_dir / "Corrected_sample.tif",
            np.full((3, FIELD, FIELD), level, dtype=np.float32),
            imagej=True,
            metadata={"axes": "TYX"},
        )
    position = str(kept.parent) + os.sep

    (corrected,) = correct_background_model_free(
        exp_dir,
        position_option=[0],
        target_channel="Channel1",
        operation="subtract",
        movie_prefix="Corrected",
        source_stacks={position: str(kept / "Corrected_sample.tif")},
        return_stacks=True,
        **QUIET,
    )

    # With the stale position in the median, the background would not be the 7 of this one.
    np.testing.assert_allclose(corrected, 0.0, atol=1e-5)


def test_a_position_without_a_source_stack_is_skipped(tmp_path):
    exp_dir, movie_dirs = write_experiment(tmp_path, [2], frames=1)

    _offset(exp_dir, source_stacks={})

    assert not [f for d in movie_dirs for f in d.glob("Corrected_*")]


# ------------------------------------------------------------------ uneven wells


def test_positions_past_the_end_of_a_well_are_skipped(tmp_path):
    exp_dir, movie_dirs = write_experiment(tmp_path, [3, 2], frames=1)

    _offset(exp_dir, well_option=[0, 1], position_option=[0, 1, 2])

    assert [(d / "Corrected_sample.tif").exists() for d in movie_dirs] == [True] * 5


def test_all_positions_reach_a_well_larger_than_the_first(tmp_path):
    exp_dir, movie_dirs = write_experiment(tmp_path, [2, 3], frames=1)

    _offset(exp_dir, well_option="*", position_option="*")

    assert [(d / "Corrected_sample.tif").exists() for d in movie_dirs] == [True] * 5


def test_a_position_index_is_taken_in_each_well(tmp_path):
    exp_dir, movie_dirs = write_experiment(tmp_path, [2, 2], frames=1)

    _offset(exp_dir, well_option=[0, 1], position_option=[1])

    assert [(d / "Corrected_sample.tif").exists() for d in movie_dirs] == [
        False,
        True,
        False,
        True,
    ]


# ------------------------------------------------------------------ failures reported


def test_fit_passes_the_activation_protocol_on(tmp_path, monkeypatch):
    exp_dir, _ = write_experiment(tmp_path, [1], frames=1)
    received = []
    original = preprocessing.field_correction

    def spy(*args, **kwargs):
        received.append(kwargs["activation_protocol"])
        return original(*args, **kwargs)

    monkeypatch.setattr(preprocessing, "field_correction", spy)
    protocol = [["gauss", 1], ["std", 3]]

    correct_background_model(
        exp_dir,
        target_channel="Channel1",
        model="plane",
        activation_protocol=protocol,
        **QUIET,
    )

    assert received == [protocol]


@pytest.mark.parametrize("model", ["plane", "paraboloid"])
def test_fit_without_background_pixels_raises(model):
    image = 100 + np.random.default_rng(0).normal(0, 5, (FIELD, FIELD))

    with pytest.raises(ValueError, match="background pixels"):
        field_correction(image, threshold=0.0, model=model)


@pytest.mark.parametrize("fit", [fit_plane, fit_paraboloid])
def test_a_masked_image_cannot_be_fitted(fit):
    with pytest.raises(ValueError, match="background pixels"):
        fit(np.ones((FIELD, FIELD)), cell_masks=np.ones((FIELD, FIELD)))


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        fit_background_model(np.ones((FIELD, FIELD)), model="sphere")


@pytest.mark.parametrize(
    "correct", [correct_background_model, correct_background_model_free]
)
def test_unknown_operation_raises(tmp_path, correct):
    exp_dir, (movie_dir,) = write_experiment(tmp_path, [1])

    with pytest.raises(ValueError, match="Unknown operation"):
        correct(exp_dir, target_channel="Channel1", operation="bad", export=True, **QUIET)

    assert not (movie_dir.parent / "log_preprocessing.txt").exists()


def test_threshold_masking_everything_is_reported_and_writes_nothing(tmp_path):
    exp_dir, (movie_dir,) = write_experiment(tmp_path, [1])

    with pytest.raises(RuntimeError, match="threshold is too low"):
        correct_background_model_free(
            exp_dir,
            target_channel="Channel1",
            threshold_on_std=0.0,
            export=True,
            **QUIET,
        )

    assert not (movie_dir / "Corrected_sample.tif").exists()


def test_skipped_well_is_reported_after_correcting_the_others(tmp_path):
    exp_dir, movie_dirs = write_experiment(tmp_path, [1, 1])
    # The first well has nothing to estimate from: its only frames are blank.
    tifffile.imwrite(
        movie_dirs[0] / "sample.tif",
        np.zeros((3, FIELD, FIELD), dtype=np.float32),
        imagej=True,
        metadata={"axes": "TYX"},
    )

    with pytest.raises(RuntimeError, match="W1"):
        correct_background_model_free(
            exp_dir,
            target_channel="Channel1",
            threshold_on_std=100,
            export=True,
            **QUIET,
        )

    assert not (movie_dirs[0] / "Corrected_sample.tif").exists()
    assert (movie_dirs[1] / "Corrected_sample.tif").exists()


def test_well_without_a_movie_is_not_reported_as_failed(tmp_path):
    # The empty folders a plate with uneven sampling starts with.
    exp_dir, movie_dirs = write_experiment(tmp_path, [1, 1])
    (movie_dirs[1] / "sample.tif").unlink()

    correct_background_model_free(
        exp_dir,
        target_channel="Channel1",
        threshold_on_std=100,
        export=True,
        **QUIET,
    )

    assert (movie_dirs[0] / "Corrected_sample.tif").exists()


def test_well_without_a_selected_position_estimates_nothing(tmp_path, monkeypatch):
    exp_dir, _ = write_experiment(tmp_path, [2, 1])
    wells_estimated = []
    estimate = preprocessing._estimate_well_background

    def spy(positions, *args, **kwargs):
        wells_estimated.append(len(positions))
        return estimate(positions, *args, **kwargs)

    monkeypatch.setattr(preprocessing, "_estimate_well_background", spy)

    correct_background_model_free(
        exp_dir,
        well_option=[0, 1],
        position_option=[1],
        target_channel="Channel1",
        threshold_on_std=100,
        export=True,
        **QUIET,
    )

    assert wells_estimated == [2]


def test_cancelling_after_a_skipped_well_is_not_reported_as_failed(tmp_path):
    exp_dir, movie_dirs = write_experiment(tmp_path, [1, 1])
    tifffile.imwrite(
        movie_dirs[0] / "sample.tif",
        np.zeros((3, FIELD, FIELD), dtype=np.float32),
        imagej=True,
        metadata={"axes": "TYX"},
    )
    asked = []

    def cancel_at_the_second_well(**kwargs):
        # Called with nothing to report only to know whether to stop.
        if not kwargs:
            asked.append(True)
            return False if len(asked) == 2 else None

    correct_background_model_free(
        exp_dir,
        target_channel="Channel1",
        threshold_on_std=100,
        export=True,
        progress_callback=cancel_at_the_second_well,
        **QUIET,
    )

    assert not (movie_dirs[1] / "Corrected_sample.tif").exists()


def test_failed_replace_leaves_no_temporary_file(tmp_path, monkeypatch):
    exp_dir, (movie_dir,) = write_experiment(tmp_path, [1])

    def locked(*args):
        raise PermissionError("the stack is open elsewhere")

    monkeypatch.setattr(preprocessing.os, "replace", locked)

    with pytest.raises(PermissionError):
        _offset(exp_dir)

    assert sorted(f.name for f in movie_dir.iterdir()) == ["sample.tif"]


def test_blank_frame_returns_the_background_when_asked():
    corrected, background = field_correction(
        np.zeros((FIELD, FIELD)), threshold=1, return_bg=True
    )

    assert np.all(corrected == 0)
    assert np.all(np.isnan(background))


def test_division_by_a_background_below_zero_is_nan():
    size = 128
    yy, xx = np.mgrid[:size, :size]
    r2 = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2) / (size / 2) ** 2
    image = 50 * (1 - 0.7 * r2)

    corrected, background = field_correction(
        image, threshold=1e5, model="paraboloid", operation="divide", return_bg=True
    )

    assert np.any(background <= 0)
    assert np.all(np.isnan(corrected[background <= 0]))
    np.testing.assert_allclose(
        corrected[background > 0], (image / background)[background > 0]
    )


def test_measurement_skips_an_unfittable_frame_and_says_which(monkeypatch):
    warned = []
    monkeypatch.setattr("celldetective.measure.logger.warning", warned.append)
    image = 100 + np.random.default_rng(0).normal(0, 5, (FIELD, FIELD, 2))
    label = np.zeros((FIELD, FIELD), dtype=int)
    label[20:30, 20:30] = 1
    norm = dict(
        correction_type="fit",
        target_channel="first",
        threshold_on_std=0.0,
        operation="divide",
        model="paraboloid",
        clip=False,
    )

    measure = dict(features=["area", "intensity_mean"], channels=["first", "second"])

    skipped = measure_features(
        image.copy(), label, normalisation_list=[norm], frame=4, **measure
    )
    fitted = measure_features(
        image.copy(), label, normalisation_list=[dict(norm, threshold_on_std=1000)], **measure
    )
    # The frames are concatenated at the end of a measurement.
    table = pd.concat([fitted, skipped], ignore_index=True)

    assert any("frame 4, channel 'first'" in message for message in warned)
    assert table["area"].tolist() == [100, 100]
    assert table["first_mean"].notna().tolist() == [True, False]
    assert table["second_mean"].iloc[1] == pytest.approx(image[20:30, 20:30, 1].mean())


def test_radial_gradient_of_an_empty_channel_is_missing():
    mask = np.ones((9, 9), dtype=bool)

    assert np.isnan(intensity_radial_gradient(mask, np.full((9, 9), np.nan))).all()


def test_unfittable_background_is_a_value_error():
    assert issubclass(UnfittableBackgroundError, ValueError)


# ------------------------------------------------------------------ surfaces


def _simulated_field(size=256):
    """An illumination paraboloid, cells of known signal on top, and noise."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:size, :size]
    illumination = 100 + 400 * (
        1 - 0.5 * ((yy - size / 2) ** 2 + (xx - size / 2) ** 2) / (size / 2) ** 2
    )
    cells = np.zeros((size, size))
    for cy, cx in rng.uniform(0, size, (40, 2)):
        cells += 300 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 6**2))
    return illumination, cells, illumination + cells + rng.normal(0, 3, (size, size))


def test_paraboloid_recovers_a_known_illumination_under_cells():
    illumination, cells, image = _simulated_field()

    corrected, background = field_correction(
        image, threshold=8, model="paraboloid", operation="subtract", return_bg=True
    )

    assert np.mean(np.abs(background - illumination) / illumination) < 0.01
    bright = cells > 30
    assert corrected[bright].sum() / cells[bright].sum() == pytest.approx(1.0, abs=0.03)


def test_plane_honours_the_downsampling(monkeypatch):
    image = np.fromfunction(lambda y, x: 2.0 * x + 3.0 * y + 5.0, (FIELD, FIELD))
    sizes = []
    from lmfit import Model

    fit = Model.fit

    def spy(self, data, *args, **kwargs):
        sizes.append(data.size)
        return fit(self, data, *args, **kwargs)

    monkeypatch.setattr(Model, "fit", spy)

    fitted = fit_background_model(image, model="plane", downsample=4)

    assert sizes == [(FIELD // 4) ** 2]
    np.testing.assert_allclose(fitted, image, rtol=1e-3)


# ------------------------------------------------------------------ registration


def test_registration_without_export_leaves_no_trace(tmp_path):
    exp_dir, (movie_dir,) = write_experiment(tmp_path, [1])

    register_stacks(exp_dir, target_channel="Channel1", export=False, **QUIET)

    assert sorted(f.name for f in movie_dir.iterdir()) == ["sample.tif"]
    assert not (movie_dir.parent / "log_preprocessing.txt").exists()


def test_failed_registration_leaves_no_shifts(tmp_path, monkeypatch):
    exp_dir, (movie_dir,) = write_experiment(tmp_path, [1])
    # Past the drift pass, which reads one channel of each of the three frames.
    _fail_on_call(monkeypatch, 5)

    with pytest.raises(OSError):
        register_stacks(exp_dir, target_channel="Channel1", export=True, **QUIET)

    assert sorted(f.name for f in movie_dir.iterdir()) == ["sample.tif"]
