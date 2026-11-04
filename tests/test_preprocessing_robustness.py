"""What the corrections do when a run goes wrong, and when a movie is not what it seems."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

import celldetective.preprocessing as preprocessing
from celldetective.measure import (
    blank_skipped_channels,
    measure_features,
    normalise_by_cell,
)
from celldetective.regionprops import regionprops_table
from celldetective.preprocessing import (
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
from tests.preprocessing_helpers import FIELD, write_experiment, write_movie

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


def test_chained_model_free_uses_the_movie_of_an_unprocessed_position(tmp_path, monkeypatch):
    exp_dir, (kept, other) = write_experiment(tmp_path, [2])
    for movie_dir, level in ((kept, 7.0), (other, 1000.0)):
        write_movie(movie_dir / "Corrected_sample.tif", np.full((3, FIELD, FIELD), level))
    write_movie(other / "sample.tif", np.full((3, FIELD, FIELD), 101.0))
    position = str(kept.parent) + os.sep
    warned = []
    monkeypatch.setattr(preprocessing.logger, "warning", warned.append)

    (corrected,) = correct_background_model_free(
        exp_dir,
        position_option=[0],
        target_channel="Channel1",
        operation="subtract",
        threshold_on_std=1000,
        movie_prefix="Corrected",
        source_stacks={position: str(kept / "Corrected_sample.tif")},
        return_stacks=True,
        **QUIET,
    )

    # The median of the well is that of 7 and of the movie at 101, not of the stale 1000.
    np.testing.assert_allclose(np.nanmedian(corrected), 7 - 54, atol=1e-3)
    assert any("not processed by the previous step" in message for message in warned)


@pytest.mark.parametrize("chained", [False, True])
def test_model_free_background_is_built_once_per_well_from_all_positions(
    tmp_path, monkeypatch, chained
):
    exp_dir, movie_dirs = write_experiment(tmp_path, [2, 2], frames=1)
    sources = first_step_outputs(exp_dir) if chained else None
    if chained:
        _offset(exp_dir)
    calls = []
    estimate = preprocessing._estimate_well_background

    def spy(positions, *args, **kwargs):
        calls.append(len(positions))
        return estimate(positions, *args, **kwargs)

    monkeypatch.setattr(preprocessing, "_estimate_well_background", spy)

    correct_background_model_free(
        exp_dir,
        position_option=[0],
        target_channel="Channel1",
        operation="subtract",
        threshold_on_std=1000,
        source_stacks=sources,
        **QUIET,
    )

    assert calls == [2, 2]


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
    write_movie(movie_dirs[0] / "sample.tif", np.zeros((3, FIELD, FIELD)))

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
    write_movie(movie_dirs[0] / "sample.tif", np.zeros((3, FIELD, FIELD)))
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


def _unfittable_normalisation():
    return dict(
        correction_type="fit",
        target_channel="first",
        threshold_on_std=0.0,
        operation="divide",
        model="paraboloid",
        clip=False,
    )


def test_a_second_correction_of_a_skipped_channel_is_skipped_too():
    image = 100 + np.random.default_rng(0).normal(0, 5, (FIELD, FIELD, 2))
    label = np.zeros((FIELD, FIELD), dtype=int)
    label[20:30, 20:30] = 1
    norm = _unfittable_normalisation()

    table = measure_features(
        image,
        label,
        features=["area", "intensity_mean"],
        channels=["first", "second"],
        normalisation_list=[norm, dict(norm, threshold_on_std=1000)],
    )

    assert "first_mean" not in table.columns
    assert table["area"].tolist() == [100]


def test_an_integer_image_can_skip_its_correction():
    image = (100 + np.random.default_rng(0).normal(0, 5, (FIELD, FIELD, 2))).astype(np.uint16)
    label = np.zeros((FIELD, FIELD), dtype=int)
    label[20:30, 20:30] = 1

    table = measure_features(
        image,
        label,
        features=["area", "intensity_mean"],
        channels=["first", "second"],
        normalisation_list=[_unfittable_normalisation()],
    )

    assert "first_mean" not in table.columns
    assert table["second_mean"].iloc[0] == pytest.approx(image[20:30, 20:30, 1].mean())


def test_measurement_uses_the_downsampling_of_the_correction(monkeypatch):
    received = {}

    def fake_correction(img, **kwargs):
        received.update(kwargs)
        return img

    monkeypatch.setattr("celldetective.measure.field_correction", fake_correction)
    label = np.zeros((FIELD, FIELD), dtype=int)
    label[20:30, 20:30] = 1

    measure_features(
        np.ones((FIELD, FIELD, 1)),
        label,
        features=["area"],
        channels=["first"],
        normalisation_list=[dict(_unfittable_normalisation(), downsample=3)],
    )

    assert received["downsample"] == 3


@pytest.mark.parametrize("target_channel", [None, "first"])
def test_a_cell_with_nothing_to_measure_keeps_the_shape_of_the_others(target_channel):
    # Unknown pixels, as a correction leaves them, entirely cover the first cell.
    image = np.ones((FIELD, FIELD, 1))
    image[:15, :15, 0] = np.nan
    label = np.zeros((FIELD, FIELD), dtype=int)
    label[5:10, 5:10] = 1
    label[20:30, 20:30] = 2

    if target_channel is None:

        def two_values(regionmask, intensity_image):
            return float(intensity_image[regionmask].mean()), 2.0

    else:

        def two_values(regionmask, intensity_image, target_channel=target_channel):
            return float(intensity_image[regionmask].mean()), 2.0

    table = regionprops_table(
        label,
        intensity_image=image,
        properties=["label"],
        extra_properties=(two_values,),
        channel_names=["first"],
    )

    values = np.array([v for k, v in table.items() if k != "label"])
    assert values.shape == (2, 2)
    assert np.isnan(values[:, 0]).all()
    assert values[:, 1].tolist() == [1.0, 2.0]


def test_skipped_channel_leaves_no_value_of_an_earlier_measurement():
    image = 100 + np.random.default_rng(0).normal(0, 5, (FIELD, FIELD, 2))
    label = np.zeros((FIELD, FIELD), dtype=int)
    label[20:30, 20:30] = 1
    norm = dict(
        correction_type="fit",
        target_channel="adhesion",
        threshold_on_std=0.0,
        operation="divide",
        model="paraboloid",
        clip=False,
    )
    channels = ["adhesion", "adhesion_channel"]
    table = measure_features(
        image.copy(),
        label,
        features=["intensity_mean"],
        channels=channels,
        normalisation_list=[norm],
    )
    earlier = pd.DataFrame(
        {
            "class_id": [1.0],
            "POSITION_X": [25.0],
            "adhesion_mean": [7.0],
            "adhesion_channel_mean": [8.0],
        }
    )

    positions = blank_skipped_channels(earlier, table, channels)
    merged = positions.merge(table, how="outer", on="class_id", suffixes=("_delme", ""))

    assert merged["adhesion_mean"].isna().all()
    assert merged["POSITION_X"].iloc[0] == 25.0
    measured = image[20:30, 20:30, 1].mean()
    assert merged["adhesion_channel_mean"].iloc[0] == pytest.approx(measured)


def test_an_empty_channel_never_reaches_a_custom_measurement():
    def fragile(regionmask, intensity_image):
        assert not np.isnan(intensity_image).all()
        return float(intensity_image[regionmask].mean()), 2.0

    image = np.ones((FIELD, FIELD, 2))
    image[:, :, 0] = np.nan
    label = np.zeros((FIELD, FIELD), dtype=int)
    label[20:30, 20:30] = 1

    table = regionprops_table(
        label,
        intensity_image=image,
        properties=["label"],
        extra_properties=(fragile,),
        channel_names=["first", "second"],
    )

    assert np.isnan(list(table.values())[1:]).sum() == 2
    assert sorted(v[0] for v in table.values() if np.isfinite(v[0])) == [1.0, 1.0, 2.0]


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


BLACK_LEVEL = 400


def test_black_level_gives_the_true_ratio():
    illumination, cells, image = _simulated_field()
    raw = image + BLACK_LEVEL
    bright = cells > 100
    truth = np.mean((illumination + cells)[bright] / illumination[bright])

    corrected = field_correction(raw, threshold=8, offset=BLACK_LEVEL)
    biased = field_correction(raw, threshold=8)

    assert corrected[bright].mean() == pytest.approx(truth, rel=0.02)
    assert biased[bright].mean() < truth * 0.9


def test_black_level_does_not_wrap_an_unsigned_image():
    image = _simulated_field()[2] + BLACK_LEVEL
    unsigned = np.clip(image, 0, None).astype(np.uint16)
    unsigned[100, 100] = BLACK_LEVEL - 100

    corrected = field_correction(unsigned, threshold=8, offset=BLACK_LEVEL)

    assert corrected[100, 100] < 0


def test_black_level_leaves_a_subtraction_unchanged():
    image = _simulated_field()[2]
    common = dict(threshold=8, operation="subtract")

    np.testing.assert_allclose(
        field_correction(image, offset=BLACK_LEVEL, **common),
        field_correction(image, **common),
    )


def test_background_is_returned_as_applied():
    image = _simulated_field()[2]

    _, background = field_correction(image, threshold=8, return_bg=True)
    _, applied = field_correction(image, threshold=8, return_bg=True, offset=BLACK_LEVEL)

    np.testing.assert_allclose(applied, background - BLACK_LEVEL)


def test_black_level_above_the_background_is_nan():
    image = _simulated_field()[2]

    corrected = field_correction(image, threshold=8, offset=10_000)

    assert np.isnan(corrected).all()


def test_fit_and_model_free_agree_with_the_same_black_level(tmp_path):
    exp_dir, movie_dirs = write_experiment(tmp_path, [5], frames=1)
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:FIELD, :FIELD]
    illumination = 300 + 100 * (1 - ((yy - 32) ** 2 + (xx - 32) ** 2) / 32**2)
    blob = lambda cy, cx: 600 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 3**2))
    places = [(22, 22), (22, 42), (42, 22), (42, 42), (32, 32)]
    for movie_dir, place in zip(movie_dirs, places):
        raw = BLACK_LEVEL + illumination + blob(*place) + rng.normal(0, 1, (FIELD, FIELD))
        write_movie(movie_dir / "sample.tif", raw[np.newaxis])
    common = dict(
        position_option=[0],
        target_channel="Channel1",
        threshold_on_std=8,
        operation="divide",
        offset=BLACK_LEVEL,
        return_stacks=True,
        **QUIET,
    )

    (fit,) = correct_background_model(exp_dir, downsample=1, **common)
    (model_free,) = correct_background_model_free(exp_dir, mode="tiles", **common)

    fit, model_free = np.squeeze(fit), np.squeeze(model_free)
    peak = illumination[22, 22] + 600
    assert fit[22, 22] == pytest.approx(peak / illumination[22, 22], rel=0.03)
    assert model_free[22, 22] == pytest.approx(fit[22, 22], rel=0.03)
    # Away from the cell, both are at 1 as the ratio should be.
    assert np.nanmedian(np.abs(fit[50:, 50:] - model_free[50:, 50:])) < 0.03


def test_measurement_uses_the_black_level():
    illumination, cells, image = _simulated_field()
    label = (cells > 100).astype(int)
    bright = label == 1
    norm = dict(
        correction_type="fit",
        target_channel="first",
        threshold_on_std=8,
        operation="divide",
        model="paraboloid",
        clip=False,
        offset=BLACK_LEVEL,
    )

    table = measure_features(
        (image + BLACK_LEVEL)[..., np.newaxis],
        label,
        features=["intensity_mean"],
        channels=["first"],
        normalisation_list=[norm],
    )

    truth = np.mean((illumination + cells)[bright] / illumination[bright])
    assert table["first_mean"].iloc[0] == pytest.approx(truth, rel=0.02)


def _cell_on_a_flat_background():
    """A cell at 300 over a background of 100, both above the black level."""
    image = np.full((FIELD, FIELD), BLACK_LEVEL + 100.0)
    label = np.zeros((FIELD, FIELD), dtype=int)
    label[20:30, 20:30] = 1
    image[label == 1] = BLACK_LEVEL + 300.0
    return image, label


def test_local_correction_takes_the_black_level():
    image, label = _cell_on_a_flat_background()

    corrected = normalise_by_cell(image, label, operation="divide", offset=BLACK_LEVEL)
    biased = normalise_by_cell(image, label, operation="divide")

    assert corrected[25, 25] == pytest.approx(3.0)
    assert biased[25, 25] == pytest.approx(700 / 500)


def test_local_correction_with_a_black_level_leaves_a_subtraction_unchanged():
    image, label = _cell_on_a_flat_background()

    with_offset = normalise_by_cell(image, label, offset=BLACK_LEVEL)

    assert with_offset[25, 25] == pytest.approx(200.0)


def test_local_black_level_above_the_background_is_nan():
    image, label = _cell_on_a_flat_background()

    corrected = normalise_by_cell(image, label, operation="divide", offset=10_000)

    assert np.isnan(corrected[25, 25])


def test_measurement_passes_the_black_level_to_the_local_correction():
    image, label = _cell_on_a_flat_background()
    norm = dict(
        correction_type="local",
        target_channel="first",
        distance=5,
        model="median",
        operation="divide",
        clip=False,
        offset=BLACK_LEVEL,
    )

    table = measure_features(
        image[..., np.newaxis],
        label,
        features=["intensity_mean"],
        channels=["first"],
        normalisation_list=[norm],
    )

    assert table["first_mean"].iloc[0] == pytest.approx(3.0)


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
