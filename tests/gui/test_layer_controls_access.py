"""
Reaching a layer's Qt controls must not go through deprecated napari API.

``Window.qt_viewer`` warns on every viewer opened in napari 0.5 and is gone in
0.6. The private attribute it wrapped is what remains, so that is tried first,
with the public one kept only for a napari old enough to lack it.
"""

import logging

import pytest

from celldetective.napari import utils as napari_utils


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable all logging to avoid Windows OSError with pytest capture."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


class _QtViewer:
    def __init__(self, widgets):
        self.controls = type("Controls", (), {"widgets": widgets})()


class _Window:
    """A napari window exposing whichever of the two attributes is asked for."""

    def __init__(self, private=None, public=None):
        if private is not None:
            self._qt_viewer = private
        if public is not None:
            self.qt_viewer = public


class _Viewer:
    def __init__(self, window):
        self.window = window


class TestLayerControls:

    def test_the_private_attribute_is_preferred(self):
        deprecated = _QtViewer({"layer": "public"})
        viewer = _Viewer(
            _Window(private=_QtViewer({"layer": "private"}), public=deprecated)
        )
        assert napari_utils._layer_controls(viewer, "layer") == "private"

    def test_the_public_attribute_is_the_fallback(self):
        viewer = _Viewer(_Window(public=_QtViewer({"layer": "public"})))
        assert napari_utils._layer_controls(viewer, "layer") == "public"

    def test_neither_available_is_not_an_error(self):
        assert napari_utils._layer_controls(_Viewer(_Window()), "layer") is None

    def test_an_unknown_layer_falls_through_to_none(self):
        viewer = _Viewer(_Window(private=_QtViewer({})))
        assert napari_utils._layer_controls(viewer, "layer") is None
