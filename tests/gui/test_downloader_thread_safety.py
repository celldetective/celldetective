"""
The downloaders must not build widgets off the GUI thread.

A model is fetched as part of being prepared, and the napari single-frame panel
prepares its model on a worker thread. Both downloaders used to open a progress
widget unconditionally -- one of them running a modal ``exec_()`` -- which from
a worker thread freezes the whole interface instead of raising. They now fall
back to the console path whenever they are not on the application's thread.
"""

import logging

import pytest
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication

from celldetective.utils import downloaders


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable all logging to avoid Windows OSError with pytest capture."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


class _Probe(QThread):
    """Run `func` on a worker thread and keep whatever it raised or returned."""

    def __init__(self, func):
        super().__init__()
        self._func = func
        self.result = None
        self.error = None

    def run(self):
        try:
            self.result = self._func()
        except BaseException as e:  # noqa: BLE001 - reported back to the test
            self.error = e


def _run_off_gui_thread(func, qapp):
    """Run `func` on a worker thread, pumping the GUI thread until it ends."""
    probe = _Probe(func)
    probe.start()
    # A plain `wait()` would block the GUI thread, so a regression that queues
    # work onto it would deadlock the test run rather than fail it.
    for _ in range(2000):
        if probe.wait(10):
            break
        qapp.processEvents()
    else:  # pragma: no cover - only on a regression
        probe.terminate()
        probe.wait()
        pytest.fail("The download did not return off the GUI thread.")
    if probe.error is not None:
        raise probe.error
    return probe.result


class TestZenodoDownloadOffTheGuiThread:
    """``download_zenodo_file`` must not run its modal dialog from a worker."""

    def test_the_gui_downloader_is_skipped(self, qapp, monkeypatch, tmp_path):
        def _no_widgets(*args, **kwargs):
            raise AssertionError("a widget was built off the GUI thread")

        monkeypatch.setattr(
            "celldetective.gui.workers.GenericProgressWindow", _no_widgets
        )
        # Stop at the first step of the console path; getting there is the point.
        monkeypatch.setattr(
            downloaders, "open", _raising_open("console path reached"), raising=False
        )

        with pytest.raises(RuntimeError, match="console path reached"):
            _run_off_gui_thread(
                lambda: downloaders.download_zenodo_file("a_model", str(tmp_path)),
                qapp,
            )

    def test_the_gui_downloader_is_used_on_the_gui_thread(self, qapp, monkeypatch, tmp_path):
        used = {}

        class _Window:
            def __init__(self, *args, **kwargs):
                used["built"] = True

            def exec_(self):
                return 0  # rejected: returns without touching the console path

        monkeypatch.setattr(
            "celldetective.gui.workers.GenericProgressWindow", _Window
        )
        assert QThread.currentThread() is qapp.thread()

        downloaders.download_zenodo_file("a_model", str(tmp_path))

        assert used.get("built") is True


def _raising_open(message):
    """An ``open`` that always raises, marking a point in the console path."""

    def _open(*args, **kwargs):
        raise RuntimeError(message)

    return _open
