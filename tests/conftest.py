import gc

import pytest
from tqdm import tqdm


@pytest.fixture(autouse=True)
def disable_tqdm_monitor():
    """
    Disable tqdm monitor thread to avoid 'Windows fatal exception: access violation'
    during Qt tests that use processEvents.
    """
    original_monitor_interval = tqdm.monitor_interval
    tqdm.monitor_interval = 0
    yield
    tqdm.monitor_interval = original_monitor_interval


@pytest.fixture(autouse=True)
def stop_leaked_threads():
    """
    Stop any celldetective QThreads that survive test teardown.

    Leaked background threads (e.g. StackLoader, BackgroundLoader) may try to access
    destroyed C++ Qt objects and trigger a fatal access-violation on Windows, or
    fail to exit and hang the CI pipeline.

    This fixture runs *after* every test and defensively stops any leaked threads.
    """
    yield

    # Drain the Qt event queue before scanning for threads.
    #
    # All CelldetectiveWidget / CelldetectiveMainWindow instances carry
    # WA_DeleteOnClose, so widget.close() only *schedules* deletion via
    # deleteLater() — the DeferredDelete event stays in the queue until the
    # event loop runs.  Child-widget events (paint, resize, QLabeledSlider
    # internal-label timers, …) remain queued *behind* the DeferredDelete.
    # If processEvents() is called later (e.g. in _build_layouts() of the
    # next test's widget __init__), Qt fires the DeferredDelete, frees the
    # C++ object, then dispatches those trailing child events to freed
    # memory → "Windows fatal exception: access violation".
    #
    # Processing events here, while the previous test's fixtures have already
    # run their teardown (LIFO order means test fixtures tear down before
    # conftest fixtures), lets deleteLater() complete and Qt remove all
    # remaining events for the deleted objects — leaving a clean queue for
    # the next test.
    try:
        from PyQt5.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.processEvents()
    except Exception:
        pass
    # Import here so non-GUI tests don't pay the import cost at collection time.
    try:
        from PyQt5.QtCore import QThread
    except Exception:
        return

    def _stop_running_celldetective_threads():
        """Stop + join every running QThread defined in our codebase.

        Returns True if at least one such thread was found running, so the
        caller can decide whether another sweep is worthwhile.
        """
        found = False
        for obj in gc.get_objects():
            try:
                if isinstance(obj, QThread) and obj.isRunning():
                    # Only mess with threads from our own codebase
                    mod_name = getattr(type(obj), '__module__', '')
                    if mod_name.startswith('celldetective.'):
                        found = True
                        try:
                            if hasattr(obj, 'frame_loaded'):
                                obj.frame_loaded.disconnect()
                        except Exception:
                            pass

                        try:
                            if hasattr(obj, 'stop'):
                                obj.stop()
                        except Exception:
                            pass

                        try:
                            obj.quit()
                        except Exception:
                            pass

                        obj.wait(2000)

                        # NOTE: Do NOT call terminate() on Windows.
                        # QThread::terminate() calls TerminateThread() which can
                        # corrupt the process heap while the thread is mid-import
                        # or inside a memory allocator, causing access violations
                        # in ANY thread (including the main thread) during the
                        # next test's event processing.
            except (ReferenceError, TypeError):
                # Object may have been collected between iteration and access
                pass
        return found

    # Sweep 1: stop + join any thread that is currently running and reachable.
    _stop_running_celldetective_threads()

    # Force collection of the now-stopped thread wrappers *here*, while we know
    # none of our QThreads is running. A StackLoader's QMutex / QWaitCondition
    # are C++ objects owned by the Python wrapper; sip frees them the moment the
    # wrapper is collected. If that collection were left to happen later — e.g.
    # mid-QApplication.processEvents() inside the next test's widget __init__ —
    # the C++ mutex/condition could be freed while a still-running run() loop is
    # parked in condition.wait(), producing the "Windows fatal exception: access
    # violation" at base_viewer.py:run. Collecting now makes the reclaim
    # deterministic and, because the threads are already joined, safe.
    try:
        gc.collect()
    except Exception:
        pass

    # Sweep 2: gc.collect() can break reference cycles and surface a thread that
    # only became reachable/running-detectable after collection; stop it too.
    if _stop_running_celldetective_threads():
        try:
            gc.collect()
        except Exception:
            pass

    # Final drain so any DeferredDelete scheduled by the collection above is
    # dispatched against still-live objects rather than the next test's.
    try:
        from PyQt5.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.processEvents()
    except Exception:
        pass
