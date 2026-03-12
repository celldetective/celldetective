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
def stop_leaked_stack_loaders():
    """
    Stop any StackLoader QThreads that survive test teardown.

    test_project.py creates AppInitWindow instances with StackVisualizer widgets
    that spawn StackLoader background threads.  If those threads are still alive
    when a later test (e.g. test_settings_tracking) runs, they may try to access
    destroyed C++ Qt objects and trigger a fatal access-violation on Windows.

    This fixture runs *after* every test and defensively stops any leaked loaders.
    """
    yield
    # Import here so non-GUI tests don't pay the import cost at collection time.
    try:
        from celldetective.gui.viewers.base_viewer import StackLoader
    except Exception:
        return

    for obj in gc.get_objects():
        try:
            if isinstance(obj, StackLoader) and obj.isRunning():
                try:
                    obj.frame_loaded.disconnect()
                except Exception:
                    pass
                obj.stop()
                obj.wait(2000)
        except (ReferenceError, TypeError):
            # Object may have been collected between iteration and access
            pass
