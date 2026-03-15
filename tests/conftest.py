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
    # Import here so non-GUI tests don't pay the import cost at collection time.
    try:
        from PyQt5.QtCore import QThread
    except Exception:
        return

    for obj in gc.get_objects():
        try:
            if isinstance(obj, QThread) and obj.isRunning():
                # Only mess with threads from our own codebase
                mod_name = getattr(type(obj), '__module__', '')
                if mod_name.startswith('celldetective.'):
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
                    
                    obj.wait(1000)
                    
                    if obj.isRunning():
                        try:
                            obj.terminate()
                            obj.wait(500)
                        except Exception:
                            pass
        except (ReferenceError, TypeError):
            # Object may have been collected between iteration and access
            pass
