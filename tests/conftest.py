import pytest
from tqdm import tqdm
import weakref


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
def safe_close_stack_visualizers(request):
    """
    Automatically track and safely close all StackVisualizer instances created during a test
    to prevent threaded background loader crashes on teardown.
    """
    from celldetective.gui.viewers.base_viewer import StackVisualizer

    # Track instances created in this test
    instances = set()

    # Store original init
    orig_init = StackVisualizer.__init__

    def tracked_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        # Store a weak reference to not prevent GC if it's naturally collected
        instances.add(weakref.ref(self))

    StackVisualizer.__init__ = tracked_init

    yield

    # Restore original init
    StackVisualizer.__init__ = orig_init

    # Safely close any remaining instances
    for ref in instances:
        instance = ref()
        if instance is not None:
            try:
                if hasattr(instance, "loader_thread") and instance.loader_thread:
                    instance.loader_thread.stop()
                    instance.loader_thread.wait(1000)
                    instance.loader_thread = None
                if hasattr(instance, "frame_cache"):
                    instance.frame_cache.clear()

                instance.close()
                instance.deleteLater()
            except RuntimeError:
                pass  # C++ object deleted
