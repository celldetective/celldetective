"""
A running QThread must not be finalized under itself.

Every loader in the GUI is built the same way: an unparented ``QThread`` whose
only reference is an attribute of the widget that started it. When that widget
goes away without its ``closeEvent`` completing -- a window with
``WA_DeleteOnClose``, a test that drops its reference, an exception on the way
through teardown -- the interpreter is free to collect the wrapper while
``run()`` is still going. sip then frees the C++ ``QThread`` along with any
``QMutex`` / ``QWaitCondition`` it owns, and the running thread walks into freed
memory: on Windows a bare ``access violation`` that kills the interpreter with
no traceback, attributed to whatever happened to be running at the time.

That failure cannot be asserted on directly -- a process that aborts reports
nothing -- so these tests pin the invariant that prevents it instead: while a
thread is running, something other than its owner holds a reference to it.
"""

import gc
import unittest

from PyQt5.QtCore import QMutex, QThread, QWaitCondition

from celldetective.gui.base.threads import (
    _LIVE_THREADS,
    start_tracked,
    stop_all_tracked_threads,
    stop_thread,
    track_thread,
)


class _ParkedThread(QThread):
    """
    A thread shaped like the real loaders: it parks on a condition it owns.

    The C++ ``QMutex`` and ``QWaitCondition`` belong to this wrapper, so they are
    exactly what gets freed underneath ``run()`` if the wrapper is collected
    while the thread is still parked.
    """

    def __init__(self):
        super().__init__()
        self.running = True
        self.mutex = QMutex()
        self.condition = QWaitCondition()

    def stop(self):
        self.mutex.lock()
        self.running = False
        self.condition.wakeAll()
        self.mutex.unlock()

    def run(self):
        while True:
            self.mutex.lock()
            if not self.running:
                self.mutex.unlock()
                return
            self.condition.wait(self.mutex, 50)
            self.mutex.unlock()


class TestThreadLifetime(unittest.TestCase):

    def tearDown(self):
        stop_all_tracked_threads(timeout=2000)

    def test_a_started_thread_outlives_its_only_other_reference(self):
        """
        The invariant. Dropping the owner must not make the thread collectable.
        """

        owner = {"thread": start_tracked(_ParkedThread())}
        thread = owner["thread"]
        self.assertTrue(thread.isRunning())

        # The widget goes away without ever closing -- the case `closeEvent`
        # cannot cover.
        owner.clear()
        gc.collect()

        self.assertIn(thread, _LIVE_THREADS)
        self.assertTrue(thread.isRunning())

        self.assertTrue(stop_thread(thread, timeout=5000))

    def test_a_finished_thread_stops_being_held(self):
        """
        The registry must not turn into a leak: it is bounded by what is running.
        """

        thread = start_tracked(_ParkedThread())
        self.assertTrue(stop_thread(thread, timeout=5000))
        self.assertNotIn(thread, _LIVE_THREADS)

    def test_stopping_is_idempotent_and_survives_a_dead_thread(self):
        """
        Teardown reaches a thread from several directions; the second caller
        must not be the one that raises.
        """

        thread = start_tracked(_ParkedThread())
        self.assertTrue(stop_thread(thread, timeout=5000))
        self.assertTrue(stop_thread(thread, timeout=5000))
        self.assertTrue(stop_thread(None))

    def test_a_never_started_thread_is_accepted(self):
        """A widget may tear down before it ever got as far as starting."""

        self.assertTrue(stop_thread(track_thread(_ParkedThread())))

    def test_stop_all_clears_everything_running(self):
        threads = [start_tracked(_ParkedThread()) for _ in range(3)]

        stop_all_tracked_threads(timeout=5000)

        for thread in threads:
            self.assertFalse(thread.isRunning())
            self.assertNotIn(thread, _LIVE_THREADS)


class TestStackVisualizerReleasesItsLoader(unittest.TestCase):
    """The real widget, not just the pattern."""

    def test_the_loader_is_tracked_while_it_runs(self):
        from celldetective.gui.viewers import base_viewer

        loader = base_viewer.StackLoader.__new__(base_viewer.StackLoader)
        QThread.__init__(loader)
        loader.stack_path = None
        loader.img_num_per_channel = None
        loader.n_channels = 1
        loader.target_channel = 0
        loader.priority_frame = 0
        loader.cache_keys = set()
        loader.running = False  # return immediately; we only pin the tracking
        loader.mutex = QMutex()
        loader.condition = QWaitCondition()

        start_tracked(loader)
        self.assertTrue(stop_thread(loader, timeout=5000))
        self.assertNotIn(loader, _LIVE_THREADS)


if __name__ == "__main__":
    unittest.main()
