"""
Keeping a running ``QThread`` alive until it has actually stopped.

A ``QThread`` whose C++ object is destroyed while the thread is still running
takes the whole process down -- on Windows as a bare ``access violation``, with
no Python traceback and no failing assertion, so whatever happens to run next is
what appears to have crashed. The threads in this package are all built the same
way::

    self.loader = SomeLoader(...)
    self.loader.start()

with no parent, so the only thing keeping the wrapper alive is the widget
attribute. A widget that goes away without its ``closeEvent`` finishing -- a
window with ``WA_DeleteOnClose``, a test that simply drops its reference, an
exception on the way through teardown -- lets the interpreter finalize a thread
that is still inside ``run()``.

This module removes the ``start()``/attribute pattern's sharp edge: a tracked
thread holds a strong reference of its own until ``finished`` fires, so it
cannot be collected mid-run no matter what happens to its owner, and
:func:`stop_thread` gives every teardown path the same stop-then-join sequence.
"""

from typing import Any, Iterable, Optional

from PyQt5.QtCore import Qt, QThread

from celldetective import get_logger

logger = get_logger(__name__)

#: Strong references to threads that are running right now. A thread adds itself
#: on :func:`track_thread` and drops out when it finishes, so this stays bounded
#: by the number of threads actually in flight rather than growing over a
#: session.
_LIVE_THREADS = set()


def track_thread(thread: QThread) -> QThread:
    """
    Keep `thread` from being collected while it runs.

    Parameters
    ----------
    thread : QThread
        The thread to hold on to. It does not have to be running yet.

    Returns
    -------
    QThread
        The same thread, so this can wrap a constructor call.
    """

    _LIVE_THREADS.add(thread)
    # DirectConnection on purpose. A QThread object lives in the thread that
    # created it, so the default AutoConnection would queue this onto the main
    # thread and only release the reference once someone spins the event loop --
    # which a worker that ends on its own cannot count on, and which no test
    # does. Direct runs it in the worker as it exits; `discard` on a set is a
    # single interpreter-level operation and the thread only removes itself.
    thread.finished.connect(
        lambda t=thread: _LIVE_THREADS.discard(t), Qt.DirectConnection
    )
    return thread


def start_tracked(thread: QThread) -> QThread:
    """
    Start `thread` and keep it alive for as long as it runs.

    The replacement for a bare ``thread.start()`` on a thread whose only other
    reference is an attribute of the widget that made it.

    Parameters
    ----------
    thread : QThread
        The thread to start.

    Returns
    -------
    QThread
        The same thread.
    """

    track_thread(thread)
    thread.start()
    return thread


def stop_thread(
    thread: Optional[QThread],
    timeout: int = 5000,
    signals: Iterable[Any] = (),
) -> bool:
    """
    Ask a thread to stop and wait for it to actually be gone.

    Safe to call on a thread that has already finished, on one whose C++ object
    has been deleted, and more than once -- teardown reaches this from several
    directions (``closeEvent``, the widget's ``destroyed`` signal, application
    exit) and the second caller must not be the one that raises.

    ``terminate()`` is deliberately never called: it leaves the thread's mutexes
    locked and produces exactly the access violation this module exists to
    prevent.

    Parameters
    ----------
    thread : QThread or None
        The thread to stop. None is accepted and does nothing, so a caller does
        not have to check an attribute that may never have been set.
    timeout : int, optional
        How long to wait for the thread to finish, in milliseconds. Default is
        5000.
    signals : iterable, optional
        Signals to disconnect before stopping, so a queued emission cannot be
        delivered to a widget that is being torn down.

    Returns
    -------
    bool
        True when the thread is no longer running by the time this returns.
    """

    if thread is None:
        return True

    for signal in signals:
        try:
            signal.disconnect()
        except (TypeError, RuntimeError) as e:
            # Nothing was connected, or the C++ object is already gone.
            logger.debug(f"Could not disconnect a thread signal: {e}")

    try:
        if not thread.isRunning():
            _LIVE_THREADS.discard(thread)
            return True
    except RuntimeError:
        # The wrapper outlived its C++ object; there is nothing left to stop.
        _LIVE_THREADS.discard(thread)
        return True

    # `stop` is this package's own convention for "leave your loop"; `quit` ends
    # the event loop of a thread that runs one. Whichever the thread uses, the
    # other is harmless.
    stop = getattr(thread, "stop", None)
    if callable(stop):
        try:
            stop()
        except RuntimeError as e:
            logger.debug(f"Could not signal a thread to stop: {e}")

    try:
        thread.quit()
        finished = thread.wait(timeout)
    except RuntimeError as e:
        logger.debug(f"Thread went away while being stopped: {e}")
        return True

    if finished:
        # Released here rather than waiting for the `finished` handler: a thread
        # that was never started emits nothing, and a caller that has just
        # joined one is entitled to see it gone.
        _LIVE_THREADS.discard(thread)
    else:
        # Left running on purpose: it still holds a strong reference through
        # `_LIVE_THREADS`, so it will not be finalized under itself, and it will
        # drop out on its own once it returns.
        logger.warning(
            f"{type(thread).__name__} did not stop within {timeout} ms; "
            "leaving it to finish in the background."
        )
    return finished


def stop_all_tracked_threads(timeout: int = 2000) -> None:
    """
    Stop every tracked thread that is still running.

    For the end of the process, and for a test suite that wants to start each
    case without threads left over from the last one.

    Parameters
    ----------
    timeout : int, optional
        How long to wait for each thread, in milliseconds. Default is 2000.
    """

    for thread in list(_LIVE_THREADS):
        stop_thread(thread, timeout=timeout)
        _LIVE_THREADS.discard(thread)
