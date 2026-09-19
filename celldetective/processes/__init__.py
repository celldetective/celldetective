class PositionSkipped(Exception):
    """
    Raised by a batch worker when a position cannot be processed.

    The workers run inside ``UnifiedBatchProcess`` as plain objects (they are
    never started as processes), so aborting means skipping the current
    position rather than terminating anything.
    """
