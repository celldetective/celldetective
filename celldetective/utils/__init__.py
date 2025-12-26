from .common import *
import importlib
import sys

# Define submodules that contain the lazy-loaded functions
_SUBMODULES = ["generic", "experiments"]


def __getattr__(name):
    # Avoid recursion if asking for the submodules themselves
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)

    # Try to find the attribute in the submodules
    for module_name in _SUBMODULES:
        try:
            # Import the module safely
            mod = importlib.import_module(f".{module_name}", __name__)

            # If the attribute exists in the module, return it
            if hasattr(mod, name):
                return getattr(mod, name)
        except ImportError as e:
            # If for some reason the submodule cannot be imported, ignore it here
            # or log it if necessary, but don't crash yet
            continue
        except RecursionError:
            # Explicitly catch recursion during import attempts
            continue

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
