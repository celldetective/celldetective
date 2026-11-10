"""Read the scores stored at training without running anything in the pickle.

Numpy array types are restored; every other global becomes an inert placeholder.
"""
import pickle
import sys

import numpy as np


class Placeholder:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass


class Inert(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] == "numpy" and name in ("ndarray", "dtype", "_reconstruct", "scalar"):
            return super().find_class(module, name)
        return Placeholder


def show(key, value):
    if isinstance(value, (int, float, np.floating, np.integer)):
        print(f"  {key}: {float(value):.4f}")
    elif isinstance(value, np.ndarray) and value.size <= 16:
        print(f"  {key}: {value.tolist()}")
    elif isinstance(value, (list, tuple)):
        print(f"  {key}: list of {len(value)}, last {value[-1] if len(value) else None}")
    elif isinstance(value, dict):
        print(f"  {key}: dict {list(value)[:12]}")
    else:
        print(f"  {key}: {type(value).__name__} {getattr(value, 'shape', '')}")


for path in sys.argv[1:]:
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        reader(f)
        data = Inert(f, encoding="latin1").load()
    data = data.item() if isinstance(data, np.ndarray) and data.shape == () else data
    print(path.split("/")[-2])
    for key, value in data.items():
        show(key, value)
