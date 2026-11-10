"""List the layer classes of a Keras .h5 model without building it."""
import collections
import json
import sys

import h5py


def layer_classes(node):
    found = collections.Counter()
    if isinstance(node, dict):
        if "class_name" in node and "config" in node:
            found[node["class_name"]] += 1
        for value in node.values():
            found += layer_classes(value)
    elif isinstance(node, list):
        for value in node:
            found += layer_classes(value)
    return found


for path in sys.argv[1:]:
    with h5py.File(path, "r") as f:
        config = f.attrs["model_config"]
    config = json.loads(config.decode() if isinstance(config, bytes) else config)
    classes = layer_classes(config)
    print(path.rsplit("/", 1)[-1], dict(classes))
    print("  Lambda present:", "Lambda" in classes)
