import os
from pathlib import Path
from typing import Union


def remove_file_if_exists(file: Union[str, Path]):
    if os.path.exists(file):
        try:
            os.remove(file)
        except Exception as e:
            print(e)
