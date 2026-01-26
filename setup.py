from setuptools import setup
import setuptools
import os
import re
from pathlib import Path

this_directory = Path(__file__).parent

# Load version
VERSIONFILE = "celldetective/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


def load_requirements(path):
    requirements = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            # Remove comments
            if "#" in line:
                line = line[: line.index("#")].strip()

            # Skip empty lines
            if not line:
                continue

            requirements.append(line)
    return requirements


requirements = load_requirements("requirements.txt")

setup(
    name="celldetective",
    version=verstr,
    description="description",
    long_description=(this_directory / "README.md").read_text(),
    # long_description=open('README.rst',encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="http://github.com/remyeltorro/celldetective",
    author="Rémy Torro",
    author_email="remy.torro@inserm.fr",
    license="GPL-3.0",
    packages=setuptools.find_packages(),
    zip_safe=False,
    package_data={
        "celldetective": [
            "*",
            os.sep.join(["scripts", "*"]),
            os.sep.join(["gui", "*"]),
            os.sep.join(["regionprops", "*"]),
            os.sep.join(["gui", "processes", "*"]),
            os.sep.join(["gui", "help", "*"]),
            os.sep.join(["models", "*", "*", "*"]),
            os.sep.join(["models", "*"]),
            os.sep.join(["models", "*", "*"]),
            os.sep.join(["icons", "*"]),
            os.sep.join(["links", "*"]),
            os.sep.join(["datasets", "*"]),
            os.sep.join(["datasets", "*", "*"]),
        ]
    },
    entry_points={
        "console_scripts": ["celldetective = celldetective.__main__:main"],
    },
    install_requires=requirements,
    extras_require={
        "tensorflow": ["tensorflow~=2.15.0", "stardist"],
        #"process": ["cellpose<3", "stardist", "tensorflow~=2.15.0"],
        "all": ["cellpose<3", "stardist", "tensorflow~=2.15.0"],
    },
    # dependency_links = links
)
