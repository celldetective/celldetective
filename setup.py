from setuptools import setup
import setuptools
from pathlib import Path

this_directory = Path(__file__).parent


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
    use_scm_version=True,
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
    # setuptools declares these globs in "/" form on every platform and
    # converts them itself, so spell them literally rather than with os.sep.
    # (The os.sep form happened to produce identical wheels -- "/" on POSIX,
    # and Windows globbing tolerates "\" -- so this is clarity, not a fix.)
    package_data={
        "celldetective": [
            "*",
            "scripts/*",
            "gui/*",
            "gui/icons/plots/*",
            "gui/icons/stats/*",
            "regionprops/*",
            "gui/processes/*",
            "gui/help/*",
            "models/*/*/*",
            "models/*",
            "models/*/*",
            "icons/*",
            "links/*",
            "datasets/*",
            "datasets/*/*",
        ]
    },
    entry_points={
        "console_scripts": ["celldetective = celldetective.__main__:main"],
    },
    install_requires=requirements,
    extras_require={
        "tensorflow": ["tensorflow~=2.15.0", "stardist"],
        # "process": ["cellpose<3", "stardist", "tensorflow~=2.15.0"],
        "all": ["cellpose<3", "stardist", "tensorflow~=2.15.0"],
    },
    # dependency_links = links
)
