import os
import sys

from setuptools import find_packages, setup

# Minimum required python version
REQUIRED_MAJOR = 3
REQUIRED_MINOR = 10

# Requirements for testing, formatting, and tutorials
REQUIRES = [
    "numpy",
    "torch",  # TODO: version?
]
TEST_REQUIRES = ["pytest", "pytest-cov"]
FMT_REQUIRES = ["flake8", "ufmt", "flake8-docstrings"]

# Check for python version
if sys.version_info < (REQUIRED_MAJOR, REQUIRED_MINOR):
    error = (
        "Your version of python ({major}.{minor}) is too old. You need "
        "python >= {required_major}.{required_minor}."
    ).format(
        major=sys.version_info.major,
        minor=sys.version_info.minor,
        required_minor=REQUIRED_MINOR,
        required_major=REQUIRED_MAJOR,
    )
    sys.exit(error)

# Assign root dir location for later use
root_dir = os.path.dirname(__file__)


def read_deps_from_file(filname):
    """Read in requirements file and return items as list of strings"""
    with open(os.path.join(root_dir, filname), "r") as fh:
        return [line.strip() for line in fh.readlines() if not line.startswith("#")]


# Read in pinned versions of the formatting tools
FMT_REQUIRES += read_deps_from_file("requirements-fmt.txt")
# Dev is test + formatting
DEV_REQUIRES = TEST_REQUIRES + FMT_REQUIRES

setup(
    name="GPyTorch mini",
    description="Minimal implementation of Exact GPs, modified from GPyTorch",
    author="Sait Cakmak",
    license="MIT",
    python_requires=f">={REQUIRED_MAJOR}.{REQUIRED_MINOR}",
    packages=find_packages(exclude=["test", "test.*"]),
    install_requires=REQUIRES,
    extras_require={
        "dev": DEV_REQUIRES,
        "test": TEST_REQUIRES,
    },
)
