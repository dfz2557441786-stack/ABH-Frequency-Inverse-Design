"""Activate the organized project modules without hard-coded external paths."""
from __future__ import print_function

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODULE_DIRECTORIES = (
    "00_USER_DESIGN",
    "01_CORE",
    os.path.join("02_ABAQUS", "geometry"),
    os.path.join("02_ABAQUS", "assembly"),
    os.path.join("02_ABAQUS", "analysis"),
    "03_DATABASE",
    "04_INVERSE_DESIGN",
    "05_VALIDATION",
    "06_POSTPROCESS",
)


def activate():
    """Make active modules importable while keeping archive folders excluded."""
    for relative in reversed(MODULE_DIRECTORIES):
        directory = os.path.join(PROJECT_ROOT, relative)
        if os.path.isdir(directory) and directory not in sys.path:
            sys.path.insert(0, directory)
    return PROJECT_ROOT

