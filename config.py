"""Compatibility import for the active configuration in ``01_CORE``.

New code imports ``config`` after activating ``project_bootstrap``.  This thin
shim keeps older explicit scripts importable while all configuration ownership
remains in `01_CORE/config.py`.
"""
from __future__ import print_function

import importlib.util
import os
import sys


_ROOT = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.join(_ROOT, "01_CORE")
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)
_SOURCE = os.path.join(_CORE, "config.py")
_SPEC = importlib.util.spec_from_file_location("_rabh_active_config", _SOURCE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
for _NAME in dir(_MODULE):
    if not _NAME.startswith("_"):
        globals()[_NAME] = getattr(_MODULE, _NAME)
