#!/usr/bin/env python3
"""Local wrapper for the thesis hierarchy utilities.

This keeps the main training/preprocessing entrypoints self-contained under
``code/`` while reusing the maintained SC2EGSet hierarchy implementation stored
in ``compare/sc2egset_hierarchy_version1.py``.
"""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_SOURCE_PATH = (
    Path(__file__).resolve().parent.parent / "compare" / "sc2egset_hierarchy_version1.py"
)
_SPEC = spec_from_file_location("_thesis_hierarchy_impl", _SOURCE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load hierarchy implementation from {_SOURCE_PATH}")
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


for _name in dir(_MODULE):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_MODULE, _name)


__all__ = [name for name in globals() if not name.startswith("_")]
