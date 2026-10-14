"""
Top-level convenience wrapper for the MORONI cellular texture engine.
The canonical implementation resides in src/moroni.py.
"""
import sys
from src import moroni as _impl

for _attr in dir(_impl):
    if not _attr.startswith("__"):
        globals()[_attr] = getattr(_impl, _attr)

__all__ = _impl.__all__

