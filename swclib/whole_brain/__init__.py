"""Whole-brain image readers with a common region interface."""

from .reader import WholeBrainReader

__all__ = ["FNTReader", "WBTReader", "WholeBrainReader"]


def __getattr__(name):
    # Keep slice-only use independent of the NRRD backend, and vice versa.
    if name == "FNTReader":
        from .fntreader import FNTReader
        return FNTReader
    if name == "WBTReader":
        from .tifreader import WBTReader
        return WBTReader
    raise AttributeError(name)
