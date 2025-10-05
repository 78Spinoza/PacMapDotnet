# PACMAP __init__.py Reference
# Downloaded from: https://github.com/YingfanWang/PaCMAP/blob/master/source/pacmap/__init__.py

"""
PaCMAP package initialization

Reference implementation for understanding the PACMAP package structure.
"""

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

from .pacmap import PaCMAP, sample_neighbors_pair, LocalMAP

__version__ = version("pacmap")

__all__ = ["PaCMAP", "sample_neighbors_pair", "LocalMAP"]