"""
Mongol ML Autocomplete - A library for Mongolian text autocompletion using PyTorch models.

This package provides the MongolMLAutocomplete class for generating word completions
based on a trained PyTorch model.
"""

from .autocomplete import MongolMLAutocomplete
from . import font_utils

__version__ = "0.1.0"
__all__ = ["MongolMLAutocomplete", "font_utils"]
