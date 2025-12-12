"""Data Analysis API package

Expose the main functions for creating virtual datasets, computing scan
averages and computing fits.

This file performs explicit imports so IDEs (Pyright/Pylance, Jedi)
can discover symbols for autocompletion and signature help.
"""
from .api import (
    create_virtual_file,
    compute_scan_avg,
    compute_model,
    mask_data,
)

__all__ = [
    'create_virtual_file',
    'compute_scan_avg',
    'compute_model',
    'mask_data',
]

__version__ = "0.0.0"
