"""
Primitive type aliases and literals for the Biomarker Service.

This module defines reusable type hints for the domain.
"""

from typing import Literal

# Eye side types
EyeSide = Literal["left", "right", "both"]

# Biomarker status types
BiomarkerStatus = Literal["COMPLETED", "FAILED"]

# Error code types
BiomarkerErrorCode = Literal[
    "BIOMARKER_TIMEOUT",
    "BIOMARKER_HTTP_ERROR",
    "BIOMARKER_CIRCUIT_OPEN",
    "BIOMARKER_ERROR",
    "BIOMARKER_SCAN_NOT_FOUND",
]
