"""
Modified ITC Module

Reads and writes the Space-Track Modified ITC ephemeris format used for
conjunction screening submissions and for Starlink's public ephemerides.

This module provides:
- ITC: A parsed message with SI-unit records and optional 6x6 covariance
- ITCHeader, ITCStateVector, ITCCovarianceFrame: Message components
- state_frame_for_data_type, data_type_for_state_frame: File-name DataType mapping

Example:
    ```python
    import brahe as bh

    itc = bh.ITC.from_file("MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt")
    print(len(itc), itc.header.state_frame, itc.has_covariance)
    ```
"""

from brahe._brahe import (
    ITC,
    ITCCovarianceFrame,
    ITCHeader,
    ITCStateVector,
    data_type_for_state_frame,
    state_frame_for_data_type,
)

__all__ = [
    "ITC",
    "ITCCovarianceFrame",
    "ITCHeader",
    "ITCStateVector",
    "data_type_for_state_frame",
    "state_frame_for_data_type",
]
