"""
IGS Module

Names and analysis centers for the International GNSS Service (IGS) precise
products. IGS analysis centers publish orbit, clock, and Earth-rotation
products on public archives under two file-name conventions: long names
(``IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz``) from GPS week 2238 and short
names (``igs21000.sp3.Z``) before it.

This module provides:
- IGSProductFileName: Parser for long and short product file names
- IGSSolutionType, IGSContentType, IGSFileFormat, IGSCompression, IGSNameStyle: Typed fields of a product file name
- IGSAnalysisCenter: Analysis centers that publish the products

Example:
    ```python
    import brahe as bh

    name = bh.igs.IGSProductFileName.parse("COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz")
    print(name.center, name.solution, name.gps_week, name.is_sp3)
    print(name.center.name)
    ```
"""

from brahe._brahe import (
    IGSAnalysisCenter,
    IGSCompression,
    IGSContentType,
    IGSFileFormat,
    IGSNameStyle,
    IGSProductFileName,
    IGSSolutionType,
)

__all__ = [
    "IGSAnalysisCenter",
    "IGSCompression",
    "IGSContentType",
    "IGSFileFormat",
    "IGSNameStyle",
    "IGSProductFileName",
    "IGSSolutionType",
]
