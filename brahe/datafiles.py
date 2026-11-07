"""
Datafiles Module

Readers and writers for scientific data file formats.

This module provides:
- SP3: Precise orbit file (SP3-a through SP3-d) with parsing, construction, and SP3-d writing
- SP3Header, SP3Record, SP3Correlation, SP3SatelliteId: Components of an SP3 file
- SP3Satellite: Per-satellite samples, interpolated states, and trajectories from one or more SP3 files
- SP3_DEFAULT_LAGRANGE_DEGREE: Default Lagrange degree of SP3Satellite interpolation and velocity derivation
- SP3Version, SP3FileType, SP3TimeSystem, SP3OrbitType, GNSSSystem: Header and identifier enumerations

Example:
    ```python
    import brahe as bh

    sp3 = bh.SP3.from_file("test_assets/sp3/example-d-2.sp3")
    print(sp3.num_satellites, sp3.num_epochs, sp3.header.agency)

    cod = bh.SP3.from_file("test_assets/sp3/COD0OPSFIN_20243150000_01D_05M_ORB_subset.SP3")
    g05 = cod.satellite("G05")
    print(g05.state_gcrf(g05.start_epoch + 450.0))
    ```
"""

from brahe._brahe import (
    SP3,
    SP3_DEFAULT_LAGRANGE_DEGREE,
    GNSSSystem,
    SP3Correlation,
    SP3FileType,
    SP3Header,
    SP3OrbitType,
    SP3Record,
    SP3Satellite,
    SP3SatelliteId,
    SP3TimeSystem,
    SP3Version,
)

__all__ = [
    "SP3",
    "SP3_DEFAULT_LAGRANGE_DEGREE",
    "GNSSSystem",
    "SP3Correlation",
    "SP3FileType",
    "SP3Header",
    "SP3OrbitType",
    "SP3Record",
    "SP3Satellite",
    "SP3SatelliteId",
    "SP3TimeSystem",
    "SP3Version",
]
