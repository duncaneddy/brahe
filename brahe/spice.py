"""
SPICE Module

Provides access to the native SPICE kernel registry: loading/unloading SPK
(.bsp) and binary PCK (.bpc) kernels, and generic ephemeris/orientation
queries against the registry.

This module provides:
- load_kernel / unload_kernel / clear_kernels / loaded_kernels: Kernel registry management
- load_common_kernels / load_all_kernels: Bulk kernel pre-loading helpers
- spk_position / spk_velocity / spk_state: Generic SPK queries against all loaded kernels
- spk_position_from_kernel / spk_velocity_from_kernel / spk_state_from_kernel: SPK queries scoped to a single named kernel
- pck_euler_angles / pck_euler_angle / pck_euler_rates / pck_euler_angle_and_rates / pck_quaternion / pck_rotation_matrix: Generic binary PCK orientation queries
- NAIFId / FrameId: NAIF body ID and PCK frame ID IntEnums

Example:
    ```python
    import brahe as bh

    bh.load_kernel("de440s")
    epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
    r_moon = bh.spk_position(bh.NAIFId.MOON, bh.NAIFId.EARTH, epc)
    ```
"""

from enum import IntEnum

from brahe._brahe import (
    load_kernel,
    unload_kernel,
    clear_kernels,
    loaded_kernels,
    load_common_kernels,
    load_all_kernels,
    spk_position,
    spk_velocity,
    spk_state,
    spk_position_from_kernel,
    spk_velocity_from_kernel,
    spk_state_from_kernel,
    pck_euler_angles,
    pck_euler_angle,
    pck_euler_rates,
    pck_euler_angle_and_rates,
    pck_quaternion,
    pck_rotation_matrix,
)


class NAIFId(IntEnum):
    """NAIF integer ID codes for solar-system bodies.

    Values pass directly to any function taking a NAIF ID; arbitrary raw
    integer IDs are equally accepted by those functions. Full ID list:
    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html
    """

    SOLAR_SYSTEM_BARYCENTER = 0
    MERCURY_BARYCENTER = 1
    VENUS_BARYCENTER = 2
    EARTH_MOON_BARYCENTER = 3
    MARS_BARYCENTER = 4
    JUPITER_BARYCENTER = 5
    SATURN_BARYCENTER = 6
    URANUS_BARYCENTER = 7
    NEPTUNE_BARYCENTER = 8
    PLUTO_BARYCENTER = 9
    SUN = 10
    MERCURY = 199
    VENUS = 299
    EARTH = 399
    MOON = 301
    MARS = 499
    JUPITER = 599
    SATURN = 699
    URANUS = 799
    NEPTUNE = 899
    PLUTO = 999
    PHOBOS = 401
    DEIMOS = 402
    IO = 501
    EUROPA = 502
    GANYMEDE = 503
    CALLISTO = 504
    TITAN = 606
    ARIEL = 701
    UMBRIEL = 702
    TITANIA = 703
    OBERON = 704
    MIRANDA = 705
    TRITON = 801
    CHARON = 901


class FrameId(IntEnum):
    """PCK body-frame class IDs. Raw integer IDs are equally accepted."""

    MOON_PA_DE440 = 31008


__all__ = [
    "load_kernel",
    "unload_kernel",
    "clear_kernels",
    "loaded_kernels",
    "load_common_kernels",
    "load_all_kernels",
    "spk_position",
    "spk_velocity",
    "spk_state",
    "spk_position_from_kernel",
    "spk_velocity_from_kernel",
    "spk_state_from_kernel",
    "pck_euler_angles",
    "pck_euler_angle",
    "pck_euler_rates",
    "pck_euler_angle_and_rates",
    "pck_quaternion",
    "pck_rotation_matrix",
    "NAIFId",
    "FrameId",
]
