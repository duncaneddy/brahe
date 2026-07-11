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
- pck_euler_angles / pck_rotation_matrix: Generic binary PCK orientation queries
- NAIF_* constants: NAIF body ID constants

Example:
    ```python
    import brahe as bh

    bh.load_kernel("de440s")
    epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
    r_moon = bh.spk_position(bh.NAIF_MOON, bh.NAIF_EARTH, epc)
    ```
"""

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
    pck_rotation_matrix,
    NAIF_SSB,
    NAIF_MERCURY_BARYCENTER,
    NAIF_VENUS_BARYCENTER,
    NAIF_EMB,
    NAIF_MARS_BARYCENTER,
    NAIF_JUPITER_BARYCENTER,
    NAIF_SATURN_BARYCENTER,
    NAIF_URANUS_BARYCENTER,
    NAIF_NEPTUNE_BARYCENTER,
    NAIF_PLUTO_BARYCENTER,
    NAIF_SUN,
    NAIF_MERCURY,
    NAIF_VENUS,
    NAIF_EARTH,
    NAIF_MOON,
    NAIF_MARS,
)

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
    "pck_rotation_matrix",
    "NAIF_SSB",
    "NAIF_MERCURY_BARYCENTER",
    "NAIF_VENUS_BARYCENTER",
    "NAIF_EMB",
    "NAIF_MARS_BARYCENTER",
    "NAIF_JUPITER_BARYCENTER",
    "NAIF_SATURN_BARYCENTER",
    "NAIF_URANUS_BARYCENTER",
    "NAIF_NEPTUNE_BARYCENTER",
    "NAIF_PLUTO_BARYCENTER",
    "NAIF_SUN",
    "NAIF_MERCURY",
    "NAIF_VENUS",
    "NAIF_EARTH",
    "NAIF_MOON",
    "NAIF_MARS",
]
