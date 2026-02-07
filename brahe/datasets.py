"""
Datasets Module

Provides access to groundstation location data and NAIF ephemeris kernels.

This module provides a source-specific API organized by data provider:
- groundstations: Curated groundstation location datasets
- naif: NASA JPL NAIF ephemeris kernels (DE series)

For CelestrakClient satellite catalog data, use the `brahe.celestrak` module instead.

Example:
    ```python
    import brahe.datasets as datasets

    # Load groundstations
    ksat_stations = datasets.groundstations.load("ksat")
    all_stations = datasets.groundstations.load_all()

    # Download NAIF DE kernel
    kernel_path = datasets.naif.download_de_kernel("de440s")
    ```
"""

from brahe._brahe import (
    # Groundstation functions
    groundstations_load,
    groundstations_load_from_file,
    groundstations_load_all,
    groundstations_list_providers,
    # NAIF functions
    naif_download_de_kernel,
)


# Create a groundstations namespace object
class _GroundstationsNamespace:
    """Groundstation datasets namespace"""

    load = staticmethod(groundstations_load)
    load_from_file = staticmethod(groundstations_load_from_file)
    load_all = staticmethod(groundstations_load_all)
    list_providers = staticmethod(groundstations_list_providers)


# Create groundstations namespace instance
groundstations = _GroundstationsNamespace()


# Create a NAIF namespace object
class _NAIFNamespace:
    """NAIF data source namespace"""

    download_de_kernel = staticmethod(naif_download_de_kernel)


# Create NAIF namespace instance
naif = _NAIFNamespace()

__all__ = [
    "groundstations",
    "naif",
]
