"""
Datasets Module

Provides access to groundstation location data, NAIF ephemeris kernels,
and GCAT (General Catalog of Artificial Space Objects) satellite catalogs.

This module provides a source-specific API organized by data provider:
- groundstations: Curated groundstation location datasets
- naif: NASA JPL NAIF ephemeris kernels (DE series)
- gcat: Jonathan McDowell's GCAT satellite catalogs (SATCAT, PSATCAT)

For CelestrakClient satellite catalog data, use the `brahe.celestrak` module instead.

Example:
    ```python
    import brahe.datasets as datasets

    # Load groundstations
    ksat_stations = datasets.groundstations.load("ksat")
    all_stations = datasets.groundstations.load_all()

    # Download NAIF DE kernel
    kernel_path = datasets.naif.download_de_kernel("de440s")

    # GCAT satellite catalogs
    satcat = datasets.gcat.get_satcat()
    iss = satcat.get_by_satcat("25544")
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
    # GCAT functions and types
    gcat_get_satcat,
    gcat_get_psatcat,
    GCATSatcatRecord,
    GCATPsatcatRecord,
    GCATSatcat,
    GCATPsatcat,
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


# Create a GCAT namespace object
class _GcatNamespace:
    """GCAT data source namespace"""

    get_satcat = staticmethod(gcat_get_satcat)
    get_psatcat = staticmethod(gcat_get_psatcat)
    SatcatRecord = GCATSatcatRecord
    PsatcatRecord = GCATPsatcatRecord
    Satcat = GCATSatcat
    Psatcat = GCATPsatcat


# Create GCAT namespace instance
gcat = _GcatNamespace()

__all__ = [
    "groundstations",
    "naif",
    "gcat",
]
