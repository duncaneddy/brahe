"""
Datasets Module

Provides access to groundstation location data, NAIF ephemeris kernels,
GCAT (General Catalog of Artificial Space Objects) satellite catalogs,
and ICGEM spherical harmonic gravity models.

This module provides a source-specific API organized by data provider:
- groundstations: Curated groundstation location datasets
- naif: NASA JPL NAIF ephemeris kernels (DE series)
- gcat: Jonathan McDowell's GCAT satellite catalogs (SATCAT, PSATCAT)
- icgem: ICGEM spherical harmonic gravity model catalog (Earth + celestial bodies)
- ssn_sensors: Vallado Space Surveillance Network sensor site dataset

For CelestrakClient satellite catalog data, use the `brahe.celestrak` module instead.

Example:
    ```python
    import brahe.datasets as datasets

    # Load groundstations
    ksat_stations = datasets.groundstations.load("ksat")
    all_stations = datasets.groundstations.load_all()

    # Download NAIF DE kernel
    kernel_path = datasets.naif.download_spice_kernel("de440s")

    # GCAT satellite catalogs
    satcat = datasets.gcat.get_satcat()
    iss = satcat.get_by_satcat("25544")

    # ICGEM gravity models
    earth_models = datasets.icgem.list_models("earth")
    path = datasets.icgem.download_model("earth", "JGM3")
    ```
"""

from brahe._brahe import (
    # Groundstation functions
    groundstations_load,
    groundstations_load_from_file,
    groundstations_load_all,
    groundstations_list_providers,
    # SSN sensor functions
    ssn_sensors_load,
    # NAIF functions
    download_spice_kernel,
    # GCAT functions and types
    gcat_get_satcat,
    gcat_get_psatcat,
    GCATSatcatRecord,
    GCATPsatcatRecord,
    GCATSatcat,
    GCATPsatcat,
    # ICGEM functions and types
    icgem_list_models,
    icgem_refresh_index,
    icgem_refresh_all_indexes,
    icgem_download_model,
    ICGEMIndexEntry,
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


class _SSNSensorsNamespace:
    """Namespace for SSN sensor dataset functions."""

    load = staticmethod(ssn_sensors_load)


# Create ssn_sensors namespace instance
ssn_sensors = _SSNSensorsNamespace()


# Create a NAIF namespace object
class _NAIFNamespace:
    """NAIF data source namespace"""

    download_spice_kernel = staticmethod(download_spice_kernel)


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


# Create an ICGEM namespace object
class _ICGEMNamespace:
    """ICGEM gravity model dataset namespace."""

    list_models = staticmethod(icgem_list_models)
    refresh_index = staticmethod(icgem_refresh_index)
    refresh_all_indexes = staticmethod(icgem_refresh_all_indexes)
    download_model = staticmethod(icgem_download_model)
    IndexEntry = ICGEMIndexEntry


# Create ICGEM namespace instance
icgem = _ICGEMNamespace()

__all__ = [
    "groundstations",
    "naif",
    "gcat",
    "icgem",
    "ssn_sensors",
]
