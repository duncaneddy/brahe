"""
Datasets Module

Provides access to satellite ephemeris and groundstation location data from various sources.

This module provides a source-specific API organized by data provider:
- celestrak: CelesTrak satellite ephemeris data
- groundstations: Curated groundstation location datasets

Example:
    ```python
    import brahe.datasets as datasets

    # Download ephemeris from CelesTrak
    ephemeris = datasets.celestrak.get_ephemeris("gnss")

    # Or get as propagators directly
    propagators = datasets.celestrak.get_ephemeris_as_propagators("gnss", 60.0)

    # Save to file
    datasets.celestrak.download_ephemeris("gnss", "gnss.json", "3le", "json")

    # Load groundstations
    ksat_stations = datasets.groundstations.load("ksat")
    all_stations = datasets.groundstations.load_all()
    ```
"""

from brahe._brahe import (
    # CelesTrak functions
    celestrak_get_ephemeris,
    celestrak_get_ephemeris_as_propagators,
    celestrak_download_ephemeris,
    # Groundstation functions
    groundstations_load,
    groundstations_load_from_file,
    groundstations_load_all,
    groundstations_list_providers,
)


# Create a celestrak namespace object
class _CelesTrakNamespace:
    """CelesTrak data source namespace"""

    get_ephemeris = staticmethod(celestrak_get_ephemeris)
    get_ephemeris_as_propagators = staticmethod(celestrak_get_ephemeris_as_propagators)
    download_ephemeris = staticmethod(celestrak_download_ephemeris)


# Create celestrak namespace instance
celestrak = _CelesTrakNamespace()


# Create a groundstations namespace object
class _GroundstationsNamespace:
    """Groundstation datasets namespace"""

    load = staticmethod(groundstations_load)
    load_from_file = staticmethod(groundstations_load_from_file)
    load_all = staticmethod(groundstations_load_all)
    list_providers = staticmethod(groundstations_list_providers)


# Create groundstations namespace instance
groundstations = _GroundstationsNamespace()

__all__ = [
    "celestrak",
    "groundstations",
]
