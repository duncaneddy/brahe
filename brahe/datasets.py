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
    ephemeris = datasets.celestrak.get_tles("gnss")

    # Or get as propagators directly
    propagators = datasets.celestrak.get_tles_as_propagators("gnss", 60.0)

    # Save to file
    datasets.celestrak.download_tles("gnss", "gnss.json", "3le", "json")

    # Load groundstations
    ksat_stations = datasets.groundstations.load("ksat")
    all_stations = datasets.groundstations.load_all()
    ```
"""

from brahe._brahe import (
    # CelesTrak functions
    celestrak_get_tles,
    celestrak_get_tles_as_propagators,
    celestrak_download_tles,
    celestrak_get_tle_by_id,
    celestrak_get_tle_by_id_as_propagator,
    celestrak_get_tle_by_name,
    celestrak_get_tle_by_name_as_propagator,
    # Groundstation functions
    groundstations_load,
    groundstations_load_from_file,
    groundstations_load_all,
    groundstations_list_providers,
)


# Create a celestrak namespace object
class _CelesTrakNamespace:
    """CelesTrak data source namespace"""

    get_tles = staticmethod(celestrak_get_tles)
    get_tles_as_propagators = staticmethod(celestrak_get_tles_as_propagators)
    download_tles = staticmethod(celestrak_download_tles)
    get_tle_by_id = staticmethod(celestrak_get_tle_by_id)
    get_tle_by_id_as_propagator = staticmethod(celestrak_get_tle_by_id_as_propagator)
    get_tle_by_name = staticmethod(celestrak_get_tle_by_name)
    get_tle_by_name_as_propagator = staticmethod(
        celestrak_get_tle_by_name_as_propagator
    )


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
