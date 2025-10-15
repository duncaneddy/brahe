"""
Datasets Module

Provides access to satellite ephemeris data from various sources.

This module provides a source-specific API organized by data provider:
- celestrak: CelesTrak data source functions

Each source provides:
- Download and parse ephemeris data as 3LE tuples (name, line1, line2)
- Create ready-to-use SGP propagators from ephemeris data
- Save data in various file formats (txt, csv, json)

Example:
    ```python
    import brahe.datasets as datasets

    # Download ephemeris from CelesTrak
    ephemeris = datasets.celestrak.get_ephemeris("gnss")

    # Or get as propagators directly
    propagators = datasets.celestrak.get_ephemeris_as_propagators("gnss", 60.0)

    # Save to file
    datasets.celestrak.download_ephemeris("gnss", "gnss.json", "3le", "json")
    ```
"""

from brahe._brahe import (
    # CelesTrak functions
    celestrak_get_ephemeris,
    celestrak_get_ephemeris_as_propagators,
    celestrak_download_ephemeris,
)


# Create a celestrak namespace object
class _CelesTrakNamespace:
    """CelesTrak data source namespace"""

    get_ephemeris = staticmethod(celestrak_get_ephemeris)
    get_ephemeris_as_propagators = staticmethod(celestrak_get_ephemeris_as_propagators)
    download_ephemeris = staticmethod(celestrak_download_ephemeris)


# Create celestrak namespace instance
celestrak = _CelesTrakNamespace()

__all__ = [
    "celestrak",
]
