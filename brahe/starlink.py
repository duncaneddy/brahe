"""
Starlink Module

Downloads Starlink's public Modified ITC ephemerides from
``https://api.starlink.com/public-files/ephemerides/``, driven by the site
manifest and cached under ``$BRAHE_CACHE/starlink``.

This module provides:
- StarlinkClient: Manifest and ephemeris retrieval with caching, rate limiting, and ``BRAHE_NETWORK_MODE`` handling
- StarlinkManifest: The manifest as a queryable table with change detection and ``to_dataframe()``
- StarlinkManifestEntry: One manifest line with the epochs decoded from the file name
- RateLimitConfig: Per-minute and per-hour request caps for the client

Example:
    ```python
    import brahe as bh

    client = bh.starlink.StarlinkClient()
    manifest = client.get_manifest()
    print(manifest.to_dataframe().head())
    traj = client.get_trajectory(manifest[0].norad_cat_id)
    ```
"""

from brahe._brahe import (
    RateLimitConfig,
    StarlinkClient,
    StarlinkManifest,
    StarlinkManifestEntry,
)

__all__ = [
    "RateLimitConfig",
    "StarlinkClient",
    "StarlinkManifest",
    "StarlinkManifestEntry",
]
