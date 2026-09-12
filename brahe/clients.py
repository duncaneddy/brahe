"""
Clients Module

Groups the network data clients under one namespace, mirroring the Rust
``brahe::clients`` module. The flat module paths (``brahe.celestrak``,
``brahe.spacetrack``, ``brahe.datasets.gcat``) remain the canonical imports;
this module re-exports them so both spellings resolve, as they do in Rust.

Example:
    ```python
    import brahe as bh

    client = bh.clients.celestrak.CelestrakClient()
    limits = bh.clients.RateLimitConfig(max_per_minute=10, max_per_hour=100)
    satcat = bh.clients.gcat.get_satcat()
    ```
"""

from brahe import celestrak, spacetrack
from brahe.datasets import gcat
from brahe.spacetrack import RateLimitConfig

__all__ = ["RateLimitConfig", "celestrak", "gcat", "spacetrack"]
