# Ephemeris Data Sources

API reference for the satellite ephemeris data source clients. Both CelesTrak and Space-Track return `GPRecord` for GP queries, with shared operator functions for filtering.

## Submodules

- [Shared Types](shared_types.md) -- `GPRecord` and operator functions shared by both clients
- [CelesTrak](celestrak.md) -- `CelestrakClient`, `CelestrakQuery`, and CelesTrak-specific types
- [Space-Track](spacetrack/index.md) -- `SpaceTrackClient`, `SpaceTrackQuery`, and Space-Track-specific types

---

## See Also

- [Ephemeris Data Sources Overview](../../learn/ephemeris/index.md) -- Conceptual introduction, GPRecord field tables, and source comparison
