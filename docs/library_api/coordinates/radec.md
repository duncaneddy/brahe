# Right Ascension / Declination Coordinates

Functions for converting between right ascension/declination (RA/Dec) and Cartesian inertial coordinates, propagating catalog positions with proper motion, and converting between topocentric RA/Dec and azimuth-elevation.

## Position Conversions

::: brahe.coordinates.position_radec_to_inertial

::: brahe.coordinates.position_inertial_to_radec

## State Conversions

::: brahe.coordinates.state_radec_to_inertial

::: brahe.coordinates.state_inertial_to_radec

## Proper Motion

::: brahe.coordinates.apply_proper_motion

## Azimuth-Elevation Conversions

::: brahe.coordinates.position_radec_to_azel

::: brahe.coordinates.position_azel_to_radec

---

## See Also

- [RA/Dec Transformations](../../learn/coordinates/radec_transformations.md) - Overview, equations, and usage guide
- [Topocentric Coordinates API Reference](topocentric.md) - ENZ/SEZ and azimuth-elevation functions
- [Cartesian Coordinates API Reference](cartesian.md) - Orbital element and Cartesian state conversions
- [Star Catalogs](../../learn/datasets/star_catalogs.md) - Catalog records that use these conversions internally
