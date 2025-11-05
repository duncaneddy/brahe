# Coordinate Enumerations

Enumerations for specifying coordinate transformation types.

## EllipsoidalConversionType

::: brahe.EllipsoidalConversionType
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

Specifies the type of ellipsoidal conversion used in topocentric coordinate transformations.

**Values:**
- `GEOCENTRIC` - Uses geocentric latitude where the angle is measured from the center of the Earth
- `GEODETIC` - Uses geodetic latitude where the angle is measured perpendicular to the WGS84 ellipsoid

---

## AngleFormat

::: brahe.AngleFormat
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

Specifies whether angles are in radians or degrees.

**Values:**
- `RADIANS` - Angles are in radians
- `DEGREES` - Angles are in degrees

---

## See Also

- [Topocentric Coordinates](topocentric.md)
- [Geodetic & Geocentric](geodetic.md)
