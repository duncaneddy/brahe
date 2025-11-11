# Keplerian Propagator

Analytical two-body orbit propagator using Keplerian orbital elements. The Keplerian propagator provides fast, analytical orbit propagation for unperturbed two-body motion. It uses closed-form solutions to Kepler's equations for orbital element propagation.

## Orbital Elements

The propagator accepts orbital elements in the following order:
1. **a** - Semi-major axis (meters)
2. **e** - Eccentricity (dimensionless)
3. **i** - Inclination (degrees radians)
4. **Ω** - Right ascension of ascending node (degrees radians)
5. **ω** - Argument of periapsis (degrees radians)
6. **M** or **ν** - Mean anomaly or true anomaly (degrees radians)

Use `OrbitRepresentation` to specify element type:
- `MEAN_ELEMENTS` - Mean orbital elements with mean anomaly
- `OSCULATING_ELEMENTS` - Osculating elements with true anomaly

## Class Reference

::: brahe.KeplerianPropagator
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 2
---

## See Also

- [SGPPropagator](sgp_propagator.md) - SGP4/SDP4 propagator for TLE data
- [Keplerian Elements](../orbits/keplerian.md) - Orbital element conversion functions
- [OrbitRepresentation](../orbits/enums.md#orbitrepresentation) - Element type specification
