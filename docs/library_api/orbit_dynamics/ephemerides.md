# Ephemerides

Celestial body position calculations for Sun, Moon, and planets.

## Analytical Models

::: brahe.sun_position

::: brahe.moon_position

## DE440 Ephemerides

::: brahe.sun_position_spice

::: brahe.sun_velocity_spice

::: brahe.sun_state_spice

::: brahe.moon_position_spice

::: brahe.moon_velocity_spice

::: brahe.moon_state_spice

::: brahe.mercury_position_spice

::: brahe.mercury_velocity_spice

::: brahe.mercury_state_spice

::: brahe.venus_position_spice

::: brahe.venus_velocity_spice

::: brahe.venus_state_spice

::: brahe.mars_position_spice

::: brahe.mars_velocity_spice

::: brahe.mars_state_spice

::: brahe.jupiter_position_spice

::: brahe.jupiter_velocity_spice

::: brahe.jupiter_state_spice

::: brahe.saturn_position_spice

::: brahe.saturn_velocity_spice

::: brahe.saturn_state_spice

::: brahe.uranus_position_spice

::: brahe.uranus_velocity_spice

::: brahe.uranus_state_spice

::: brahe.neptune_position_spice

::: brahe.neptune_velocity_spice

::: brahe.neptune_state_spice

## Planetary-System Barycenters

The `*_position_spice`, `*_velocity_spice`, and `*_state_spice` functions for Mars,
Jupiter, Saturn, Uranus, and Neptune return the planet **body center**, which
auto-downloads the planet's satellite ephemeris kernel on first use. The
`*_barycenter_*_spice` variants below return the planetary-system **barycenter**
using only the DE kernel and are preferred for third-body force applications.

::: brahe.mars_barycenter_position_spice

::: brahe.mars_barycenter_velocity_spice

::: brahe.mars_barycenter_state_spice

::: brahe.jupiter_barycenter_position_spice

::: brahe.jupiter_barycenter_velocity_spice

::: brahe.jupiter_barycenter_state_spice

::: brahe.saturn_barycenter_position_spice

::: brahe.saturn_barycenter_velocity_spice

::: brahe.saturn_barycenter_state_spice

::: brahe.uranus_barycenter_position_spice

::: brahe.uranus_barycenter_velocity_spice

::: brahe.uranus_barycenter_state_spice

::: brahe.neptune_barycenter_position_spice

::: brahe.neptune_barycenter_velocity_spice

::: brahe.neptune_barycenter_state_spice

::: brahe.solar_system_barycenter_position_spice

::: brahe.solar_system_barycenter_velocity_spice

::: brahe.solar_system_barycenter_state_spice

::: brahe.ssb_position_spice

::: brahe.ssb_velocity_spice

::: brahe.ssb_state_spice

## See Also

- [Third-Body Perturbations](third_body.md) - Third-body acceleration calculations
- [SPICE Kernels](../spice/index.md) - Kernel registry, generic NAIF-ID queries, and PCK orientation
- [Datasets: NAIF](../../learn/datasets/naif.md) - DE440s ephemeris data
- [Orbital Dynamics Module](index.md) - Complete orbit dynamics API reference
