# Ephemerides

Celestial body position calculations for Sun, Moon, and planets.

## Analytical Models

::: brahe.sun_position

::: brahe.moon_position

## DE440 Ephemerides

::: brahe.initialize_ephemeris

::: brahe.sun_position_de

::: brahe.sun_velocity_de

::: brahe.sun_state_de

::: brahe.moon_position_de

::: brahe.moon_velocity_de

::: brahe.moon_state_de

::: brahe.mercury_position_de

::: brahe.mercury_velocity_de

::: brahe.mercury_state_de

::: brahe.venus_position_de

::: brahe.venus_velocity_de

::: brahe.venus_state_de

::: brahe.mars_position_de

::: brahe.mars_velocity_de

::: brahe.mars_state_de

::: brahe.jupiter_position_de

::: brahe.jupiter_velocity_de

::: brahe.jupiter_state_de

::: brahe.saturn_position_de

::: brahe.saturn_velocity_de

::: brahe.saturn_state_de

::: brahe.uranus_position_de

::: brahe.uranus_velocity_de

::: brahe.uranus_state_de

::: brahe.neptune_position_de

::: brahe.neptune_velocity_de

::: brahe.neptune_state_de

## Planetary-System Barycenters

The `*_position_de`, `*_velocity_de`, and `*_state_de` functions for Mars,
Jupiter, Saturn, Uranus, and Neptune return the planet **body center**, which
auto-downloads the planet's satellite-system kernel on first use. The
`*_barycenter_*_de` variants below return the planetary-system **barycenter**
using only the DE kernel and are preferred for third-body force applications.

::: brahe.mars_barycenter_position_de

::: brahe.mars_barycenter_velocity_de

::: brahe.mars_barycenter_state_de

::: brahe.jupiter_barycenter_position_de

::: brahe.jupiter_barycenter_velocity_de

::: brahe.jupiter_barycenter_state_de

::: brahe.saturn_barycenter_position_de

::: brahe.saturn_barycenter_velocity_de

::: brahe.saturn_barycenter_state_de

::: brahe.uranus_barycenter_position_de

::: brahe.uranus_barycenter_velocity_de

::: brahe.uranus_barycenter_state_de

::: brahe.neptune_barycenter_position_de

::: brahe.neptune_barycenter_velocity_de

::: brahe.neptune_barycenter_state_de

::: brahe.solar_system_barycenter_position_de

::: brahe.solar_system_barycenter_velocity_de

::: brahe.solar_system_barycenter_state_de

::: brahe.ssb_position_de

::: brahe.ssb_velocity_de

::: brahe.ssb_state_de

## See Also

- [Third-Body Perturbations](third_body.md) - Third-body acceleration calculations
- [SPICE Kernels](../spice/index.md) - Kernel registry, generic NAIF-ID queries, and PCK orientation
- [Datasets: NAIF](../../learn/datasets/naif.md) - DE440s ephemeris data
- [Orbital Dynamics Module](index.md) - Complete orbit dynamics API reference
