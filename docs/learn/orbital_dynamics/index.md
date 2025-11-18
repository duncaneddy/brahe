# Orbital Dynamics and Perturbations

Orbital dynamics describes how satellites and celestial bodies move under the influence of various forces. While simple two-body motion (Keplerian orbits) provides a useful first approximation, real satellite motion is affected by numerous perturbations that cause deviations from these idealized orbits.

## Overview of Perturbation Forces

The motion of an Earth-orbiting satellite is influenced by several perturbation forces beyond the central body's point-mass gravity:

- **Non-spherical Earth gravity**: The is not actually a perfect, uniform-density sphere. The Earth's non-spherical shape and mass distribution create additional gravitational forces. This distribution we model using spherical harmonics
- **Third-body perturbations**: Gravitational effects from the Moon, Sun, and planets also impact satellite orbits with the Moon and Sun being the most significant
- **Atmospheric drag**: Despite being in space, satellites in low Earth orbit still encounter traces of the Earth's atmosphere which create drag forces that cause orbital decay. Drag is a non-conservative force that dissipates energy from the orbit. It is highly dependent on atmospheric density, which varies with altitude, solar activity, and geomagnetic conditions, making it challenging to model accurately and the greatest source of uncertainty in LEO orbit prediction. Drag does not affect higher altitude orbits significantly.
- **Solar radiation pressure**: The sun emits photons that exert pressure on satellite surfaces. This force is more pronounced for satellites with large surface areas relative to their mass, such as those with solar panels or large antennas.
- **Relativistic effects**: While generally small, corrections from general relativity can be significant for high-precision orbit determination and timekeeping.

The relative importance of these forces varies significantly with orbital altitude and satellite characteristics.

## Force Magnitude Comparison

The following interactive plot shows the magnitude of various perturbation accelerations as a function of altitude for a satellite with $500 \text{ kg}$ mass, an area of $2.0 \text{ m}^2$, a coefficient of drag of $C_d = 2.3$, and a coefficient of reflectivity of $C_r = 1.8$. This visualization helps determine which force models are necessary for different orbital regimes from low Earth orbit (LEO) through geostationary orbit (GEO).

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/fig_perturbation_magnitudes_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/fig_perturbation_magnitudes_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="fig_perturbation_magnitudes.py"
    --8<-- "./plots/fig_perturbation_magnitudes.py:12"
    ```

## Available Perturbation Models

Brahe provides implementations of the following perturbation acceleration models:

- [**Gravity Models**](gravity.md): Point-mass and spherical harmonic expansion (EGM2008, GGM05S, JGM3)
- [**Third-Body Perturbations**](third_body.md): Sun, Moon, and planetary effects using analytical or DE440s ephemerides
- [**Atmospheric Drag**](drag.md): With Harris-Priester density model
- [**Solar Radiation Pressure**](solar_radiation_pressure.md): Including conical Earth shadow model
- [**Relativistic Effects**](relativity.md): General relativistic corrections

Each model is available in both Rust and Python with identical interfaces. See the individual pages for detailed explanations and usage examples.

## References

1. [Montenbruck, O., & Gill, E. (2000). *Satellite Orbits: Models, Methods, and Applications*. Springer. Chapter 3: Force Model.](https://link.springer.com/book/10.1007/978-3-642-58351-3)
