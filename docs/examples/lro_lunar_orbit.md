# LRO Lunar Orbit

In this example we'll set up the Lunar Reconnaissance Orbiter's (LRO) low
lunar science orbit and propagate it with brahe's lunar force model. LRO has
flown a roughly 30 x 200 km class polar, near-frozen science orbit around
the Moon since 2009, mapping the surface at high resolution. We'll compare
a full-fidelity propagation against a point-mass
Moon to quantify how the Moon's lumpy gravity field ("mascons" - dense,
gravitationally anomalous regions beneath several lunar maria) perturbs a low
lunar orbit, then visualize the trajectory in 3D around a textured Moon.

---

## Why Lunar Orbits Are Different

Unlike Earth, the Moon's gravity field is dominated by large, irregular mass
concentrations rather than a smooth oblateness term. These mascons produce
gravity anomalies strong enough that most low lunar orbits are unstable on
timescales of weeks to months without station-keeping: eccentricity grows
until the spacecraft impacts the surface. Orbit designers counter this by
choosing a "frozen orbit" - an inclination and argument of perilune where the
long-period perturbation from the gravity field's dominant harmonics
averages to zero, so eccentricity and perilune altitude stay bounded over
many orbits instead of drifting monotonically. LRO's polar, ~85 degree
inclination orbit with perilune fixed over the southern hemisphere
(argument of perilune 270 degrees) is one such configuration.

## Orbit Setup

The propagator integrates in the Moon-Centered Inertial (LCI) frame, whose
axes are ICRF-aligned: the LCI z-axis is the ICRF pole, which sits about 22
degrees from the Moon's spin pole. `state_koe_to_eci_for_body` is a
frame-agnostic Keplerian-to-Cartesian conversion that measures inclination
against the z-axis of whatever basis its output is read in, so interpreting
its output directly as an LCI state would reference the 85.2 degree
inclination and the south-pole perilune to the ICRF pole, not the lunar
equator. To place the orbit correctly, we build a lunar-equatorial inertial
basis - z-axis on the lunar mean pole (the third row of the LCI-to-LFME
rotation), x-axis on the ascending node of the lunar equator on the ICRF
equator - construct the state in that basis, and rotate its position and
velocity into LCI. `state_koe_to_eci_for_body` converts the Keplerian elements
to a Cartesian state using an arbitrary body's gravitational parameter, here
`GM_MOON`, since `state_koe_to_eci` assumes Earth:

``` python
--8<-- "./examples/examples/lro_lunar_orbit.py:preamble"
```

``` python
--8<-- "./examples/examples/lro_lunar_orbit.py:orbit_setup"
```

## Propagation

We propagate the same initial state under two force models: `lunar_default()`
(50x50 GRGM660PRIM gravity, SRP occulted by the Moon and Earth, and Earth/Sun
third-body perturbations) and a point-mass Moon via `ForceModelConfig.for_body`
with `GravityConfiguration.point_mass()`. Both propagators integrate in the
Moon-Centered Inertial (LCI) frame:

``` python
--8<-- "./examples/examples/lro_lunar_orbit.py:propagation"
```

!!! note "state_bci vs. state_eci"
    `state_bci` returns the propagator's native state in the central body's
    body-centered inertial frame (LCI for a Moon-centered propagator).
    `state_eci` instead always returns an Earth-centered state - for a
    Moon-centered propagator it adds the Moon's Earth-relative position,
    which would report altitudes near the Earth-Moon distance rather than
    above the lunar surface. Use `state_bci` whenever you need the
    distance from the body being orbited.

## Altitude Comparison

Sampling both trajectories over the 7-day propagation and computing altitude
above `R_MOON` from the Moon-centered radius, then plotting the difference
between the full-gravity and point-mass solutions, shows how steadily the
mascons perturb the orbit:

``` python
--8<-- "./examples/examples/lro_lunar_orbit.py:perilune_history"
```

``` python
--8<-- "./examples/examples/lro_lunar_orbit.py:altitude_plot"
```

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/lro_lunar_orbit_altitude_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/lro_lunar_orbit_altitude_dark.html"  loading="lazy"></iframe>
</div>

Over 7 days, the difference between the full and point-mass solutions grows
to about 17 km at its peak - a substantial fraction of the orbit's own 150 km
altitude range between perilune and apolune - while the full-model orbit's
perilune altitude stays above 26 km, remaining bound to the Moon.

## 3D Visualization

`plot_trajectory_3d` accepts `central_body="moon"` to render an interactive
3D view of the trajectory around a textured Moon. Non-Earth central bodies
require the plotted trajectory to already be in
`OrbitFrame.BodyCenteredInertial(naif_id)` for that body; a Moon-centered
`NumericalOrbitPropagator`'s `.trajectory` is already in that frame, so no
conversion is needed. We plot the final 12 hours of the full-gravity
trajectory:

``` python
--8<-- "./examples/examples/lro_lunar_orbit.py:plot_3d"
```

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/lro_lunar_orbit_3d_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/lro_lunar_orbit_3d_dark.html"  loading="lazy"></iframe>
</div>

*Body textures: [Solar System Scope](https://www.solarsystemscope.com/textures/), CC BY 4.0.*

## Full Code Example

```python title="lro_lunar_orbit.py"
--8<-- "./examples/examples/lro_lunar_orbit.py:all"
```

---

## See Also

- [Numerical Orbit Propagation](../learn/orbit_propagation/numerical_propagation/index.md) - Propagator fundamentals
- [Force Models](../learn/orbit_propagation/numerical_propagation/force_models.md) - Configuring force models, including `lunar_default()`
- [3D Trajectory Plotting](../learn/plots/3d_trajectory.md) - Advanced options for trajectory visualization, including non-Earth central bodies
