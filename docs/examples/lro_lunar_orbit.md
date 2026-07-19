# LRO Lunar Orbit

In this example we'll set up an LRO-like low lunar science orbit and
propagate it with brahe's lunar force model. The Lunar Reconnaissance
Orbiter (LRO) has flown a polar, near-frozen low lunar science orbit around
the Moon since 2009, mapping the surface at high resolution; this example
uses an LRO-like ~30 x 180 km class orbit rather than reproducing its exact
orbital history. We'll compare a full-fidelity propagation against a
point-mass Moon to quantify how the Moon's lumpy gravity field
("[mascons](https://en.wikipedia.org/wiki/Mass_concentration_%28astronomy%29)" -
dense, gravitationally anomalous regions beneath several lunar maria)
perturbs a low lunar orbit, then visualize the trajectory in 3D around a
textured Moon.

---

## Why Lunar Orbits Are Different

Unlike Earth, the Moon's gravity field is dominated by large, irregular mass
concentrations rather than a smooth oblateness term. Orbit designers counter
the resulting perturbations by choosing a "frozen orbit" - an inclination and
argument of perilune where the long-period perturbation from the gravity
field's dominant harmonics averages to zero, so eccentricity and perilune
altitude stay bounded over many orbits instead of drifting monotonically.
Even so, these mascon-driven gravity anomalies are strong enough that most
low lunar orbits are unstable on timescales of weeks to months without
station-keeping. Proper modeling of the high-order gravity dynamics enable
design of station-keeping maneuvers that minimize the growth of these
instabilities, conserving fuel and extending mission-lifetime.

## Orbit Setup

The propagator integrates in the Moon-Centered Inertial (LCI) frame, whose
axes are ICRF-aligned: the LCI z-axis is the ICRF pole, which sits about 22
degrees from the Moon's spin pole. Passing the elements straight to
[`state_koe_to_eci`](../library_api/coordinates/cartesian.md#brahe.coordinates.state_koe_to_eci) would measure the 85.2 degree inclination and the
south-pole perilune against the ICRF pole, not the lunar equator.
[`state_koe_to_inertial_for_body`](../library_api/coordinates/cartesian.md#brahe.coordinates.state_koe_to_inertial_for_body) instead references the elements to the body's
mean equator at J2000 - the plane normal to the body's IAU pole, with the
x-axis on the ascending node of that equator on the ICRF equator - and returns
the state directly in the body-centered inertial frame (LCI for the Moon). It
takes a [`CentralBody`](../library_api/propagators/force_model_config.md#central-body) (which supplies both the Moon's gravitational parameter
and its pole), so the orbit is placed against the lunar equator with no manual
basis construction. We also evaluate the lunar mean pole at J2000 (the third
row of the ICRF-to-lunar-body-fixed rotation) to confirm the geometry below:

``` python
--8<-- "./examples/examples/lro_lunar_orbit.py:preamble"
```

With the standard preamble in place, the next step sets up the frozen orbit geometry.

``` python
--8<-- "./examples/examples/lro_lunar_orbit.py:orbit_setup"
```

## Propagation

We propagate the same initial state under two force models: `lunar_default()`
(50x50 GRGM660PRIM gravity, SRP occulted by the Moon and Earth, and Earth/Sun
third-body perturbations) and a point-mass Moon via [`ForceModelConfig.for_body`](../library_api/propagators/force_model_config.md#brahe.ForceModelConfig.for_body)
with `GravityConfiguration.point_mass()`. Both propagators integrate in the
Moon-Centered Inertial (LCI) frame:

``` python
--8<-- "./examples/examples/lro_lunar_orbit.py:propagation"
```

!!! note "state_bci vs. state_eci"
    [`state_bci`](../learn/orbit_propagation/numerical_propagation/cislunar_lunar_propagation.md)
    ([API](../library_api/propagators/numerical_orbit_propagator.md#brahe.NumericalOrbitPropagator.state_bci))
    returns the propagator's native state in the central body's
    body-centered inertial frame (LCI for a Moon-centered propagator).
    [`state_eci`](../learn/orbit_propagation/numerical_propagation/numerical_orbit_propagator.md#state-at-arbitrary-epochs)
    ([API](../library_api/propagators/numerical_orbit_propagator.md#brahe.NumericalOrbitPropagator.state_eci))
    instead always returns an Earth-centered state - for a
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

[`plot_trajectory_3d`](../library_api/plots/3d_trajectory.md) accepts `central_body="moon"` to render an interactive
3D view of the trajectory around a textured Moon. Non-Earth central bodies
require the plotted trajectory to already be in
[`OrbitFrame.BodyCenteredInertial(naif_id)`](../library_api/orbits/enums.md#brahe.OrbitFrame.BodyCenteredInertial) for that body; a Moon-centered
[`NumericalOrbitPropagator`](../library_api/propagators/numerical_orbit_propagator.md)'s `.trajectory` is already in that frame, so no
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

??? "Full Code"

    ```python title="lro_lunar_orbit.py"
    --8<-- "./examples/examples/lro_lunar_orbit.py:all"
    ```

---

## See Also

- [Numerical Orbit Propagation](../learn/orbit_propagation/numerical_propagation/index.md) - Propagator fundamentals
- [Force Models](../learn/orbit_propagation/numerical_propagation/force_models.md) - Configuring force models, including `lunar_default()`
- [3D Trajectory Plotting](../learn/plots/3d_trajectory.md) - Advanced options for trajectory visualization, including non-Earth central bodies
