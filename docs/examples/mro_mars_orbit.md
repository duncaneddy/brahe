# MRO Mars Orbit

In this example we'll set up the Mars Reconnaissance Orbiter's (MRO)
sun-synchronous science orbit and propagate it with brahe's Mars force model.
MRO has flown a ~255 x 320 km, 92.6 degree inclination orbit since 2006,
imaging the surface at a consistent local solar time on every pass. We'll
track how the osculating orbital elements evolve under the full force model
over a two-day propagation, then visualize the trajectory in 3D around a
textured Mars.

---

## Why Sun-Synchronous Orbits Work at Mars

A sun-synchronous orbit keeps its orbital plane at a fixed orientation
relative to the Sun, so the spacecraft crosses each latitude at the same
local solar time on every pass - valuable for consistent lighting in surface
imagery. This only works because a planet's oblateness (the J2 zonal
harmonic) causes the orbital plane to precess: for a near-polar, slightly
retrograde inclination, the nodal precession rate can be tuned to exactly
match the planet's mean motion around the Sun. At Mars, that inclination is
about 92.6 degrees - close to polar, with the small excess over 90 degrees
giving just enough retrograde precession to track Mars's slower (687-day)
year. MRO's 255 x 320 km, 92.6 degree orbit is designed around this
resonance.

## Orbit Setup

We define the orbit directly in Mars-centered Keplerian elements: a 255 x 320
km altitude sun-synchronous orbit. `state_koe_to_eci_for_body` converts
Keplerian elements to a Cartesian state using an arbitrary body's
gravitational parameter, here `GM_MARS`, since `state_koe_to_eci` assumes
Earth:

``` python
--8<-- "./examples/examples/mro_mars_orbit.py:preamble"
```

``` python
--8<-- "./examples/examples/mro_mars_orbit.py:orbit_setup"
```

## Propagation

We propagate under `ForceModelConfig.mars_default()`: 50x50 GMM-2B gravity,
exponential atmospheric drag, SRP occulted by Mars, and Sun third-body
perturbations. The propagator integrates in the Mars-Centered Inertial (MCI)
frame:

``` python
--8<-- "./examples/examples/mro_mars_orbit.py:propagation"
```

## Element Evolution

Sampling the trajectory over the 2-day propagation and converting each
Cartesian state back to Keplerian elements with `state_eci_to_koe_for_body`
shows how semi-major axis, eccentricity, and inclination evolve under the
full force model:

``` python
--8<-- "./examples/examples/mro_mars_orbit.py:element_history"
```

!!! note "state_bci vs. state_eci"
    `state_bci` returns the propagator's native state in the central body's
    body-centered inertial frame (MCI for a Mars-centered propagator).
    `state_eci` instead always returns an Earth-centered state - for a
    Mars-centered propagator it adds Mars's Earth-relative position, which
    would report a semi-major axis near the Earth-Mars distance rather than
    Mars orbit elements. Use `state_bci` whenever you need elements or
    altitude relative to the body being orbited.

``` python
--8<-- "./examples/examples/mro_mars_orbit.py:element_plot"
```

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/mro_mars_orbit_elements_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/mro_mars_orbit_elements_dark.html"  loading="lazy"></iframe>
</div>

The osculating inclination oscillates by about 1.1 degrees peak-to-peak over
the 2-day window - the 50x50 gravity field, drag, and SRP all perturb the
instantaneous element beyond the mean value used to design the orbit - while
staying centered close to the design inclination of 92.6 degrees. Semi-major
axis and eccentricity likewise show short-period oscillation without net
secular decay over this timespan, and periapsis altitude stays comfortably
above the Mars surface throughout.

## 3D Visualization

`plot_trajectory_3d` accepts `central_body="mars"` to render an interactive
3D view of the trajectory around a textured Mars. Non-Earth central bodies
require the plotted trajectory to already be in
`OrbitFrame.BodyCenteredInertial(naif_id)` for that body; a Mars-centered
`NumericalOrbitPropagator`'s `.trajectory` is already in that frame, so no
conversion is needed:

``` python
--8<-- "./examples/examples/mro_mars_orbit.py:plot_3d"
```

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/mro_mars_orbit_3d_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/mro_mars_orbit_3d_dark.html"  loading="lazy"></iframe>
</div>

*Body textures: [Solar System Scope](https://www.solarsystemscope.com/textures/), CC BY 4.0.*

## Full Code Example

```python title="mro_mars_orbit.py"
--8<-- "./examples/examples/mro_mars_orbit.py:all"
```

---

## See Also

- [Numerical Orbit Propagation](../learn/orbit_propagation/numerical_propagation/index.md) - Propagator fundamentals
- [Force Models](../learn/orbit_propagation/numerical_propagation/force_models.md) - Configuring force models, including `mars_default()`
- [3D Trajectory Plotting](../learn/plots/3d_trajectory.md) - Advanced options for trajectory visualization, including non-Earth central bodies
