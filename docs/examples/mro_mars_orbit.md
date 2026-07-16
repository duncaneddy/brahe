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

The propagator integrates in the Mars-Centered Inertial (MCI) frame, whose
axes are ICRF-aligned: the MCI z-axis is the ICRF pole, which sits about 37
degrees from Mars's spin pole. `state_koe_to_eci_for_body` is a frame-agnostic
Keplerian-to-Cartesian conversion that measures inclination against the
z-axis of whatever basis its output is interpreted in, so feeding it 92.6
degrees and reading the result as an MCI state would reference the inclination
to the ICRF pole, not Mars's equator. To place the orbit correctly, we build a
Mars-equatorial inertial basis - z-axis on the Mars spin pole (the third row
of the MCI-to-MCMF rotation), x-axis on the ascending node of the Mars equator
on the ICRF equator - construct the state in that basis, and rotate its
position and velocity into MCI.

The same step derives the ascending node from the Sun's direction rather than
hardcoding it: the right ascension of the ascending node is set to the Sun's
right ascension in the Mars-equatorial basis plus 45 degrees, placing the node
near a 15:00 (mid-afternoon) local solar time. `state_koe_to_eci_for_body`
converts the Keplerian elements to a Cartesian state using an arbitrary body's
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
Cartesian state back to Keplerian elements with `state_eci_to_koe_for_body`,
then plotting the full six-element set with `plot_keplerian_trajectory`,
shows how the orbit evolves under the full force model:

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

`plot_keplerian_trajectory` renders all six elements in a 2x3 grid: semi-major
axis, eccentricity, and inclination on the top row, RAAN, argument of
periapsis, and mean anomaly on the bottom row. RAAN and argument of periapsis
are plotted as `state_eci_to_koe_for_body` returns them - referenced to the
ICRF pole - so their absolute values don't match the RAAN and 270 degree
argument of periapsis used to design the orbit in the Mars-equatorial basis;
only their evolution over time is meaningful here. Inclination, by contrast,
is measured against Mars's spin pole (the same pole the orbit was designed
around), not the ICRF pole that `state_eci_to_koe_for_body` references, so it
is directly comparable to the 92.6 degree design value. Over this 2-day
window it stays within about 0.1 degree of that value: $J_2$ drives the
nodal precession that makes the orbit
sun-synchronous but produces no secular change in inclination, so the residual
motion is a bounded short-period oscillation rather than decay. Semi-major
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
