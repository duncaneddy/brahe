# MRO Mars Orbit

In this example we'll set up an MRO-like sun-synchronous science orbit and
propagate it with brahe's Mars force model. The Mars Reconnaissance Orbiter
(MRO) has flown a sun-synchronous, near-polar science orbit since 2006,
imaging the surface at a consistent local solar time on every pass
([Zurek & Smrekar, 2007](https://doi.org/10.1029/2006JE002701)); this example
uses an MRO-like ~255 x 320 km, 92.6 degree inclination orbit. We'll track
how the osculating orbital elements evolve under the full force model over a
two-day propagation, then visualize the trajectory in 3D around a textured
Mars.

---

## Why Sun-Synchronous Orbits Work at Mars

A sun-synchronous orbit keeps its orbital plane at a fixed orientation
relative to the Sun, so the spacecraft crosses each latitude at the same
local solar time on every pass - valuable for consistent lighting in surface
imagery. This only works because a planet's oblateness (the $J_2$ zonal
harmonic) causes the orbital plane to precess at a rate

$$
\dot{\Omega} = -\frac{3}{2} n J_2 \left(\frac{R}{p}\right)^2 \cos i
$$

where $n$ is the orbit's mean motion, $R$ is the planet's radius, $p =
a(1-e^2)$ is the semi-latus rectum, and $i$ is the inclination. For a
near-polar, slightly retrograde inclination, this precession rate can be
tuned to exactly match the planet's mean motion around the Sun. Setting
$\dot{\Omega}$ equal to Mars's heliocentric mean motion (about 0.524 deg/day,
from its 687-day year) and solving for $i$ at this example's semi-major axis
and eccentricity, with Mars's $J_2 = 1.96045 \times 10^{-3}$ and $R =
R_{\text{Mars}}$, gives $i \approx 92.6$ degrees - close to polar, with the
small excess over 90 degrees giving just enough retrograde precession to
track Mars's slower year. This example's 255 x 320 km, 92.6 degree orbit is
designed around this resonance.

## Orbit Setup

The propagator integrates in the Mars-Centered Inertial (MCI) frame, whose
axes are ICRF-aligned: the MCI z-axis is the ICRF pole, which sits about 37
degrees from Mars's spin pole. Passing 92.6 degrees straight to
[`state_koe_to_eci`](../library_api/coordinates/cartesian.md#brahe.coordinates.state_koe_to_eci) would measure the inclination against the ICRF pole, not
Mars's equator. [`state_koe_to_inertial_for_body`](../library_api/coordinates/cartesian.md#brahe.coordinates.state_koe_to_inertial_for_body) instead references the
elements to Mars's mean equator at J2000 - the plane normal to the Mars IAU
pole, with the x-axis on the ascending node of that equator on the ICRF equator
- and returns the state directly in MCI. It takes a [`CentralBody`](../library_api/propagators/force_model_config.md#central-body), which
supplies both Mars's gravitational parameter and its pole, so the orbit is
placed against the Mars equator with no manual basis construction.

The ascending node is still derived from the Sun's direction rather than
hardcoded: the right ascension of the ascending node is set to the Sun's right
ascension in the Mars-equatorial plane plus 45 degrees, placing the node near a
15:00 (mid-afternoon) local solar time. The Sun direction is projected onto the
equatorial basis vectors (the ascending node and its quadrature axis) evaluated
at J2000, the same reference plane the elements are referenced to:

``` python
--8<-- "./examples/examples/mro_mars_orbit.py:preamble"
```

With the standard preamble in place, the next step sets up the sun-synchronous orbit geometry.

``` python
--8<-- "./examples/examples/mro_mars_orbit.py:orbit_setup"
```

## Propagation

We propagate under [`ForceModelConfig.mars_default()`](../library_api/propagators/force_model_config.md#brahe.ForceModelConfig.mars_default): 50x50 GMM-2B gravity,
exponential atmospheric drag, SRP occulted by Mars, and Sun third-body
perturbations. The propagator integrates in the Mars-Centered Inertial (MCI)
frame:

``` python
--8<-- "./examples/examples/mro_mars_orbit.py:propagation"
```

## Element Evolution

Sampling the trajectory over the 2-day propagation and converting each
Cartesian state back to Keplerian elements with [`state_inertial_to_koe_for_body`](../library_api/coordinates/cartesian.md#brahe.coordinates.state_inertial_to_koe_for_body),
then plotting the full six-element set with [`plot_keplerian_trajectory`](../library_api/plots/orbital_trajectories.md#brahe.plot_keplerian_trajectory),
shows how the orbit evolves under the full force model:

``` python
--8<-- "./examples/examples/mro_mars_orbit.py:element_history"
```

``` python
--8<-- "./examples/examples/mro_mars_orbit.py:element_plot"
```

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/mro_mars_orbit_elements_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/mro_mars_orbit_elements_dark.html"  loading="lazy"></iframe>
</div>

`plot_keplerian_trajectory` renders all six elements in a 2x3 grid: semi-major
axis, eccentricity, and inclination on the top row, RAAN, argument of
periapsis, and mean anomaly on the bottom row. Because
`state_inertial_to_koe_for_body` references all six elements to Mars's mean
equator at J2000 - the same plane the orbit was designed in - the inclination,
RAAN, and argument of periapsis are directly comparable to their design values
(92.6 degrees, the sun-synchronous node, and 270 degrees). The inclination
stays within about 0.1 degree of 92.6 degrees over this 2-day window: $J_2$
drives the nodal precession that makes the orbit sun-synchronous but produces
no secular change in inclination, so the residual motion is a bounded
short-period oscillation rather than decay. Semi-major
axis and eccentricity likewise show short-period oscillation without net
secular decay over this timespan, and periapsis altitude stays comfortably
above the Mars surface throughout.

## 3D Visualization

[`plot_trajectory_3d`](../library_api/plots/3d_trajectory.md) accepts `central_body="mars"` to render an interactive
3D view of the trajectory around a textured Mars. Non-Earth central bodies
require the plotted trajectory to already be in
[`OrbitFrame.BodyCenteredInertial(naif_id)`](../library_api/orbits/enums.md#brahe.OrbitFrame.BodyCenteredInertial) for that body; a Mars-centered
[`NumericalOrbitPropagator`](../library_api/propagators/numerical_orbit_propagator.md)'s `.trajectory` is already in that frame, so no
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

??? "Full Code"

    ```python title="mro_mars_orbit.py"
    --8<-- "./examples/examples/mro_mars_orbit.py:all"
    ```

---

## See Also

- [Numerical Orbit Propagation](../learn/orbit_propagation/numerical_propagation/index.md) - Propagator fundamentals
- [Force Models](../learn/orbit_propagation/numerical_propagation/force_models.md) - Configuring force models, including `mars_default()`
- [3D Trajectory Plotting](../learn/plots/3d_trajectory.md) - Advanced options for trajectory visualization, including non-Earth central bodies
