# Dawn at Ceres

In this example we'll define Ceres as a fully user-supplied central body and
fly an orbit inspired by the Dawn spacecraft's Low Altitude Mapping Orbit
(LAMO) around it. Unlike Earth, the Moon, and Mars, Ceres has no built-in
constants in brahe: no built-in [`CentralBody`](../library_api/propagators/force_model_config.md#central-body) variant, no named
inertial/fixed frame pair, no spin model. Its gravitational parameter and
radius are resolved from the JPL Small-Body Database (SBDB); its spin pole,
prime meridian, and body-fixed frame - which SBDB does not provide - are
supplied by the user via [`CentralBody.Custom`](../library_api/propagators/force_model_config.md#brahe.CentralBody.Custom) and
`register_custom_frame`. The same recipe applies to any body brahe doesn't
have built-in constants for: another dwarf planet, an asteroid, or a comet
nucleus.

NASA's Dawn spacecraft orbited Ceres from 2015 to 2018, spending much of its
final year in LAMO: a ~375 km, near-circular polar orbit used for its
highest-resolution gravity and neutron/gamma-ray mapping. This example is
inspired by that LAMO phase rather than reproducing its exact mission
parameters.

---

## Body Definition

Ceres's orientation model is an IAU-style pole (right ascension,
declination), a prime-meridian angle at J2000, and a spin rate, which
together define the rotation of the body-fixed frame at any epoch. SBDB
doesn't provide these, so they're set here from the IAU WGCCRE 2015 values.
The gravitational parameter and mean radius, which SBDB does provide, are
resolved separately below.

``` python
--8<-- "./examples/examples/dawn_ceres_orbit.py:preamble"
```

``` python
--8<-- "./examples/examples/dawn_ceres_orbit.py:body_definition"
```

## Resolving Ceres and Loading an Ephemeris

Ceres's NAIF/SPK ID and SI physical parameters - gravitational parameter and
mean radius - come from the JPL Small-Body Database (SBDB) rather than being
hardcoded. A targeted SPK covering the propagation span is then generated and
loaded from JPL Horizons, so the Sun/Jupiter third-body accelerations and the
solar radiation pressure model below have an ephemeris to resolve Ceres's
position against. Both the SBDB lookup and the Horizons SPK are cached under
the brahe cache directory, so the network is used only on the first run per
machine.

``` python
--8<-- "./examples/examples/dawn_ceres_orbit.py:body_resolution"
```

The returned SPK segment is centered on the Sun (NAIF ID 10), not the solar
system barycenter. Chaining it through the `de440s` kernel loaded above -
which does carry the Sun's position relative to the barycenter - is what lets
brahe resolve Ceres's position relative to the Sun, Jupiter, or any other
body with a loaded ephemeris.

## Body-Fixed Frame

`register_custom_frame` takes two callbacks: `rotation(epoch)`, returning the
3x3 ICRF-to-body-fixed direction cosine matrix, and an optional
`omega(epoch)`, returning the body-fixed angular velocity vector used for the
velocity transport term (if omitted, it's derived numerically from
`rotation` by central differencing). Here both are built from the standard
IAU pole/prime-meridian rotation

$$R = R_z(W) \, R_x\!\left(\frac{\pi}{2} - \delta\right) \, R_z\!\left(\frac{\pi}{2} + \alpha\right)$$

where $\alpha$, $\delta$ are the pole's ICRF right ascension and declination
and $W$ is the prime-meridian angle, which advances linearly with time at the
body's spin rate. The x-axis of the underlying equatorial basis (used below
to compute the orbit's initial state) is the ascending node of the body's
equator on the ICRF equator - the standard IAU orientation convention
([Archinal et al., 2018](https://doi.org/10.1007/s10569-017-9805-5)) - since
$\hat{z}_{\text{ICRF}} \times \hat{p}$ is perpendicular to both poles, hence
lies in both equatorial planes: the line of nodes. Once registered under an
integer key, `ReferenceFrame.BodyFixedCustom(naif_id, key)` is usable
anywhere a [`ReferenceFrame`](../library_api/frames/router.md#referenceframe) is accepted:

``` python
--8<-- "./examples/examples/dawn_ceres_orbit.py:body_fixed_frame"
```

!!! note "The angular-velocity callback drives the velocity transport term"
    `register_custom_frame` accepts an optional `omega(epoch)` callback
    returning the body-fixed angular velocity. When it is omitted, the rate is
    recovered by central-differencing `rotation(epoch)`, which costs extra
    rotation evaluations per query and is only as accurate as the
    finite-difference step. Supplying `omega` analytically - as this example
    does from the constant IAU spin rate - makes the velocity transport term
    exact and avoids that overhead. See
    [Reference Frame Router](../learn/frames/frame_transformations.md) and
    [Propagation Around Other Central Bodies](../learn/orbit_propagation/numerical_propagation/other_central_bodies.md).

## Force Model

`CentralBody.Custom` bundles the body's GM, radius, spin vector, and
body-fixed frame into the object [`ForceModelConfig`](../library_api/propagators/force_model_config.md#forcemodelconfig) propagates relative to.
ICGEM's only cataloged Ceres model, `sphericalRFM_CERES_2519`, is a
degree-2519 crustal forward-modeling research product - impractically large
to download for this purpose and not normalized to the body's true GM (its
$C_{0,0} \approx 0.126$, not the conventional 1.0) - so it isn't a drop-in
gravity field here. This example models Ceres's gravity as a point mass
instead, which is the standard starting point when defining your own body.

With the Ceres SPK loaded above, third-body and solar radiation pressure
perturbations resolve around Ceres. The Sun (`ThirdBody.SUN`) and Jupiter -
added as a `ThirdBody.Custom` barycenter (NAIF ID 5), since Jupiter has no
built-in `ThirdBody` variant - dominate the third-body signal at Ceres's
distance from the Sun. The SRP occulting body is Ceres itself, supplied as an
`OccultingBody.Custom`: an SRP shadow cast by Earth is meaningless 2.8 AU
away:

``` python
--8<-- "./examples/examples/dawn_ceres_orbit.py:force_model"
```

## Propagation

Dawn's LAMO is a ~375 km circular polar orbit. The propagator integrates in a
Ceres-centered inertial frame whose axes are ICRF-aligned, so its z-axis is
the ICRF pole - about 23 degrees from Ceres's spin pole. We use
[`state_koe_to_inertial_for_body`](../library_api/coordinates/cartesian.md#brahe.coordinates.state_koe_to_inertial_for_body) to reference the elements to the body's mean
equator at J2000 and compute the state in the Ceres-centered, ICRF-aligned
inertial frame. Because `ceres` is a `CentralBody.Custom` that carries a
body-fixed frame (`ceres_fixed`), the function reads Ceres's pole from that
registered frame - the same recipe works for any user-defined body - so the
90 degree inclination is placed against Ceres's equator with no manual basis
construction:

``` python
--8<-- "./examples/examples/dawn_ceres_orbit.py:orbit_setup"
```

``` python
--8<-- "./examples/examples/dawn_ceres_orbit.py:propagation"
```

[`state_bci`](../library_api/propagators/numerical_orbit_propagator.md#brahe.NumericalOrbitPropagator.state_bci) returns the propagator's native state in the central
body's body-centered inertial frame - here, Ceres-centered - the frame the
orbit is actually integrated in. Sampling it over the 2-day propagation
confirms the orbit stays in a tight band around the design altitude.
`state_eci` would instead re-center the state onto Earth via SPK ephemeris -
which now resolves, since the Ceres SPK is loaded - but that isn't the frame
wanted here. [`state_in_frame`](../library_api/propagators/numerical_orbit_propagator.md#brahe.NumericalOrbitPropagator.state_in_frame) converts the
final inertial state directly into the registered Ceres-fixed frame - the
same frame-router entry point used for the built-in `ITRF`/`LFPA`/`MCMF`
frames, but backed entirely by the callbacks registered above.

## Two-Body Baseline Comparison

To confirm the third-body and SRP perturbations are actually doing something
- rather than just adding integration noise - a second propagator is built
from the same initial state with a two-body (point-mass, no third-body, no
SRP) force model, and the two trajectories are compared over the same span:

``` python
--8<-- "./examples/examples/dawn_ceres_orbit.py:baseline_comparison"
```

At a ~375 km, ~5.4-hour LAMO orbit, the solar and Jovian third-body
accelerations and SRP are small enough that they perturb the trajectory by
tens of meters over 2 days (~9 revolutions) rather than reshaping it, but the
divergence from the two-body baseline is measurable and grows with time - a
signature a pure two-body run cannot reproduce.

## 3D Visualization

[`plot_trajectory_3d`](../library_api/plots/3d_trajectory.md) accepts `central_body="ceres"`: Ceres is already in the
plotting library's body-visuals registry (radius and NAIF ID for frame
validation, plus a default texture), so no custom dict is needed. Non-Earth
central bodies require the plotted trajectory to already be in
[`OrbitFrame.BodyCenteredInertial(naif_id)`](../library_api/orbits/enums.md#brahe.OrbitFrame.BodyCenteredInertial) for that body; a Ceres-centered
[`NumericalOrbitPropagator`](../library_api/propagators/numerical_orbit_propagator.md)'s `.trajectory` is already in that frame:

``` python
--8<-- "./examples/examples/dawn_ceres_orbit.py:plot_3d"
```

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/dawn_ceres_orbit_3d_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/dawn_ceres_orbit_3d_dark.html"  loading="lazy"></iframe>
</div>

*Body textures: [Solar System Scope](https://www.solarsystemscope.com/textures/), CC BY 4.0 - the Ceres texture is an artistic impression, not an actual surface map.*

## Full Code Example

??? "Full Code"

    ```python title="dawn_ceres_orbit.py"
    --8<-- "./examples/examples/dawn_ceres_orbit.py:all"
    ```

---

## See Also

- [Numerical Orbit Propagation](../learn/orbit_propagation/numerical_propagation/index.md) - Propagator fundamentals
- [Force Models](../learn/orbit_propagation/numerical_propagation/force_models.md) - Configuring force models, including `CentralBody.Custom`
- [Reference Frames](../learn/frames/index.md) - Frame conventions, including custom body-fixed frames
- [3D Trajectory Plotting](../learn/plots/3d_trajectory.md) - Advanced options for trajectory visualization, including non-Earth central bodies
