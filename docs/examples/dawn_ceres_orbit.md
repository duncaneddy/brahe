# Dawn at Ceres

In this example we'll define Ceres as a fully user-supplied central body and
fly an orbit inspired by the Dawn spacecraft's Low Altitude Mapping Orbit
(LAMO) around it. Unlike Earth, the Moon, and Mars, Ceres has no built-in
constants in brahe: no built-in [`CentralBody`](../library_api/propagators/force_model_config.md#central-body) variant, no named
inertial/fixed frame pair, no spin model. Everything about Ceres here - its
gravitational parameter, radius, spin pole, prime meridian, and body-fixed
frame - is supplied by the user via [`CentralBody.Custom`](../library_api/propagators/force_model_config.md#brahe.CentralBody.Custom) and
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

Ceres's physical model is four numbers plus a spin rate: a gravitational
parameter, a mean radius, and an IAU-style pole (right ascension,
declination) and prime-meridian angle that together define the orientation
and rotation of the body-fixed frame at any epoch.

``` python
--8<-- "./examples/examples/dawn_ceres_orbit.py:preamble"
```

``` python
--8<-- "./examples/examples/dawn_ceres_orbit.py:body_definition"
```

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
gravity field here. This example models Ceres as a point mass instead, which
is the standard starting point when defining your own body. It also omits
third-body perturbations: the DE kernels brahe loads carry no Ceres
ephemeris. A Ceres SPK can be generated through
[JPL Horizons](https://ssd-api.jpl.nasa.gov/doc/horizons.html), which would
enable third-body perturbations, and a Horizons client is planned
([issue #402](https://github.com/duncaneddy/brahe/issues/402)); this example
keeps gravity-only for self-containment:

``` python
--8<-- "./examples/examples/dawn_ceres_orbit.py:force_model"
```

!!! note "A custom body has no ephemeris until you supply one"
    Third-body perturbations require the perturbing body's position from a
    loaded SPK kernel. The DE kernels brahe ships carry no Ceres ephemeris, so
    a `CentralBody.Custom` defined this way can be neither perturbed by nor
    perturb other bodies until a Ceres SPK is loaded. Point-mass, gravity-only
    propagation is therefore the self-contained starting point for a
    user-defined body; adding third-body forces means first generating and
    loading a Ceres SPK.

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

Sampling the trajectory over the 2-day propagation confirms the orbit stays
in a tight band around the design altitude, and [`state_in_frame`](../library_api/propagators/numerical_orbit_propagator.md#brahe.NumericalOrbitPropagator.state_in_frame) converts the
final inertial state directly into the registered Ceres-fixed frame - the
same frame-router entry point used for the built-in `ITRF`/`LFPA`/`MCMF`
frames, but backed entirely by the callbacks registered above.

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
