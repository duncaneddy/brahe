# Propagation Around Other Central Bodies

Beyond Earth and the cislunar bodies, `NumericalOrbitPropagator` integrates about any body a `CentralBody` can describe. Mars has a dedicated built-in variant with a `mars_default()` force model; every other body is reached through `CentralBody.from_naif_id` (for bodies in an embedded table) or `CentralBody.Custom` (for anything else, including bodies with no catalogued NAIF ID). Bodies without SPICE orientation data can still be given a body-fixed frame through a user-supplied rotation callback.

## Mars

`CentralBody::Mars` integrates in the Mars-Centered Inertial (`MCI`) frame, centered on the Mars body center (NAIF ID 499), and reports body-fixed states in `MCMF`.

| `CentralBody` | NAIF ID | Inertial frame | Fixed frame |
|---|---|---|---|
| `Mars` | 499 | `MCI` | `MCMF` |

`ForceModelConfig.mars_default()` pairs it with a Mars-tuned force model:

| Constructor | Central body | Gravity | Drag | SRP occultation | Third bodies |
|---|---|---|---|---|---|
| `mars_default()` | `Mars` | 50x50 GGM2B (`ggm2bc80`) | Exponential | Mars | Sun |

`mars_default()` requires a spacecraft parameter vector `[mass, drag_area, Cd, srp_area, Cr]` at propagator construction; all five slots are used. Call `.validate()` to confirm the configuration is compatible with the central body before propagating.

### Example: Propagating an Orbit About Mars

The example below builds a Mars force model with `ForceModelConfig.mars_default()`, propagates a low Mars orbit for six hours, and reports the initial and final state in the Mars-fixed MCMF frame via `state_in_frame`.

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/mars_orbit.py:18"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/mars_orbit.rs:15"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/numerical_propagation/mars_orbit.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/numerical_propagation/mars_orbit.rs.txt"
        ```

## Selecting a Central Body by NAIF ID

`CentralBody.from_naif_id(naif_id)` constructs one of the five built-in variants for their NAIF IDs (`399` &rarr; `Earth`, `301` &rarr; `Moon`, `4` or `499` &rarr; `Mars`, `3` &rarr; `EMB`, `0` &rarr; `SSB`), or a pre-populated `Custom` body for a fixed table of other commonly used bodies: Mercury, Venus, Jupiter, Saturn, Uranus, Neptune, the Sun, Phobos, Deimos, Enceladus, and Titan. Each table entry carries the body's GM and radius, and its `fixed_frame` is set to `BodyFixedIAU(naif_id)` when the body has a compiled-in IAU/WGCCRE rotation model (true for every body in the table). A NAIF ID outside this set returns an error &mdash; use `CentralBody.Custom` directly for those.

## User-Defined Central Bodies

For a body outside the built-in table, construct `CentralBody.Custom(name, naif_id, gm, radius=None, omega=None, fixed_frame=None)` with the body's physical properties. For a body without a catalogued NAIF ID (e.g. a newly observed asteroid), self-assign a unique **negative** `naif_id`, mirroring NAIF's own convention for non-catalogued objects. The ID is used for frame identity and force-model validation; ephemeris queries against it will surface an SPK lookup error unless a kernel covering that ID is loaded.

Pair a custom body with a force model using `ForceModelConfig.for_body(central_body, gravity, drag=None, srp=None, third_body=None, relativity=False, mass=None)`, which fills in the frame-transformation default so callers specify only the terms that vary per body. `for_body` does not validate its result &mdash; call `.validate()` afterward to confirm the chosen options are compatible with the central body (for example, spherical-harmonic gravity requires the body to have a `fixed_frame`).

## Body-Fixed Frames Without SPICE Data

A custom body that has no SPICE orientation kernel and no compiled-in IAU model can still be given a rotating body-fixed frame. Register a rotation callback with `register_custom_frame(key, rotation, omega=None)`: `rotation` maps an `Epoch` to the ICRF&rarr;body-fixed rotation matrix, and the optional `omega` callback supplies the frame's angular velocity for the velocity transport term (derived numerically by central differencing when omitted). Reference the registered frame as `ReferenceFrame.BodyFixedCustom(center, key)`, using the same `key`, and set it as the custom body's `fixed_frame`. This supports orientation models Brahe does not ship &mdash; e.g. an asteroid spin state from the DAMIT database &mdash; without any change to the propagator or router API. See [Generic NAIF-ID Variants](../../frames/frame_transformations.md#generic-naif-id-variants) for the frame-router side.

## See Also

- [Cislunar and Lunar Propagation](cislunar_lunar_propagation.md) - `Moon`/`EMB` central bodies and barycenter third-body physics
- [Mars Reference Frames](../../frames/mars_frames.md) - MCI/MCMF frame definitions
- [Reference Frame Router](../../frames/frame_transformations.md) - `ReferenceFrame`, `BodyFixedCustom`, and `register_custom_frame`
- [Force Models](force_models.md) - Building a `ForceModelConfig` from individual force terms
- [Force Model Configuration API Reference](../../../library_api/propagators/force_model_config.md)
