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

`mars_default()` requires a spacecraft parameter vector `[mass, drag_area, Cd, srp_area, Cr]` at propagator construction; all five slots are used. Configuration validation runs automatically when the propagator is constructed; `.validate()` can also be called directly to check a configuration earlier.

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

Pair a custom body with a force model using `ForceModelConfig.for_body(central_body, gravity, drag=None, srp=None, third_bodies=None, relativity=False, mass=None)`, which fills in the frame-transformation default so callers specify only the terms that vary per body. Validation runs automatically when the propagator is constructed, rejecting options incompatible with the central body (for example, spherical-harmonic gravity requires the body to have a `fixed_frame`); `.validate()` can also be called directly to check a configuration earlier.

## Body-Fixed Frames Without SPICE Data

A custom body that has no SPICE orientation kernel and no compiled-in IAU model can still be given a rotating body-fixed frame. Register a rotation callback with `register_custom_frame(key, rotation, omega=None)`:

- `rotation` is a callback function: it receives an `Epoch` and returns the 3x3 ICRF&rarr;body-fixed rotation matrix at that instant.
- `omega` is an optional second callback function: it receives an `Epoch` and returns the frame's angular velocity vector (rad/s), used for the exact velocity transport term. When omitted, the angular velocity is derived numerically by central differencing of `rotation`.
- `key` is an arbitrary integer handle that names the registered callback. It does **not** need to match (and is unrelated to) the central body's NAIF ID; the body's NAIF ID appears separately as the `center` of the frame variant.

Reference the registered frame as `ReferenceFrame.BodyFixedCustom(center, key)`, using the same `key`, and set it as the custom body's `fixed_frame`. This enables support for user-defined orientation models, enabling the framework to extend to additional bodies without needing hard-coded support for them in the library. See [Generic NAIF-ID Variants](../../frames/frame_transformations.md#generic-naif-id-variants) for the frame-router side.

### Example: Registering a Custom Body-Fixed Frame

The example below registers a uniform spin state for an uncatalogued body (self-assigned negative NAIF ID), converts a state between the body's inertial and body-fixed frames, and verifies that a co-rotating surface point is stationary in the body-fixed frame. No kernels are required.

=== "Python"

    ``` python
    --8<-- "./examples/frames/custom_frame.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/custom_frame.rs:5"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/frames/custom_frame.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/frames/custom_frame.rs.txt"
        ```

## See Also

- [Cislunar and Lunar Propagation](cislunar_lunar_propagation.md) - `Moon`/`EMB` central bodies and barycenter third-body physics
- [Mars Reference Frames](../../frames/mars_frames.md) - MCI/MCMF frame definitions
- [Reference Frame Router](../../frames/frame_transformations.md) - `ReferenceFrame`, `BodyFixedCustom`, and `register_custom_frame`
- [Force Models](force_models.md) - Building a `ForceModelConfig` from individual force terms
- [Force Model Configuration API Reference](../../../library_api/propagators/force_model_config.md)
