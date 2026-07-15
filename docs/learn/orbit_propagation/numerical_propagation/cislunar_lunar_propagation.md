# Cislunar and Lunar Propagation

`NumericalOrbitPropagator` integrates relative to a `CentralBody`, which identifies the body an orbit is propagated about and bundles the gravitational parameter (GM, $\mu$), radius, spin rate, and inertial/fixed frame pair needed for body-orientation based force models (gravity, drag). Two central bodies cover cislunar work: `Moon`, for orbits about the Moon, and `EMB`, the Earth-Moon barycenter, whose integration frame is `EMBI`, the inertial (ICRF-aligned) Earth-Moon barycenter frame, for cislunar transfer trajectories. `ForceModelConfig.lunar_default()` and `ForceModelConfig.cislunar_default()` pair each with a physically appropriate force model.

| `CentralBody` | NAIF ID | Inertial frame | Fixed frame |
|---|---|---|---|
| `Moon` | 301 | `LCI` | `LFPA` |
| `EMB` | 3 | `EMBI` | none |

The propagator integrates in the central body's inertial frame (`LCI` for `Moon`, `EMBI` for `EMB`). `state_in_frame(frame, epoch)` converts the integrated state into any [`ReferenceFrame`](../../frames/frame_transformations.md), routing directly from the integration frame: for a Moon-centered propagator, `state_in_frame(ReferenceFrame.LCI, epoch)` is the identity (no SPK round trip), and `state_in_frame(ReferenceFrame.LFPA, epoch)` gives the Moon-fixed state without first converting to Earth-centered `GCRF`. `state_bci(epoch)` returns the raw body-centered inertial state without conversion, and `state_bcbf(epoch)` returns the body-centered body-fixed state (`LFPA` for `Moon`; `EMB` has no body-fixed frame and returns an error).

## Force-Model Defaults

`ForceModelConfig` provides factory methods that pair a `CentralBody` with a force model tuned for it:

| Constructor | Central body | Gravity | Drag | SRP occultation | Third bodies |
|---|---|---|---|---|---|
| `lunar_default()` | `Moon` | 50x50 GRGM660PRIM | none | Moon, Earth | Earth, Sun |
| `cislunar_default()` | `EMB` | Point mass | none | Earth, Moon | Earth, Moon, Sun |

Both require a spacecraft parameter vector at propagator construction, since mass/area/coefficients are wired up via `ParameterSource::ParameterIndex`. Each expects `[mass, drag_area, Cd, srp_area, Cr]`; because neither model includes drag, the `drag_area` and `Cd` slots (indices 1 and 2) are unused placeholders but must still be present. `.validate()` can be used on the config to check it up front for inconsistent configuration options. For example, with a `Moon` central body it fails on Earth-specific options (e.g. Harris-Priester or NRLMSISE-00 atmospheric drag attributed to the Moon, `EarthZonal` central gravity), and with the `EMB` barycenter it rejects spherical-harmonic central gravity and drag without an attributed body, since a barycenter has no mass or rotation of its own. The same validation also runs automatically when a `NumericalOrbitPropagator` is constructed, so a misconfigured force model fails at construction even without an explicit `.validate()` call.

## Barycenter Third-Body Physics

Third-body acceleration relative to a body-centered frame is normally a *differential* term: the direct attraction of the perturber on the spacecraft, minus its attraction on the (accelerating) central body, whose motion moves the frame origin. The Earth-Moon barycenter is a special case for its two internal bodies. Because internal Earth-Moon gravitational forces are equal and opposite, neither Earth nor the Moon can accelerate their mutual barycenter, so their contributions about `EMB` use the **direct term only** (no indirect subtraction). Other perturbers &mdash; the Sun, planets &mdash; do accelerate the Earth-Moon system as a whole and still use the differential form about `EMB`. (The Solar System barycenter `SSB` uses the direct form for every perturber, since nothing external to the modeled system accelerates it.)

This handling is automatic: selecting `CentralBody::EMB` (via `cislunar_default()` or directly) routes Earth and Moon perturbations through the direct form and all others through the differential form.

## Earth-Attributed Force Models

An EMB-centered trajectory that passes through low Earth altitudes can carry Earth-fidelity forces without switching central bodies. A third-body entry's `gravity` accepts a `SphericalHarmonic` or `EarthZonal` field evaluated at the object's position relative to that body, and `DragConfiguration.body` attributes atmospheric drag to a body other than the central body (density and relative wind are evaluated at the object's Earth-relative state). See [Force Models: Body Attribution](force_models.md#body-attribution) for the configuration details and a complete EMB-centered example with an 8x8 Earth field and Earth-attributed NRLMSISE-00 drag.

## Example: Propagating an Orbit About the Moon

The example below builds a lunar force model with `ForceModelConfig.lunar_default()`, propagates a low lunar orbit for six hours, and reports the initial and final state in the Moon-fixed LFPA frame via `state_in_frame`.

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/lunar_orbit.py:18"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/lunar_orbit.rs:15"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/numerical_propagation/lunar_orbit.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/numerical_propagation/lunar_orbit.rs.txt"
        ```

## See Also

- [Propagation Around Other Central Bodies](other_central_bodies.md) - Mars, custom bodies, and user-defined body-fixed frames
- [Lunar Reference Frames](../../frames/lunar_frames.md) - LCI/LFPA/LFME frame definitions
- [Reference Frame Router](../../frames/frame_transformations.md) - `state_in_frame` and cross-frame conversion
- [Force Models](force_models.md) - Building a `ForceModelConfig` from individual force terms
- [Third-Body Perturbations](../../orbital_dynamics/third_body.md) - Third-body acceleration model
- [Force Model Configuration API Reference](../../../library_api/propagators/force_model_config.md)
