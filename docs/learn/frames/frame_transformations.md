# Reference Frame Router and Multibody Propagation

`ReferenceFrame` is a single enum spanning every reference frame in Brahe &mdash; Earth-centered, lunar, Martian, barycentric, and generic NAIF-ID variants &mdash; and three router functions (`rotation_frame_to_frame`, `position_frame_to_frame`, `state_frame_to_frame`) that convert between *any* two of them. `CentralBody` and `ForceModelConfig`'s `lunar_default()`/`mars_default()`/`cislunar_default()` constructors extend numerical propagation to bodies other than Earth, using the same frame machinery.

## Available Frames

| Frame | Center (NAIF ID) | Kind |
|---|---|---|
| `GCRF` | Earth (399) | Inertial |
| `ITRF` | Earth (399) | Earth-fixed |
| `EME2000` | Earth (399) | Inertial |
| `LCI` | Moon (301) | Inertial |
| `LFPA` | Moon (301) | Moon-fixed |
| `LFME` | Moon (301) | Moon-fixed |
| `MCI` | Mars system barycenter (4) | Inertial |
| `MCMF` | Mars system barycenter (4) | Mars-fixed |
| `EMBI` | Earth-Moon barycenter (3) | Inertial |
| `SSBI` | Solar System barycenter (0) | Inertial |
| `BodyCenteredICRF(naif_id)` | `naif_id` | Inertial |
| `BodyFixedIAU(naif_id)` | `naif_id` | Body-fixed |
| `BodyFixedPCK(center, frame_id)` | `center` | Body-fixed |

See [Lunar Reference Frames](lunar_frames.md) and [Mars Reference Frames](mars_frames.md) for `LCI`/`LFPA`/`LFME` and `MCI`/`MCMF` respectively.

## Router Functions

`rotation_frame_to_frame`, `position_frame_to_frame`, and `state_frame_to_frame` take a `from` frame, a `to` frame, and an epoch (plus a position/state vector for the latter two), and dispatch through a hub-and-spoke design: the source is rotated into ICRF axes centered on its own origin, re-centered to the target's origin if the two frames have different centers, and rotated into the target's axes. Same-frame calls short-circuit to an identity/no-op with no SPK query at all; same-center calls (e.g. `GCRF` &harr; `ITRF`, `LCI` &harr; `LFPA`) skip the re-centering step and are bit-identical to the underlying pairwise function.

=== "Python"

    ```python
    import numpy as np
    import brahe as bh

    epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    x_gcrf = np.array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0])

    r = bh.rotation_frame_to_frame(bh.ReferenceFrame.MCI, bh.ReferenceFrame.MCMF, epc)
    x_lfpa = bh.state_frame_to_frame(bh.ReferenceFrame.GCRF, bh.ReferenceFrame.LFPA, epc, x_gcrf)
    ```

=== "Rust"

    ```rust
    use brahe as bh;

    let epc = bh::Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let x_gcrf = bh::SVector6::new(1e7, 2e7, 3e7, 1.0, 2.0, 3.0);

    let r = bh::rotation_frame_to_frame(bh::ReferenceFrame::MCI, bh::ReferenceFrame::MCMF, epc).unwrap();
    let x_lfpa = bh::state_frame_to_frame(bh::ReferenceFrame::GCRF, bh::ReferenceFrame::LFPA, epc, x_gcrf).unwrap();
    ```

## Generic NAIF-ID Variants

Three variants cover bodies without a dedicated named frame:

- **`BodyCenteredICRF(naif_id)`**: ICRF-aligned axes centered on `naif_id`. Requires an SPK segment for `naif_id` when translating to/from a differently-centered frame.
- **`BodyFixedIAU(naif_id)`**: the compiled-in IAU/WGCCRE body-fixed rotation for `naif_id`, centered on `naif_id` itself. Requires no kernel. `iau_rotation_model_ids()` returns the supported NAIF IDs: the Sun, Mercury, Venus, the Moon (a lower-precision IAU model, distinct from `LFPA`), Mars, the Galilean moons (Io/Europa/Ganymede/Callisto), Phobos, Deimos, Enceladus, Titan, Jupiter, Saturn, Uranus, and Neptune.
- **`BodyFixedPCK { center, frame_id }`**: a body-fixed frame evaluated from a loaded binary PCK's `frame_id` (e.g. 31008 for `MOON_PA_DE440`), centered on `center`. The PCK must already be loaded &mdash; the router never auto-loads a generic PCK.

!!! warning "`BodyFixedIAU(499)` (Mars) is centered on the body, not the barycenter"

    For bodies without a direct SPK ephemeris segment for their own body ID (Mars, NAIF 499, is the motivating case &mdash; `de440s` only carries its barycenter, NAIF 4), a translation into or out of `BodyFixedIAU(499)` from a differently-centered frame surfaces the underlying SPK lookup error. Use `ReferenceFrame::MCMF` (or another named, barycenter-anchored frame) for translated Mars-system conversions; `BodyFixedIAU(499)` remains exact for *rotation-only* queries, which dispatch identically to `MCMF`.

## Kernel Requirements Per Frame

| Frame(s) | Kernel needed | Auto-loaded? |
|---|---|---|
| `GCRF`, `ITRF`, `EME2000` | None (SOFA-based) | N/A |
| `LCI` (rotation only) | None (ICRF-aligned) | N/A |
| `LCI`, `MCI`, `EMBI`, `SSBI` (translation to/from another center) | `de440s` SPK | Yes, on first `spk_*` query |
| `LFPA`, `LFME` | `moon_pa_de440` binary PCK | Yes, on first LCI &harr; LFPA/LFME conversion |
| `MCMF` | None (compiled-in WGCCRE polynomial) | N/A |
| `BodyFixedIAU(naif_id)` | None (compiled-in), if `naif_id` is in `iau_rotation_model_ids()` | N/A |
| `BodyFixedPCK { .. }` | The named binary PCK | No &mdash; must be loaded explicitly with `load_kernel` |

The lunar PCK auto-load is a narrow exception to the general SPICE registry rule that binary PCKs are never auto-initialized (see [SPICE Kernels](../spice/index.md)); it exists because `LFPA`/`LFME` have no meaning without `moon_pa_de440` loaded, so every lunar body-fixed conversion loads it transparently on first use.

## Central-Body Propagation Defaults

`CentralBody` identifies which body a `NumericalOrbitPropagator` integrates relative to, and bundles the GM, radius, spin rate, and inertial/fixed frame pair a force model needs:

| `CentralBody` | NAIF ID | Inertial frame | Fixed frame |
|---|---|---|---|
| `Earth` | 399 | `GCRF` | `ITRF` |
| `Moon` | 301 | `LCI` | `LFPA` |
| `Mars` | 4 | `MCI` | `MCMF` |
| `EMB` | 3 | `EMBI` | none |
| `SSB` | 0 | `SSBI` | none |
| `Custom(CustomBody)` | user-defined | user-defined | optional |

`CentralBody::from_naif_id` (`CentralBody.from_naif_id` in Python) constructs one of the five built-in variants for their NAIF IDs (accepting either 4 or 499 for Mars), or a pre-populated `Custom` body for a fixed table of other commonly used bodies (Mercury, Venus, Jupiter, Saturn, Uranus, Neptune, the Sun, Phobos, Deimos, Enceladus, Titan).

`ForceModelConfig` provides three factory methods that pair a `CentralBody` with a physically appropriate force model:

| Constructor | Central body | Gravity | Drag | SRP occultation | Third bodies |
|---|---|---|---|---|---|
| `lunar_default()` | `Moon` | 50x50 GRGM660PRIM | none | Moon, Earth | Earth, Sun |
| `mars_default()` | `Mars` | 50x50 GGM2B (`ggm2bc80`) | Exponential | Mars | Sun |
| `cislunar_default()` | `EMB` | Point mass | none | Earth, Moon | Earth, Moon, Sun |

Each of these requires a spacecraft parameter vector at propagator construction, since mass/area/coefficients are wired up via `ParameterSource::ParameterIndex`: `lunar_default()`/`mars_default()` expect `[mass, drag_area, Cd, srp_area, Cr]` (the drag slots are unused, but must still be present, for `lunar_default()`, since it has no drag model). Call `.validate()` on the resulting `ForceModelConfig` to check it up front &mdash; it rejects Earth-specific options (e.g. Harris-Priester/NRLMSISE-00 drag, `EarthZonal` gravity) on non-Earth bodies, and rejects spherical-harmonic gravity or drag on barycenters (`EMB`/`SSB`), which have no mass or rotation of their own.

For a body without a dedicated `CentralBody` variant, build a custom force model with `ForceModelConfig.for_body(central_body, gravity, drag=None, srp=None, third_body=None, relativity=False, mass=None)`, passing a `CentralBody::Custom`/`CentralBody.Custom(...)`.

### `state_in_frame`

A `NumericalOrbitPropagator` integrates in its central body's inertial frame (`LCI` for `Moon`, `MCI` for `Mars`, `GCRF` for `Earth`, ...). `state_in_frame(epoch, frame)` converts the propagated state into any other `ReferenceFrame` by routing through `state_frame_to_frame(central_body.inertial_frame(), frame, epoch, x)` &mdash; for a Moon-centered propagator, `state_in_frame(epoch, ReferenceFrame.LCI)` is the identity (no SPK round trip), and `state_in_frame(epoch, ReferenceFrame.LFPA)` gives the Moon-fixed ground-track state directly, without first converting to Earth-centered `GCRF`. `state_central_inertial(epoch)` returns the raw integrated state with no frame conversion applied at all.

See the [Lunar Orbit](lunar_frames.md#example-propagating-an-orbit-about-the-moon) and [Mars Orbit](mars_frames.md#example-propagating-an-orbit-about-mars) examples for `state_in_frame` used end-to-end with `lunar_default()`/`mars_default()`.

## See Also

- [Lunar Reference Frames](lunar_frames.md)
- [Mars Reference Frames](mars_frames.md)
- [ECI &harr; ECEF Transformations](eci_ecef.md) - Earth-only frame transformations
- [SPICE Kernels](../spice/index.md) - Kernel loading and the underlying SPK/PCK registry
- [Reference Frame Router API Reference](../../library_api/frames/router.md)
- [Force Model Configuration API Reference](../../library_api/propagators/force_model_config.md)
