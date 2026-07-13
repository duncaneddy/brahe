# Reference Frame Router

`ReferenceFrame` is a single enum spanning every reference frame in Brahe &mdash; Earth-centered, lunar, Martian, barycentric, and generic NAIF-ID variants &mdash; and three router functions (`rotation_frame_to_frame`, `position_frame_to_frame`, `state_frame_to_frame`) that convert between *any* two of them. The router is the frame machinery underlying multibody numerical propagation; see [Cislunar and Lunar Propagation](../orbit_propagation/numerical_propagation/cislunar_lunar_propagation.md) and [Propagation Around Other Central Bodies](../orbit_propagation/numerical_propagation/other_central_bodies.md) for the propagation side.

## Available Frames

The **transformation source** column names how each frame's orientation is realized: `native` frames are computed from models compiled into Brahe (no kernel), `SPICE` frames are evaluated from a loaded NAIF kernel. Re-centering between frames with different origins is a separate step resolved through the loaded SPK kernels in the global registry (last-loaded-wins for overlapping segments); the default `de440s` ephemeris is loaded automatically when no SPK is resident.

| Frame | Kind | Transformation source |
|---|---|---|
| `GCRF` | Inertial | ICRF-aligned identity |
| `ITRF` | Earth-fixed | IAU 2006/2000A (native) |
| `EME2000` | Inertial | IAU 2006 frame bias (native) |
| `LCI` | Inertial | ICRF-aligned identity |
| `LFPA` | Moon-fixed | DE440 binary PCK (SPICE) |
| `LFME` | Moon-fixed | DE440 PA + constant PA&rarr;ME rotation (native) |
| `MCI` | Inertial | ICRF-aligned identity |
| `MCMF` | Mars-fixed | IAU/WGCCRE analytic (native) |
| `EMBI` | Inertial | ICRF-aligned identity |
| `SSBI` | Inertial | ICRF-aligned identity |
| `BodyCenteredICRF(naif_id)` | Inertial | ICRF-aligned identity |
| `BodyFixedIAU(naif_id)` | Body-fixed | IAU/WGCCRE analytic (native) |
| `BodyFixedPCK(center, frame_id)` | Body-fixed | Loaded binary PCK (SPICE) |
| `BodyFixedCustom(center, key)` | Body-fixed | User rotation callback (native) |

See [Lunar Reference Frames](lunar_frames.md) and [Mars Reference Frames](mars_frames.md) for `LCI`/`LFPA`/`LFME` and `MCI`/`MCMF` respectively.

## Router Functions

`rotation_frame_to_frame`, `position_frame_to_frame`, and `state_frame_to_frame` take a `from` frame, a `to` frame, and an epoch (plus a position/state vector for the latter two), and dispatch through a hub-and-spoke design: the source is rotated into ICRF axes centered on its own origin, re-centered to the target's origin if the two frames have different centers, and rotated into the target's axes. Same-frame calls short-circuit to an identity/no-op with no SPK query at all; same-center calls (e.g. `GCRF` &harr; `ITRF`, `LCI` &harr; `LFPA`) skip the re-centering step and are bit-identical to the underlying pairwise function.

The example below routes a `GCRF` (Earth-centered) state into `LCI` (Moon-centered) with `state_frame_to_frame` &mdash; a cross-center conversion that re-centers through the DE440 ephemeris &mdash; and takes a rotation-only `GCRF` &rarr; `MCMF` transform with `rotation_frame_to_frame`, which uses the compiled-in Mars model and needs no kernel.

=== "Python"

    ```python
    --8<-- "./examples/frames/frame_router.py:8"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/frames/frame_router.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/frames/frame_router.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/frames/frame_router.rs.txt"
        ```

## Generic NAIF-ID Variants

Four variants cover bodies without a dedicated named frame:

- **`BodyCenteredICRF(naif_id)`**: ICRF-aligned axes centered on `naif_id`. Requires an SPK segment for `naif_id` when translating to/from a differently-centered frame.
- **`BodyFixedIAU(naif_id)`**: the compiled-in IAU/WGCCRE body-fixed rotation for `naif_id`, centered on `naif_id` itself. Requires no kernel. `iau_rotation_model_ids()` returns the supported NAIF IDs: the Sun, Mercury, Venus, the Moon (a lower-precision IAU model, distinct from `LFPA`), Mars, the Galilean moons (Io/Europa/Ganymede/Callisto), Phobos, Deimos, Enceladus, Titan, Jupiter, Saturn, Uranus, and Neptune.
- **`BodyFixedPCK { center, frame_id }`**: a body-fixed frame evaluated from a loaded binary PCK's `frame_id` (e.g. 31008 for `MOON_PA_DE440`), centered on `center`. The PCK must already be loaded &mdash; the router never auto-loads a generic PCK.
- **`BodyFixedCustom { center, key }`**: a body-fixed frame evaluated from a user-supplied rotation callback registered under `key` with `register_custom_frame(key, rotation, omega=None)`. The callback maps an epoch to the ICRF&rarr;body-fixed rotation matrix; an optional angular-velocity callback supplies the velocity transport term (derived numerically by central differencing when omitted). This plugs in orientation models Brahe does not ship &mdash; e.g. an asteroid spin state &mdash; without any change to the router API. For a body with no catalogued NAIF ID, self-assign a unique negative `center`, mirroring NAIF's convention for non-catalogued objects.

## Kernel Requirements Per Frame

| Frame(s) | Kernel needed | Auto-loaded? |
|---|---|---|
| `GCRF`, `ITRF`, `EME2000` | None (SOFA-based) | N/A |
| `LCI` (rotation only) | None (ICRF-aligned) | N/A |
| `LCI`, `MCI`, `EMBI`, `SSBI` (translation to/from another center) | `de440s` SPK | Yes, on first `spk_*` query |
| `MCI`, `MCMF` (translation to/from another center) | `de440s` SPK + `mar099s` satellite ephemeris | Yes, on first Mars body-center query |
| `LFPA`, `LFME` | `moon_pa_de440` binary PCK | Yes, on first LCI &harr; LFPA/LFME conversion |
| `MCMF` (rotation only) | None (compiled-in WGCCRE polynomial) | N/A |
| `BodyFixedIAU(naif_id)` | None (compiled-in), if `naif_id` is in `iau_rotation_model_ids()` | N/A |
| `BodyFixedPCK { .. }` | The named binary PCK | No &mdash; must be loaded explicitly with `load_kernel` |
| `BodyFixedCustom { .. }` | None (user callback) | N/A |

The lunar PCK auto-load is a narrow exception to the general SPICE registry rule that binary PCKs are never auto-initialized (see [SPICE Kernels](../spice/index.md)); it exists because `LFPA`/`LFME` have no meaning without `moon_pa_de440` loaded, so every lunar body-fixed conversion loads it transparently on first use. `MCI`/`MCMF` are centered on the Mars body center (NAIF 499); a translation to or from another center resolves the body-center leg through the `mar099s` satellite ephemeris kernel, which is auto-loaded the same way.

## Propagator State in Any Frame

A `NumericalOrbitPropagator` integrates in its central body's inertial frame (`LCI` for `Moon`, `MCI` for `Mars`, `GCRF` for `Earth`, ...). `state_in_frame(epoch, frame)` converts the propagated state into any other `ReferenceFrame` by routing through `state_frame_to_frame(central_body.inertial_frame(), frame, epoch, x)` &mdash; for a Moon-centered propagator, `state_in_frame(epoch, ReferenceFrame.LCI)` is the identity (no SPK round trip), and `state_in_frame(epoch, ReferenceFrame.LFPA)` gives the Moon-fixed ground-track state directly, without first converting to Earth-centered `GCRF`. `state_bci(epoch)` returns the raw integrated body-centered inertial state with no frame conversion applied, and `state_bcbf(epoch)` returns the body-centered body-fixed state (the central body's fixed frame).

The `CentralBody` that determines each propagator's integration frame, and the `ForceModelConfig` defaults that pair a central body with an appropriate force model, are documented alongside the propagation examples in [Cislunar and Lunar Propagation](../orbit_propagation/numerical_propagation/cislunar_lunar_propagation.md) and [Propagation Around Other Central Bodies](../orbit_propagation/numerical_propagation/other_central_bodies.md).

## See Also

- [Lunar Reference Frames](lunar_frames.md)
- [Mars Reference Frames](mars_frames.md)
- [ECI &harr; ECEF Transformations](eci_ecef.md) - Earth-only frame transformations
- [Cislunar and Lunar Propagation](../orbit_propagation/numerical_propagation/cislunar_lunar_propagation.md) - `Moon`/`EMB` central bodies and force-model defaults
- [Propagation Around Other Central Bodies](../orbit_propagation/numerical_propagation/other_central_bodies.md) - `Mars`, custom bodies, and user-defined body-fixed frames
- [SPICE Kernels](../spice/index.md) - Kernel loading and the underlying SPK/PCK registry
- [Reference Frame Router API Reference](../../library_api/frames/router.md)
- [Force Model Configuration API Reference](../../library_api/propagators/force_model_config.md)
