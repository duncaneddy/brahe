# Mars Reference Frames

Brahe provides two Mars-centered reference frames: **MCI** (inertial) and **MCMF** (body-fixed). MCI is used as the integration frame for Mars-centered propagation; MCMF expresses a state relative to Mars's rotating surface.

## MCI (Mars-Centered Inertial)

MCI axes are ICRF-aligned (treated as equivalent to J2000, consistent with the rest of Brahe). No kernel is required for the MCI orientation itself. Converting between MCI and ECI requires Mars's position relative to Earth, looked up from the `de440s` SPK kernel (auto-loaded on first use).

!!! warning "MCI is centered on the Mars system barycenter, not the Mars body center"

    The IAU/WGCCRE rotation model used by MCMF is defined for the Mars body center (NAIF ID 499). However, `state_eci_to_mci`/`state_mci_to_eci` (and the position-only equivalents) use the ephemeris state of the **Mars system barycenter** (NAIF ID 4) relative to Earth, because the bundled `de440s` kernel provides no direct ephemeris segment for body 499 &mdash; only for its barycenter. `CentralBody::Mars.naif_id()` and `ReferenceFrame::MCI`/`MCMF` both use NAIF ID 4 for this reason.

    The offset between the Mars barycenter and the Mars body center is at the centimeter level, dominated by the barycentric motion of Phobos and Deimos about their common center of mass with Mars. This is negligible for orbit propagation and most other uses, but is documented here for completeness. Use `OccultingBody::Mars.naif_id()` (499, the physical body) when the *shadow-casting* radius matters (e.g. eclipse modeling), and `OccultingBody::Mars.naif_position_id()` (4) when querying its ephemeris position.

## MCMF (Mars-Centered Mars-Fixed)

MCMF is the body-fixed frame defined by the IAU Working Group on Cartographic Coordinates and Rotational Elements (WGCCRE) pole and prime-meridian model for Mars (NAIF ID 499). Unlike the Moon's LFPA frame, no external kernel is required &mdash; the WGCCRE polynomial model is compiled directly into Brahe (see [`rotation_icrf_to_body_fixed_iau`](frame_transformations.md#generic-naif-id-variants) and [`iau_rotation_model_ids`](frame_transformations.md#generic-naif-id-variants)).

## Function Reference

| Conversion | Rust / Python function |
|---|---|
| MCI &rarr; MCMF | `rotation_mci_to_mcmf`, `position_mci_to_mcmf`, `state_mci_to_mcmf` |
| MCMF &rarr; MCI | `rotation_mcmf_to_mci`, `position_mcmf_to_mci`, `state_mcmf_to_mci` |
| ECI &harr; MCI | `position_eci_to_mci`/`position_mci_to_eci`, `state_eci_to_mci`/`state_mci_to_eci` |

All rotation functions return a 3x3 direction cosine matrix; all `position_*`/`state_*` functions take and return SI units (m, m/s). See the [Mars Frames API Reference](../../library_api/frames/mars.md) for full signatures.

## Example: Propagating an Orbit About Mars

The example below builds a Mars force model with [`ForceModelConfig.mars_default()`](frame_transformations.md#central-body-propagation-defaults), propagates a low Mars orbit for six hours, and reports the initial and final state in the Mars-fixed MCMF frame via `state_in_frame`.

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

## See Also

- [Lunar Reference Frames](lunar_frames.md) - LCI/LFPA/LFME frames
- [Reference Frame Router and Multibody Propagation](frame_transformations.md) - `ReferenceFrame`, kernel requirements, and central-body propagation defaults
- [Mars Frames API Reference](../../library_api/frames/mars.md)
