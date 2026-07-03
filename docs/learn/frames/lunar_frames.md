# Lunar Reference Frames

Brahe provides three Moon-centered reference frames: **LCI** (inertial), **LFPA** (body-fixed, principal-axis), and **LFME** (body-fixed, mean-Earth/polar-axis). LCI is used as the integration frame for lunar-centered propagation; LFPA and LFME express a state relative to the Moon's rotating surface.

## LCI (Lunar-Centered Inertial)

LCI axes are ICRF-aligned (treated as equivalent to J2000, consistent with the rest of Brahe) and centered on the Moon (NAIF ID 301). No kernel is required for the LCI orientation itself, since it shares axes with GCRF. Converting between LCI and ECI requires the Moon's position relative to Earth, which is looked up from the `de440s` SPK kernel (auto-loaded on first use of any of the functions in the [Function Reference](#function-reference) below, or via `initialize_ephemeris`).

## LFPA (Lunar-Fixed Principal Axis)

LFPA is the DE440 lunar principal-axis frame (`MOON_PA_DE440`, NAIF frame class ID 31008). It is evaluated live from the binary PCK `moon_pa_de440`, which is downloaded and cached automatically the first time any LCI &harr; LFPA/LFME function is called — no explicit `load_kernel` call is required for this specific frame pair (see [Kernel Requirements](frame_transformations.md#kernel-requirements-per-frame) for the general rule this is an exception to).

## LFME (Lunar-Fixed Mean-Earth/Polar-Axis)

LFME is the "mean Earth/polar axis" frame, in which the Moon's mean pole and mean prime meridian (nominally facing Earth) are aligned with the frame axes. It differs from LFPA by a small **constant** rotation of about 104 arcseconds total (roughly 875 m of surface displacement at the lunar radius), transcribed from NAIF's lunar frames kernel `moon_de440_220930.tf`:

$$
R_{\text{LFME} \to \text{LFPA}} = R_z(67.8526'') \, R_y(78.6944'') \, R_x(0.2785'')
$$

Because this rotation is constant, `rotation_lfme_to_lfpa`/`rotation_lfpa_to_lfme` take no epoch argument, unlike every other rotation in this module.

!!! info "PA vs. ME: which one should I use?"

    LFPA is the frame the DE440 lunar orientation kernel defines directly and is the appropriate choice when working with other DE440-derived products (e.g. NAIF lunar surface data). LFME is the frame most lunar cartographic products (e.g. LOLA topography, landing site coordinates) are published in. Convert explicitly between them with `rotation_lfpa_to_lfme`/`rotation_lfme_to_lfpa` rather than assuming they are interchangeable — the ~875 m offset is significant for surface operations.

## Function Reference

| Conversion | Rust / Python function |
|---|---|
| LCI &rarr; LFPA | `rotation_lci_to_lfpa`, `position_lci_to_lfpa`, `state_lci_to_lfpa` |
| LFPA &rarr; LCI | `rotation_lfpa_to_lci`, `position_lfpa_to_lci`, `state_lfpa_to_lci` |
| LCI &rarr; LFME | `rotation_lci_to_lfme`, `position_lci_to_lfme`, `state_lci_to_lfme` |
| LFME &rarr; LCI | `rotation_lfme_to_lci`, `position_lfme_to_lci`, `state_lfme_to_lci` |
| LFPA &harr; LFME | `rotation_lfpa_to_lfme`, `rotation_lfme_to_lfpa` (constant, no epoch) |
| ECI &harr; LCI | `position_eci_to_lci`/`position_lci_to_eci`, `state_eci_to_lci`/`state_lci_to_eci` |

All rotation functions return a 3x3 direction cosine matrix; all `position_*`/`state_*` functions take and return SI units (m, m/s). See the [Lunar Frames API Reference](../../library_api/frames/lunar.md) for full signatures.

## Example: Propagating an Orbit About the Moon

The example below builds a lunar force model with [`ForceModelConfig.lunar_default()`](frame_transformations.md#central-body-propagation-defaults), propagates a low lunar orbit for six hours, and reports the initial and final state in the Moon-fixed LFPA frame via `state_in_frame`.

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

- [Mars Reference Frames](mars_frames.md) - MCI/MCMF frames and the Mars barycenter caveat
- [Reference Frame Router and Multibody Propagation](frame_transformations.md) - `ReferenceFrame`, kernel requirements, and central-body propagation defaults
- [Lunar Frames API Reference](../../library_api/frames/lunar.md)
