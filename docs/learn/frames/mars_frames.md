# Mars Reference Frames

Brahe provides two Mars-centered reference frames: **MCI** (inertial) and **MCMF** (body-fixed). MCI is used as the integration frame for Mars-centered propagation; MCMF expresses a state relative to Mars's rotating surface using the IAU/WGCCRE pole and prime-meridian model[^archinal2018].

## MCI (Mars-Centered Inertial)

MCI axes are ICRF-aligned (treated as equivalent to J2000, consistent with the rest of Brahe) and centered on the Mars body center (NAIF ID 499). No kernel is required for the MCI orientation itself. Converting between MCI and ECI requires Mars's position relative to Earth: the bundled `de440s` SPK kernel carries only the Mars system barycenter (NAIF ID 4), so the body-center leg from the barycenter to NAIF 499 is resolved through the `mar099s` satellite ephemeris kernel, which is auto-loaded on first use alongside `de440s`.

## MCMF (Mars-Centered Mars-Fixed)

MCMF is the body-fixed frame defined by the IAU Working Group on Cartographic Coordinates and Rotational Elements (WGCCRE) pole and prime-meridian model for Mars[^archinal2018]. Unlike the Moon's LFPA frame, no external kernel is required &mdash; the WGCCRE polynomial model is compiled directly into Brahe (see [`rotation_icrf_to_body_fixed_iau`](frame_transformations.md#generic-naif-id-variants) and [`iau_rotation_model_ids`](frame_transformations.md#generic-naif-id-variants)). It is equivalent to the `IAU_MARS` frame realized by SPICE.

## Function Reference

| Conversion | Function |
|---|---|
| MCI &rarr; MCMF | [`rotation_mci_to_mcmf`](../../library_api/frames/mars.md#brahe.rotation_mci_to_mcmf), [`position_mci_to_mcmf`](../../library_api/frames/mars.md#brahe.position_mci_to_mcmf), [`state_mci_to_mcmf`](../../library_api/frames/mars.md#brahe.state_mci_to_mcmf) |
| MCMF &rarr; MCI | [`rotation_mcmf_to_mci`](../../library_api/frames/mars.md#brahe.rotation_mcmf_to_mci), [`position_mcmf_to_mci`](../../library_api/frames/mars.md#brahe.position_mcmf_to_mci), [`state_mcmf_to_mci`](../../library_api/frames/mars.md#brahe.state_mcmf_to_mci) |
| ECI &harr; MCI | [`position_eci_to_mci`](../../library_api/frames/mars.md#brahe.position_eci_to_mci)/[`position_mci_to_eci`](../../library_api/frames/mars.md#brahe.position_mci_to_eci), [`state_eci_to_mci`](../../library_api/frames/mars.md#brahe.state_eci_to_mci)/[`state_mci_to_eci`](../../library_api/frames/mars.md#brahe.state_mci_to_eci) |

All rotation functions return a 3x3 direction cosine matrix; all `position_*`/`state_*` functions take and return SI units (m, m/s). See the [Mars Frames API Reference](../../library_api/frames/mars.md) for full signatures.

For propagating an orbit about Mars and reporting it in the MCMF frame, see [Propagation Around Other Central Bodies](../orbit_propagation/numerical_propagation/other_central_bodies.md).

## See Also

- [Lunar Reference Frames](lunar_frames.md) - LCI/LFPA/LFME frames
- [Reference Frame Router](frame_transformations.md) - `ReferenceFrame`, kernel requirements, and cross-frame conversion
- [Propagation Around Other Central Bodies](../orbit_propagation/numerical_propagation/other_central_bodies.md) - Mars force-model defaults and a worked propagation example
- [Mars Frames API Reference](../../library_api/frames/mars.md)

[^archinal2018]: Archinal, B. A., Acton, C. H., A'Hearn, M. F., et al. (2018). Report of the IAU Working Group on Cartographic Coordinates and Rotational Elements: 2015. *Celestial Mechanics and Dynamical Astronomy*, 130(3), 22. Brahe's embedded Mars pole/prime-meridian polynomial is taken from this report and matches SPICE's `IAU_MARS` realization.
