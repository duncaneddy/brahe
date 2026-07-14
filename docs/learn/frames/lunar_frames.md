# Lunar Reference Frames

Brahe provides three Moon-centered reference frames: **LCI** (inertial), **LFPA** (body-fixed, principal-axis), and **LFME** (body-fixed, mean-Earth/polar-axis). LCI is used as the integration frame for lunar-centered propagation; LFPA and LFME express a state relative to the Moon's rotating surface. LFPA is the lunar principal-axis frame realized directly by the JPL DE440 ephemeris[^park2021]; LFME is the mean-Earth/polar-axis frame that most lunar cartographic products are published in.

## LCI (Lunar-Centered Inertial)

LCI axes are ICRF-aligned (treated as equivalent to J2000, consistent with the rest of Brahe) and centered on the Moon (NAIF ID 301). No kernel is required for the LCI orientation itself, since it shares axes with GCRF. Converting between LCI and ECI requires the Moon's position relative to Earth, which is looked up from the `de440s` SPK kernel (auto-loaded on first use of any of the functions in the [Function Reference](#function-reference) below, or via `initialize_ephemeris`).

## LFPA (Lunar-Fixed Principal Axis)

LFPA is the DE440 lunar principal-axis frame[^park2021], distributed by NAIF as the binary PCK `moon_pa_de440` (frame class ID 31008). It is evaluated live from that kernel, which is downloaded and cached automatically the first time any LCI &harr; LFPA/LFME function is called &mdash; no explicit `load_kernel` call is required for this specific frame pair (see [Kernel Requirements](frame_transformations.md#kernel-requirements-per-frame) for the general rule this is an exception to).

## LFME (Lunar-Fixed Mean-Earth/Polar-Axis)

LFME is the "mean Earth/polar axis" frame, in which the Moon's mean pole and mean prime meridian (nominally facing Earth) are aligned with the frame axes. It is derived from LFPA by a small **constant** rotation[^lsdc]:

$$
R_{\text{LFME} \to \text{LFPA}} = R_z(67.8526'') \, R_y(78.6944'') \, R_x(0.2785'')
$$

Because this rotation is constant, `rotation_lfme_to_lfpa`/`rotation_lfpa_to_lfme` take no epoch argument, unlike every other rotation in this module.

!!! info "PA vs. ME: which one should I use?"

    LFPA is the frame the DE440 lunar orientation kernel defines directly and is the appropriate choice when working with other DE440-derived products (e.g. NAIF lunar surface data). LFME is the frame most lunar cartographic products (e.g. LOLA topography, landing site coordinates) are published in. Convert explicitly between them with `rotation_lfpa_to_lfme`/`rotation_lfme_to_lfpa` rather than assuming they are interchangeable.

## Function Reference

| Conversion | Function |
|---|---|
| LCI &rarr; LFPA | [`rotation_lci_to_lfpa`](../../library_api/frames/lunar.md#brahe.rotation_lci_to_lfpa), [`position_lci_to_lfpa`](../../library_api/frames/lunar.md#brahe.position_lci_to_lfpa), [`state_lci_to_lfpa`](../../library_api/frames/lunar.md#brahe.state_lci_to_lfpa) |
| LFPA &rarr; LCI | [`rotation_lfpa_to_lci`](../../library_api/frames/lunar.md#brahe.rotation_lfpa_to_lci), [`position_lfpa_to_lci`](../../library_api/frames/lunar.md#brahe.position_lfpa_to_lci), [`state_lfpa_to_lci`](../../library_api/frames/lunar.md#brahe.state_lfpa_to_lci) |
| LCI &rarr; LFME | [`rotation_lci_to_lfme`](../../library_api/frames/lunar.md#brahe.rotation_lci_to_lfme), [`position_lci_to_lfme`](../../library_api/frames/lunar.md#brahe.position_lci_to_lfme), [`state_lci_to_lfme`](../../library_api/frames/lunar.md#brahe.state_lci_to_lfme) |
| LFME &rarr; LCI | [`rotation_lfme_to_lci`](../../library_api/frames/lunar.md#brahe.rotation_lfme_to_lci), [`position_lfme_to_lci`](../../library_api/frames/lunar.md#brahe.position_lfme_to_lci), [`state_lfme_to_lci`](../../library_api/frames/lunar.md#brahe.state_lfme_to_lci) |
| LFPA &harr; LFME | [`rotation_lfpa_to_lfme`](../../library_api/frames/lunar.md#brahe.rotation_lfpa_to_lfme), [`rotation_lfme_to_lfpa`](../../library_api/frames/lunar.md#brahe.rotation_lfme_to_lfpa) (constant, no epoch) |
| ECI &harr; LCI | [`position_eci_to_lci`](../../library_api/frames/lunar.md#brahe.position_eci_to_lci)/[`position_lci_to_eci`](../../library_api/frames/lunar.md#brahe.position_lci_to_eci), [`state_eci_to_lci`](../../library_api/frames/lunar.md#brahe.state_eci_to_lci)/[`state_lci_to_eci`](../../library_api/frames/lunar.md#brahe.state_lci_to_eci) |

All rotation functions return a 3x3 direction cosine matrix; all `position_*`/`state_*` functions take and return SI units (m, m/s). See the [Lunar Frames API Reference](../../library_api/frames/lunar.md) for full signatures.

For propagating an orbit about the Moon and reporting it in the LFPA frame, see [Cislunar and Lunar Propagation](../orbit_propagation/numerical_propagation/cislunar_lunar_propagation.md).

## See Also

- [Mars Reference Frames](mars_frames.md) - MCI/MCMF frames
- [Reference Frame Router](frame_transformations.md) - `ReferenceFrame`, kernel requirements, and cross-frame conversion
- [Cislunar and Lunar Propagation](../orbit_propagation/numerical_propagation/cislunar_lunar_propagation.md) - lunar force-model defaults and a worked propagation example
- [Lunar Frames API Reference](../../library_api/frames/lunar.md)

[^park2021]: Park, R. S., Folkner, W. M., Williams, J. G., & Boggs, D. H. (2021). The JPL Planetary and Lunar Ephemerides DE440 and DE441. *The Astronomical Journal*, 161(3), 105. The DE440 lunar orientation defines the principal-axis (PA) frame; the mean-Earth/polar-axis (ME) frame and the constant PA&rarr;ME rotation are given in NAIF's DE440 lunar frames kernel `moon_de440_220930.tf`.

[^lsdc]: Folta, D., Bosanac, N., Elliott, I., Mann, L., Mesarch, R., & Rosales, J. (2022). [*Astrodynamics Convention and Modeling Reference for Lunar, Cislunar, and Libration Point Orbits*, NASA/TP-20220014814](https://ntrs.nasa.gov/api/citations/20220014814/downloads/NASA%20TP%2020220014814%20final.pdf), Equation 52, which gives the constant PA &rarr; ME frame-bias matrix $[\boldsymbol{B}_M] = [\boldsymbol{R}_1(-0.2785'')][\boldsymbol{R}_2(-78.6944'')][\boldsymbol{R}_3(-67.8526'')]$ &mdash; the transpose of the LFME &rarr; LFPA rotation above. The same angles appear in NAIF's DE440 lunar frames kernel `moon_de440_220930.tf`.
