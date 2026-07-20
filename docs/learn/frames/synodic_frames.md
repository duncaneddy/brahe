# Synodic Reference Frames

Brahe provides three synodic (two-body rotating) reference frames from NASA TP-20220014814[^tp]: **EMR** (Earth-Moon Rotating), **SER** (Sun-Earth Rotating), and **GSE** (Geocentric Solar Ecliptic). Synodic frames rotate with the line between two primary bodies and are the natural frames for cislunar trajectory analysis, libration point missions, and comparing trajectories to circular restricted three-body problem (CR3BP) solutions.

All three frames share the same axis construction from the relative position $\boldsymbol{r}_{12}$ and velocity $\boldsymbol{v}_{12}$ of the secondary with respect to the primary:

$$
\hat{\boldsymbol{x}} = \frac{\boldsymbol{r}_{12}}{\|\boldsymbol{r}_{12}\|}, \qquad
\hat{\boldsymbol{z}} = \frac{\boldsymbol{r}_{12} \times \boldsymbol{v}_{12}}{\|\boldsymbol{r}_{12} \times \boldsymbol{v}_{12}\|}, \qquad
\hat{\boldsymbol{y}} = \hat{\boldsymbol{z}} \times \hat{\boldsymbol{x}}
$$

The velocity transformation uses the exact time derivative of the rotation matrix (the GTDS/STK convention in TP-20220014814 §4.6.1), including the $d\hat{\boldsymbol{z}}/dt$ term evaluated from the relative acceleration $\boldsymbol{a}_{12}$, which Brahe computes by analytically differentiating the SPK ephemeris Chebyshev polynomials (see `spk_acceleration`).

## EMR (Earth-Moon Rotating)

Primaries: Earth → Moon. Origin: the Earth-Moon barycenter (NAIF ID 3). The Moon lies permanently on the $+\hat{\boldsymbol{x}}$ axis, the Earth on $-\hat{\boldsymbol{x}}$. This is the standard frame for cislunar trajectory visualization and Earth-Moon libration point analysis.

## SER (Sun-Earth Rotating)

Primaries: Sun → Earth. Origin: the Sun-Earth barycenter. The SEB has no NAIF ID or SPK ephemeris entry; Brahe computes it as the $GM$-weighted combination of the Sun and Earth SPK states and identifies it internally by the synthetic center ID `SUN_EARTH_BARYCENTER_ID`. The Earth lies on $+\hat{\boldsymbol{x}}$, the Sun ~450 km from the origin on $-\hat{\boldsymbol{x}}$.

## GSE (Geocentric Solar Ecliptic)

Origin: Earth center. $\hat{\boldsymbol{x}}$ points from the Earth to the Sun — the **reversed** sense relative to SER — and $\hat{\boldsymbol{z}}$ is normal to the instantaneous ecliptic plane (~23.44° from the GCRF $z$-axis). GSE is common in space-weather and magnetospheric work. Because GSE is Earth-centered, converting between GCRF and GSE involves no translation.

## Generic Synodic Frames

EMR, SER, and GSE are named instances of a generic two-body synodic frame, `ReferenceFrame.Synodic(origin, primary, secondary)`. `origin` is a `SynodicOrigin` (`Primary`, `Secondary`, or `Barycenter`); `primary` and `secondary` are the NAIF IDs of the two bodies, and any NAIF ID is accepted for any origin. For a `Barycenter` origin, the pair is encoded into a synthetic negative center ID as `primary * 1000 + secondary`. Both bodies must have packaged $GM$ constants. IDs outside that range still work but produce a different encoding that no longer maps back to a synthetic center — this surfaces as an SPK/GM lookup error at transform time rather than a silent collision. The axis construction is identical to the formula above, with $\boldsymbol{r}_{12}$ and $\boldsymbol{v}_{12}$ taken between `primary` and `secondary`.

The three named frames are equivalent to these generic configurations:

$$
\text{EMR} \equiv \texttt{Synodic(Barycenter, 399, 301)} \\
\text{SER} \equiv \texttt{Synodic(Barycenter, 10, 399)} \\
\text{GSE} \equiv \texttt{Synodic(Primary, 399, 10)}
$$

A `Barycenter` origin is computed analytically as the $GM$-weighted combination of the primary and secondary SPK states — the same scheme used for the Sun-Earth barycenter above — rather than read from a dedicated SPK ephemeris entry. Both bodies must therefore have packaged $GM$ constants (the Sun, the planets, the planetary barycenters, and the Moon). For the Earth-Moon pair, this computed barycenter differs from the SPK Earth-Moon Barycenter (NAIF ID 3) by well under a meter.

The origin, primary, and secondary of any synodic frame (`Synodic`, `EMR`, `SER`, or `GSE`) are recoverable via the `frame.synodic_origin`, `frame.synodic_primary`, and `frame.synodic_secondary` properties, which return `None` for non-synodic frames.

The generic frame has no dedicated `rotation_gcrf_to_synodic`-style functions; it is used through the frame router (`rotation_frame_to_frame`, `position_frame_to_frame`, `state_frame_to_frame`). The following example transforms a low Earth orbit state from GCRF into a Sun-Mars rotating frame:

=== "Python"

    ```python
    --8<-- "./examples/frames/generic_synodic_frame.py:9"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/frames/generic_synodic_frame.rs:6"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/frames/generic_synodic_frame.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/frames/generic_synodic_frame.rs.txt"
        ```

## Function Reference

| Conversion | Function |
|---|---|
| GCRF &rarr; EMR | [`rotation_gcrf_to_emr`](../../library_api/frames/synodic.md#brahe.rotation_gcrf_to_emr), [`position_gcrf_to_emr`](../../library_api/frames/synodic.md#brahe.position_gcrf_to_emr), [`state_gcrf_to_emr`](../../library_api/frames/synodic.md#brahe.state_gcrf_to_emr) |
| EMR &rarr; GCRF | [`rotation_emr_to_gcrf`](../../library_api/frames/synodic.md#brahe.rotation_emr_to_gcrf), [`position_emr_to_gcrf`](../../library_api/frames/synodic.md#brahe.position_emr_to_gcrf), [`state_emr_to_gcrf`](../../library_api/frames/synodic.md#brahe.state_emr_to_gcrf) |
| GCRF &rarr; SER | [`rotation_gcrf_to_ser`](../../library_api/frames/synodic.md#brahe.rotation_gcrf_to_ser), [`position_gcrf_to_ser`](../../library_api/frames/synodic.md#brahe.position_gcrf_to_ser), [`state_gcrf_to_ser`](../../library_api/frames/synodic.md#brahe.state_gcrf_to_ser) |
| SER &rarr; GCRF | [`rotation_ser_to_gcrf`](../../library_api/frames/synodic.md#brahe.rotation_ser_to_gcrf), [`position_ser_to_gcrf`](../../library_api/frames/synodic.md#brahe.position_ser_to_gcrf), [`state_ser_to_gcrf`](../../library_api/frames/synodic.md#brahe.state_ser_to_gcrf) |
| GCRF &rarr; GSE | [`rotation_gcrf_to_gse`](../../library_api/frames/synodic.md#brahe.rotation_gcrf_to_gse), [`position_gcrf_to_gse`](../../library_api/frames/synodic.md#brahe.position_gcrf_to_gse), [`state_gcrf_to_gse`](../../library_api/frames/synodic.md#brahe.state_gcrf_to_gse) |
| GSE &rarr; GCRF | [`rotation_gse_to_gcrf`](../../library_api/frames/synodic.md#brahe.rotation_gse_to_gcrf), [`position_gse_to_gcrf`](../../library_api/frames/synodic.md#brahe.position_gse_to_gcrf), [`state_gse_to_gcrf`](../../library_api/frames/synodic.md#brahe.state_gse_to_gcrf) |

All three frames are also available through the frame router as `ReferenceFrame.EMR`, `ReferenceFrame.SER`, and `ReferenceFrame.GSE`, usable in `rotation_frame_to_frame`, `position_frame_to_frame`, `state_frame_to_frame`, and every provider's `state_in_frame`/`states_in_frame`. The `de440s` SPK kernel is auto-loaded on first use.

## See Also

- [Lunar Reference Frames](lunar_frames.md) - LCI/LFPA/LFME frames
- [Reference Frame Router](frame_transformations.md) - `ReferenceFrame`, kernel requirements, and cross-frame conversion
- [Synodic Frames API Reference](../../library_api/frames/synodic.md)

[^tp]: Folta, D., Bosanac, N., Elliott, I., Mann, L., Mesarch, R., & Rosales, J. (2022). [*Astrodynamics Convention and Modeling Reference for Lunar, Cislunar, and Libration Point Orbits*, NASA/TP-20220014814](https://ntrs.nasa.gov/api/citations/20220014814/downloads/NASA%20TP%2020220014814%20final.pdf), §2.5 and §4.6.
