# GCRF ↔ TEME ↔ ITRF Transformations

TEME is the true equator, mean equinox of date frame in which SGP4 expresses its output. Brahe defines it from the ITRF through Greenwich mean sidereal time on the IAU 1982 model, and it is available both as pairwise functions and through the frame router as `CelestialFrame.TEME`.

## Reference Frame

TEME is defined by the relation

$$[\mathrm{ITRF}] = W \, R_3(\mathrm{GMST}_{82}) \, [\mathrm{TEME}]$$

where $W$ is polar motion and $\mathrm{GMST}_{82}$ is Greenwich mean sidereal time on the IAU 1982 model evaluated on UT1. This is the convention of [Vallado et al., *Revisiting Spacetrack Report #3*](https://celestrak.org/publications/AIAA/2006-6753/AIAA-2006-6753-Rev3.pdf), Appendix C, which fixes the meaning of the frame SGP4 produces. Combined with the CIO-based chain

$$[\mathrm{ITRF}] = W \, R_3(\mathrm{ERA}) \, C \, [\mathrm{GCRF}]$$

from the [SOFA C transformation cookbook](https://www.iausofa.org/s/sofa_pn_c.pdf), the rotation from the GCRF is

$$[\mathrm{TEME}] = R_3(\mathrm{ERA} - \mathrm{GMST}_{82}) \, C \, [\mathrm{GCRF}]$$

where $C$ is the bias-precession-nutation matrix used by [GCRF ↔ ITRF Transformations](gcrf_itrf.md). The TEME equator is therefore the true equator of date. Its origin of right ascension is the mean equinox implied by the IAU 1982 sidereal time model, which differs from the IAU 2006 mean equinox of [MOD](equinox_frames.md) by the precession-model offset and from the true equinox of TOD by the equation of the equinoxes. The rotation from TEME to TOD is $R_3(\mathrm{GMST}_{82} - \mathrm{GAST})$, where $\mathrm{GAST}$ is Greenwich apparent sidereal time (GAST), evaluated by the frame router.

## Velocities

TEME is treated as non-rotating relative to the GCRF: its precession and nutation rates are below $10^{-11}$ rad/s, so state transforms between GCRF and TEME rotate position and velocity by the same matrix. The TEME to ITRF transformation includes the Earth rotation transport term, exactly as the GCRF to ITRF transformation does.

## SGP4 Propagator

`SGPPropagator.state` returns the raw SGP4 output in TEME. The configured output frame, `GCRF` by default, applies to `propagate_to`, the stored trajectory, and `current_state`/`initial_state`. The frame-specific accessors `state_gcrf` and `state_itrf` always return the GCRF and ITRF states respectively, using the pairwise TEME transforms, regardless of the configured output frame. `state_in_frame` likewise always returns the requested frame, converting the TEME output through the frame router.

## GCRF to TEME

### State Vector

Transform a complete state vector (position and velocity) from GCRF to TEME:

=== "Python"

    ``` python
    --8<-- "./examples/frames/gcrf_to_teme_state.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/gcrf_to_teme_state.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/frames/gcrf_to_teme_state.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/frames/gcrf_to_teme_state.rs.txt"
        ```

## TEME to ITRF

### State Vector

Transform a TEME state vector from an SGP4 propagator into the ITRF:

=== "Python"

    ``` python
    --8<-- "./examples/frames/teme_to_itrf_state.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/teme_to_itrf_state.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/frames/teme_to_itrf_state.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/frames/teme_to_itrf_state.rs.txt"
        ```

## SGP4 Output in TOD

`state_in_frame` converts the raw TEME output of `SGPPropagator.state` into any other router frame, here TOD, and can be compared against the `GCRF` accessor:

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/sgp_propagation/state_in_tod.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/sgp_propagation/state_in_tod.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/orbit_propagation/sgp_propagation/state_in_tod.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/orbit_propagation/sgp_propagation/state_in_tod.rs.txt"
        ```

## References

- [SOFA C Transformation Cookbook](https://www.iausofa.org/s/sofa_pn_c.pdf)
- Vallado, D. A. et al., 2006, [*Revisiting Spacetrack Report #3*](https://celestrak.org/publications/AIAA/2006-6753/AIAA-2006-6753-Rev3.pdf), AIAA 2006-6753-Rev3, Appendix C
- Vallado, D. A., *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 3.7

## See Also

- [GCRF ↔ ITRF Transformations](gcrf_itrf.md)
- [GCRF ↔ MOD ↔ TOD Transformations](equinox_frames.md)
- [Reference Frame Router](frame_transformations.md)
- [Reference Frames Overview](index.md)
