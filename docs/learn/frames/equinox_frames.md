# GCRF ↔ MOD ↔ TOD Transformations

MOD and TOD are the classical equinox-based Earth frames. Brahe defines them from the GCRF using the IAU 2000 precession and IAU 2000B nutation models together with the loaded Earth orientation data, and they are available both as pairwise functions and through the frame router as `CelestialFrame.MOD` / `CelestialFrame.TOD`.

## Reference Frames

### Mean Equator and Equinox of Date (MOD)

MOD is defined by the mean equator and mean equinox of date: the frame bias and precession from the GCRF, with no nutation applied. It is the intermediate frame `P B [GCRF]` in the SOFA cookbook Appendix p. A4 summary table, and corresponds to the mean-of-date frame in NASA TP-20220014814.

### True Equator and Equinox of Date (TOD)

TOD is defined by the true equator and true equinox of date: nutation applied on top of MOD, giving the intermediate frame `N P B [GCRF]` in the same cookbook table. TOD is the frame in which the classical equation of the equinoxes and Greenwich apparent sidereal time are defined.

## Relationship to the CIO-Based Chain

The SOFA cookbook gives two equivalent factorizations of the transformation between the GCRF and the ITRF:

```
CIO based:      [ITRF] = W  R3(ERA)   C    [GCRF]
equinox based:  [ITRF] = W  R3(GAST)  N P B [GCRF]
```

`W` is polar motion, `C` is the CIO-based bias-precession-nutation matrix used by [GCRF ↔ ITRF Transformations](gcrf_itrf.md), and `N`, `P`, `B` are the classical nutation, precession, and frame bias matrices of the equinox chain. Brahe evaluates both chains on the IAU 2000/2000B model basis and computes Greenwich apparent sidereal time (GAST) as `ERA` minus the equation of the origins taken from the same combined nutation-precession-bias matrix used to reach TOD. As a result, converting a state from GCRF to ITRF through TOD agrees with the direct GCRF to ITRF transformation at the microarcsecond level.

## Frame Bias and Earth Orientation Corrections

The frame bias between GCRF and the classical J2000.0 mean equator and equinox is applied explicitly inside the bias-precession matrix that defines GCRF to MOD. The IERS dX/dY celestial pole offsets are residuals relative to this already-biased IAU 2000 model, so no bias is double-counted; they enter the nutation step after conversion to dPsi/dEps corrections following SOFA cookbook Section 5.4. This differs from the IAU 1976/1980 convention, where the IERS dPsi/dEps corrections are measured against the unbiased model and therefore absorb the bias themselves, so no separate bias matrix is applied. TOD computed from the two conventions differs at the sub-milliarcsecond level, which is immaterial for interpreting data products labeled TOD.

## Velocities

MOD and TOD are treated as non-rotating relative to the GCRF: their precession and nutation rates are below $10^{-11}$ rad/s, under $10^{-4}$ m/s in low Earth orbit, so state transforms among GCRF, MOD, and TOD rotate position and velocity by the same matrix. This differs from EME2000, whose bias rotation relative to the GCRF is exactly fixed; MOD and TOD rotate slowly relative to the GCRF, and that rate is neglected here. The TOD to ITRF transformation includes the Earth rotation transport term, exactly as the GCRF to ITRF transformation does.

## GCRF to TOD

### State Vector

Transform a complete state vector (position and velocity) from GCRF to TOD:

=== "Python"

    ``` python
    --8<-- "./examples/frames/gcrf_to_tod_state.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/gcrf_to_tod_state.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/frames/gcrf_to_tod_state.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/frames/gcrf_to_tod_state.rs.txt"
        ```

### Rotation Matrix

Get the GCRF to TOD rotation matrix and compare it with the CIO-based bias-precession-nutation matrix:

=== "Python"

    ``` python
    --8<-- "./examples/frames/gcrf_to_tod_rotation.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/gcrf_to_tod_rotation.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/frames/gcrf_to_tod_rotation.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/frames/gcrf_to_tod_rotation.rs.txt"
        ```

## TOD to GCRF

### State Vector

Transform a complete state vector (position and velocity) from TOD to GCRF:

=== "Python"

    ``` python
    --8<-- "./examples/frames/tod_to_gcrf_state.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/tod_to_gcrf_state.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/frames/tod_to_gcrf_state.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/frames/tod_to_gcrf_state.rs.txt"
        ```

## GCRF to MOD

### Rotation Matrix

Get the GCRF to MOD rotation matrix and confirm it reduces to the EME2000 frame bias at J2000.0, where the IAU 2000 precession is identity:

=== "Python"

    ``` python
    --8<-- "./examples/frames/gcrf_to_mod_rotation.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/gcrf_to_mod_rotation.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/frames/gcrf_to_mod_rotation.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/frames/gcrf_to_mod_rotation.rs.txt"
        ```

## MOD to TOD

### Rotation Matrix

Get the MOD to TOD nutation matrix and recover the nutation angle it represents:

=== "Python"

    ``` python
    --8<-- "./examples/frames/mod_to_tod_rotation.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/mod_to_tod_rotation.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/frames/mod_to_tod_rotation.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/frames/mod_to_tod_rotation.rs.txt"
        ```

## TOD to ITRF

### State Vector

Transform a state vector from TOD to ITRF directly, and compare it with the GCRF-mediated path:

=== "Python"

    ``` python
    --8<-- "./examples/frames/tod_to_itrf_state.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/tod_to_itrf_state.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/frames/tod_to_itrf_state.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/frames/tod_to_itrf_state.rs.txt"
        ```

## References

- [SOFA C Transformation Cookbook](https://www.iausofa.org/s/sofa_pn_c.pdf), Sections 2.7-2.9, 3.1-3.2, 3.5-3.6, 4.1, 5.4, and Appendix p. A4
- IERS Conventions (2010), IERS Technical Note 36, Chapter 5
- NASA TP-20220014814, *Astrodynamics Convention and Modeling Reference for Lunar, Cislunar, and Libration Point Orbits*, Section 4.3.5
- Wallace, P. T. & Capitaine, N., 2006, A&A 459, 981

## See Also

- [GCRF ↔ ITRF Transformations](gcrf_itrf.md)
- [EME2000 ↔ GCRF Transformations](eme2000_gcrf.md)
- [Reference Frame Router](frame_transformations.md)
- [Reference Frames Overview](index.md)
