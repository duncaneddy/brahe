# Precession-Nutation Model

Brahe evaluates the IAU 2006/2000A precession-nutation model by default. The truncated IAU 2000B model, IAU 2000 precession with the abridged IAU 2000B nutation series, is selectable with a single global setting. The setting applies to every subsequent transformation that depends on the orientation of the Celestial Intermediate Pole: the [GCRF ↔ ITRF](gcrf_itrf.md) and [ECI ↔ ECEF](eci_ecef.md) transformations, the [GCRF ↔ MOD ↔ TOD](equinox_frames.md) equinox chain, the [reference frame router](frame_transformations.md), the [batch forms](vectorized.md), and the Earth-rotation-dependent force models used during propagation.

## Querying and Setting the Model

`get_precession_nutation_model` returns the model in effect and `set_precession_nutation_model` replaces it. The setting is crate-wide and persists until it is changed again, so a program that switches models temporarily should restore the previous value when it is done.

=== "Python"

    ``` python
    --8<-- "./examples/frames/precession_nutation_model.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/precession_nutation_model.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/frames/precession_nutation_model.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/frames/precession_nutation_model.rs.txt"
        ```

## Accuracy and Speed

IAU 2000B evaluates the Celestial Intermediate Pole about eight times faster than IAU 2006/2000A, because its nutation series has 77 terms instead of roughly 1300. The two models place the pole within 0.3 mas RMS and 1.2 mas at worst over 1990 to 2040, which moves a position by at most 25 cm at geostationary altitude and 4 cm in low Earth orbit. Use IAU 2000B where throughput matters more than that difference, and the default elsewhere.

## Per-Call Model Selection

`bias_precession_nutation_model`, `bias_precession`, `nutation`, and `gast_rotation` take the model as an explicit argument instead of reading the global setting, so a caller can evaluate either model without changing it for the rest of the program.

## See Also

- [GCRF ↔ ITRF Transformations](gcrf_itrf.md)
- [GCRF ↔ MOD ↔ TOD Transformations](equinox_frames.md)
- [Reference Frame Router](frame_transformations.md)
- [Reference Frames Overview](index.md)
