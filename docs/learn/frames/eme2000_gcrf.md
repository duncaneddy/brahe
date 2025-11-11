# EME2000 ↔ GCRF Transformations

The EME2000 (Earth Mean Equator and Equinox of J2000.0) to GCRF (Geocentric Celestial Reference Frame) transformation accounts for the frame bias between the classical J2000.0 reference frame and the modern ICRS-aligned GCRF.

!!! tip "When to Use EME2000"
    EME2000 should primarily be used when:

    - Working with older systems or datasets that use EME2000 coordinates
    - Interfacing with software that requires EME2000 input/output
    - Comparing results with historical analyses performed in EME2000

    For new applications, **use GCRF as your standard inertial frame**. GCRF is the current IAU/IERS standard and provides the most accurate representation of an inertial reference frame.


## Reference Frames

### EME2000 (Earth Mean Equator and Equinox of J2000.0)

EME2000, also known as J2000.0, is the classical inertial reference frame defined by the mean equator and mean equinox of the Earth at the J2000.0 epoch (January 1, 2000, 12:00 TT). This frame was widely used in older astrodynamics systems and is still found in many datasets and applications.

Key characteristics:

- Inertial frame (non-rotating)
- Defined using the mean equator and equinox at J2000.0
- Origin at Earth's center of mass

### Geocentric Celestial Reference Frame (GCRF)

The GCRF is the modern standard inertial reference frame, aligned with the International Celestial Reference System (ICRS). It is realized using observations of distant quasars and represents the current best realization of an inertial frame.

Key characteristics:

- Inertial frame (non-rotating)
- ICRS-aligned (quasi-inertial with respect to distant objects)
- Origin at Earth's center of mass
- Standard frame for modern astrodynamics applications

## Frame Bias

The transformation between EME2000 and GCRF is a **constant frame bias** that does not vary with time. This bias accounts for the small offset between the J2000.0 mean equator/equinox and the ICRS alignment arising from the improved observational data used to define the ICRS.

The bias is very small (on the order of milliarcseconds) but can matter for high-precision applications.

!!! tip "Time Independence"

    Unlike GCRF ↔ ITRF transformations, which are time-dependent and require Earth Orientation Parameters, the EME2000 ↔ GCRF transformation is **constant** and does not require an epoch parameter. The transformation is the same at all times.

## EME2000 to GCRF

Transform coordinates from the EME2000 frame to the modern GCRF.

### State Vector

Transform a complete state vector (position and velocity) from EME2000 to GCRF:

=== "Python"

    ``` python
    --8<-- "./examples/frames/eme2000_to_gcrf_state.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/eme2000_to_gcrf_state.rs:4"
    ```

!!! tip "Velocity Transformation"
    Because the transformation does not vary with time, velocity vectors are directly rotated without additional correction terms. There is no time-varying rotation rate to account for.

### Position Vector

Transform a position vector from EME2000 to GCRF:

=== "Python"

    ``` python
    --8<-- "./examples/frames/eme2000_to_gcrf_position.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/eme2000_to_gcrf_position.rs:4"
    ```

### Rotation Matrix

Get the constant rotation matrix from EME2000 to GCRF:

=== "Python"

    ``` python
    --8<-- "./examples/frames/eme2000_to_gcrf_rotation.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/eme2000_to_gcrf_rotation.rs:4"
    ```

## GCRF to EME2000

Transform coordinates from the modern GCRF to the older EME2000 frame.

### State Vector

Transform a complete state vector (position and velocity) from GCRF to EME2000:

=== "Python"

    ``` python
    --8<-- "./examples/frames/gcrf_to_eme2000_state.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/gcrf_to_eme2000_state.rs:4"
    ```

### Position Vector

Transform a position vector from GCRF to EME2000:

=== "Python"

    ``` python
    --8<-- "./examples/frames/gcrf_to_eme2000_position.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/gcrf_to_eme2000_position.rs:4"
    ```

### Rotation Matrix

Get the constant rotation matrix from GCRF to EME2000:

=== "Python"

    ``` python
    --8<-- "./examples/frames/gcrf_to_eme2000_rotation.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/gcrf_to_eme2000_rotation.rs:4"
    ```

## Frame Bias Matrix

The underlying frame bias transformation can also be accessed directly:

=== "Python"

    ``` python
    --8<-- "./examples/frames/bias_eme2000.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/bias_eme2000.rs:4"
    ```

The bias matrix is identical to `rotation_gcrf_to_eme2000()` and represents the constant transformation from GCRF to EME2000.

## See Also

- [GCRF ↔ ITRF Transformations](gcrf_itrf.md) - Time-dependent transformations between inertial and Earth-fixed frames
- [Reference Frames Overview](index.md) - Complete overview of all reference frames in Brahe
