# GCRF ↔ ITRF Transformations

The Geocentric Celestial Reference Frame (GCRF) and International Terrestrial Reference Frame (ITRF) are the modern IAU/IERS standard reference frames for Earth-orbiting satellite applications. 

## Reference Frames

### Geocentric Celestial Reference Frame (GCRF)

The Geocentric Celestial Reference Frame is the standard modern inertial reference frame for Earth-orbiting satellites. It is aligned with the International Celestial Reference Frame (ICRF) and realized using the positions of distant quasars. The GCRF has its origin at the Earth's center of mass and its axes are fixed with respect to distant stars.

The GCRF is an Earth-centered inertial (ECI) frame, meaning it does not rotate with the Earth.

### International Terrestrial Reference Frame (ITRF)

The International Terrestrial Reference Frame is the standard Earth-fixed reference frame maintained by the International Earth Rotation and Reference Systems Service (IERS). The ITRF rotates with the Earth and its axes are aligned with the Earth's geographic coordinate system (polar axis and Greenwich meridian).

The ITRF is an Earth-centered Earth-fixed (ECEF) frame, meaning it rotates with the Earth.

## Transformation Model

Brahe implements the IAU 2006/2000A precession-nutation model with Celestial Intermediate Origin (CIO) based transformation, following IERS conventions. The transformation is accomplished using the IAU 2006/2000A, CIO-based theory using classical angles. The method as described in section 5.5 of the [SOFA C transformation cookbook](https://www.iausofa.org/s/sofa_pn_c.pdf). The transformation accounts for:

- **Precession and nutation** of Earth's rotation axis
- **Earth's rotation** about its instantaneous spin axis
- **Polar motion** and UT1-UTC corrections

These transformations are **time-dependent** and require Earth Orientation Parameters (EOP) for high accuracy. The transformations will use the currently loaded Earth orientation data provider to obtain the necessary parameters automatically. See [Earth Orientation Data](../eop/index.md) for more details.

## GCRF to ITRF

Transform coordinates from the inertial GCRF to the Earth-fixed ITRF.

### State Vector

Transform a complete state vector (position and velocity) from GCRF to ITRF:

=== "Python"

    ``` python
    --8<-- "./examples/frames/gcrf_to_itrf_state.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/gcrf_to_itrf_state.rs:4"
    ```

!!! warning "Velocity Transformation"
    Simply rotating velocity vectors will not yield correct velocity components in the ITRF frame due to the Earth's rotation. State vector transformation functions properly account for observed velocity changes in the ITRF frame due to Earth's rotation.

### Rotation Matrix

Get the rotation matrix from GCRF to ITRF:

=== "Python"

    ``` python
    --8<-- "./examples/frames/gcrf_to_itrf_rotation.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/gcrf_to_itrf_rotation.rs:4"
    ```

## ITRF to GCRF

Transform coordinates from the Earth-fixed ITRF to the inertial GCRF.

### State Vector

Transform a complete state vector (position and velocity) from ITRF to GCRF:

=== "Python"

    ``` python
    --8<-- "./examples/frames/itrf_to_gcrf_state.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/itrf_to_gcrf_state.rs:4"
    ```

!!! warning "Velocity Transformation"
    Simply rotating velocity vectors will not yield correct velocity components in the GCRF frame due to the Earth's rotation. State vector transformation functions properly account for observed velocity changes when transforming from the rotating ITRF frame.

### Rotation Matrix

Get the rotation matrix from ITRF to GCRF:

=== "Python"

    ``` python
    --8<-- "./examples/frames/itrf_to_gcrf_rotation.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/itrf_to_gcrf_rotation.rs:4"
    ```

## Intermediate Rotation Matrices

The full GCRF to ITRF transformation is composed of three sequential rotations. Brahe provides access to these intermediate rotation matrices for advanced applications or for understanding the transformation components.

The complete transformation chain is:

```
GCRF ↔ CIRS ↔ TIRS ↔ ITRF
      (BPN)   (ER)   (PM)
```

where:

- **BPN** = Bias-Precession-Nutation: Accounts for Earth's precession and nutation
- **ER** = Earth Rotation: Accounts for Earth's daily rotation
- **PM** = Polar Motion: Accounts for polar motion and UT1-UTC corrections

### Bias-Precession-Nutation Matrix

Get the bias-precession-nutation matrix (GCRF to CIRS):

=== "Python"

    ``` python
    --8<-- "./examples/frames/bias_precession_nutation.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/bias_precession_nutation.rs:4"
    ```

### Earth Rotation Matrix

Get the Earth rotation matrix (CIRS to TIRS):

=== "Python"

    ``` python
    --8<-- "./examples/frames/earth_rotation.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/earth_rotation.rs:4"
    ```

### Polar Motion Matrix

Get the polar motion matrix (TIRS to ITRF):

=== "Python"

    ``` python
    --8<-- "./examples/frames/polar_motion.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/polar_motion.rs:4"
    ```

!!! note
    For most applications, use the combined `rotation_gcrf_to_itrf` or `state_gcrf_to_itrf` functions rather than computing intermediate matrices separately. The intermediate matrices are provided for educational purposes and specialized applications.

## See Also

- [ECI ↔ ECEF Naming Convention](eci_ecef.md) - Legacy naming convention that maps to GCRF/ITRF
- [Reference Frames Overview](index.md) - Complete overview of all reference frames in Brahe
