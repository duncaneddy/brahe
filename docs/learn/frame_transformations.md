# Frame Transformations

Reference frame transformations are a fundamental aspect of astrodynamics. Different tasks require working in different reference frames, and accurate transformations between these frames are essential for precise calculations. The two primary reference frames used in orbital mechanics are:

- **ECI (Earth-Centered Inertial)**: A non-rotating frame fixed with respect to distant stars. It is an inertial frame and can be used for integration of equations of motion.
- **ECEF (Earth-Centered Earth-Fixed)**: A rotating frame fixed to the Earth's surface, ideal for computing positions and motions relative to terrestrial locations and observers.

Brahe uses the [IAU SOFA](https://www.iausofa.org/) (Standards of Fundamental Astronomy) library reference frame transformations under the hood to provide speed, accuracy, and reliability. The implementation follows the IAU 2006/2000A precession-nutation model with Celestial Intermediate Origin (CIO) based transformation. Brahe follows the IERS conventions for reference system definition. To learn more about these models, refer to the [IERS Conventions (2010)](https://www.iers.org/IERS/EN/Publications/TechnicalNotes/tn36.html).

## Reference Frames

### Geocentric Celestial Reference Frame (GCRF)

The Geocentric Celestial Reference Frame (GCRF) is the standard modern inertial reference frame for Earth-orbiting satellites. It is aligned with the International Celestial Reference Frame (ICRF) and realized using the positions of distant quasars. The GCRF has its origin at the Earth's center of mass and its axes are fixed with respect to distant stars.

The GCRF is an Earth-centered inertial (ECI) frame, meaning it does not rotate with the Earth.

### International Terrestrial Reference Frame (ITRF)

The International Terrestrial Reference Frame (ITRF) is the standard Earth-fixed reference frame maintained by the International Earth Rotation and Reference Systems Service (IERS). The ITRF rotates with the Earth and its axes are aligned with the Earth's geographic coordinate system (polar axis and Greenwich meridian).

The ITRF is an Earth-centered Earth-fixed (ECEF) frame, meaning it rotates with the Earth.

!!! tip "Naming Conventions"

    Brahe provides two sets of function names for frame transformations, both referring to the same underlying implementations:

    - **ECI/ECEF naming**: Common coordinate system names (e.g., `rotation_eci_to_ecef`, `state_eci_to_ecef`)
    - **GCRF/ITRF naming**: Explicit reference frame names (e.g., `rotation_gcrf_to_itrf`, `state_gcrf_to_itrf`)

    Both naming conventions are fully supported and provide identical results. The ECI/ECEF names are intuitive and widely used in the astrodynamics community, while the GCRF/ITRF names explicitly identify the specific reference frame implementations. Users can choose whichever naming convention they prefer - all examples in this documentation use the ECI/ECEF convention for simplicity.

    The ECI/ECEF implementation will be updated to use whatever the current "best" reference frame implementations are, while the GCRF/ITRF implementation will always refer to the specific IAU SOFA implementations described above.

## ECI to ECEF

Converting from ECI to ECEF accounts for the Earth's rotation, polar motion, and precession-nutation effects. These transformations are time-dependent and require Earth Orientation Parameters (EOP) for high accuracy.

### State Vector

Transform a complete state vector (position and velocity) from ECI to ECEF:

=== "Python"

    ``` python
    --8<-- "./examples/frames/eci_to_ecef_state.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/eci_to_ecef_state.rs:4"
    ```

!!! warning "Caution"
    Simply rotating veloicty vectors will not yield correct velocity components in the ECEF frame due to the Earth's rotation. State vector transformation functions properly account for observed velocity changes in the ECEF frame due to Earth's rotation.

### Rotation Matrix

Get the rotation matrix from ECI to ECEF:

=== "Python"

    ``` python
    --8<-- "./examples/frames/eci_to_ecef_rotation.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/eci_to_ecef_rotation.rs:4"
    ```

## ECEF to ECI

Converting from ECEF to ECI reverses the transformation, converting Earth-fixed coordinates back to the inertial frame.

### State Vector

Transform a complete state vector (position and velocity) from ECEF to ECI:

=== "Python"

    ``` python
    --8<-- "./examples/frames/ecef_to_eci_state.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/ecef_to_eci_state.rs:4"
    ```

!!! warning "Caution"
    Simply rotating velocity vectors will not yield correct velocity components in the ECEF frame due to the Earth's rotation. State vector transformation functions properly account for observed velocity changes in the ECEF frame due to Earth's rotation.

### Rotation Matrix

Get the rotation matrix from ECEF to ECI:

=== "Python"

    ``` python
    --8<-- "./examples/frames/ecef_to_eci_rotation.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/ecef_to_eci_rotation.rs:4"
    ```

## Intermediate Rotation Matrices

The full ECI to ECEF transformation is composed of three sequential rotations. Brahe provides access to these intermediate rotation matrices for advanced applications or for understanding the transformation components.

The complete transformation chain is

```
GCRF ↔ CIRS ↔ TIRS ↔ ITRF
      (BPN)   (ER)   (PM)
```

where

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
    For most applications, use the combined `rotation_eci_to_ecef` or `state_eci_to_ecef` functions rather than computing intermediate matrices separately. The intermediate matrices are provided for educational purposes and specialized applications.
