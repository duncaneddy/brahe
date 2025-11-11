# ECI ↔ ECEF Transformations

The ECI (Earth-Centered Inertial) and ECEF (Earth-Centered Earth-Fixed) naming convention is a traditional and widely-used terminology in the astrodynamics community.

!!! info "Naming Convention"

    Brahe provides two sets of function names for frame transformations, both currently mapping to the same underlying implementations:

    - **ECI/ECEF naming**: Common coordinate system names (e.g., `rotation_eci_to_ecef`, `state_eci_to_ecef`)
    - **GCRF/ITRF naming**: Explicit reference frame names (e.g., `rotation_gcrf_to_itrf`, `state_gcrf_to_itrf`)

    The ECI/ECEF naming will always use the "best" available transformations in Brahe, while the GCRF/ITRF naming ensures consistent use of specific reference frame implementations.

    
## Reference Frames

### ECI (Earth-Centered Inertial)

- A non-rotating frame fixed with respect to distant stars
- Inertial frame suitable for integration of equations of motion
- **Current Realization**: GCRF (Geocentric Celestial Reference Frame)

### ECEF (Earth-Centered Earth-Fixed)

- A rotating frame fixed to the Earth's surface
- Ideal for computing positions and motions relative to terrestrial locations and observers
- **Current Realization**: ITRF (International Terrestrial Reference Frame)

## ECI to ECEF

Converting from ECI to ECEF accounts for the Earth's rotation, polar motion, and precession-nutation effects. These transformations are time-dependent and require Earth Orientation Parameters (EOP) for high accuracy. The transformations will use the currently loaded Earth orientation data provider to obtain the necessary parameters automatically. See [Earth Orientation Data](../eop/index.md) for more details.

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

!!! warning "Velocity Transformation"
    Simply rotating velocity vectors will not yield correct velocity components in the ECEF frame due to the Earth's rotation. State vector transformation functions properly account for observed velocity changes in the ECEF frame due to Earth's rotation.

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

!!! warning "Velocity Transformation"
    Simply rotating velocity vectors will not yield correct velocity components in the ECI frame due to the Earth's rotation. State vector transformation functions properly account for observed velocity changes when transforming from the rotating ECEF frame.

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

## See Also

- [GCRF ↔ ITRF Transformations](gcrf_itrf.md) - Detailed documentation of the underlying reference frame implementations
- [Reference Frames Overview](index.md) - Complete overview of all reference frames in Brahe
