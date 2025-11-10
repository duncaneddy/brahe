# EME2000 ↔ GCRF Transformations

The EME2000 (Earth Mean Equator and Equinox of J2000.0) to GCRF (Geocentric Celestial Reference Frame) transformation accounts for the frame bias between the classical J2000.0 reference frame and the modern ICRS-aligned GCRF.

## Reference Frames

### EME2000 (Earth Mean Equator and Equinox of J2000.0)

EME2000, also known as J2000.0, is the classical inertial reference frame defined by the mean equator and mean equinox of the Earth at the J2000.0 epoch (January 1, 2000, 12:00 TT). This frame was widely used in legacy astrodynamics systems and is still found in many datasets and applications.

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

The transformation between EME2000 and GCRF is a **constant frame bias** that does not vary with time. This bias accounts for the small offset between the J2000.0 mean equator/equinox and the ICRS alignment, arising from:

- Difference in the definition of the celestial pole
- Difference in the origin of right ascension
- Corrections to the IAU 1976 precession constant

The bias is very small (on the order of milliarcseconds) but significant for high-precision applications.

!!! info "Time Independence"

    Unlike GCRF ↔ ITRF transformations, which are time-dependent and require Earth Orientation Parameters, the EME2000 ↔ GCRF transformation is **constant** and does not require an epoch parameter. The transformation is the same at all times.

## EME2000 to GCRF

Transform coordinates from the legacy EME2000 frame to the modern GCRF.

### State Vector

Transform a complete state vector (position and velocity) from EME2000 to GCRF:

=== "Python"

    ``` python
    import brahe as bh
    import numpy as np

    # State vector in EME2000 [x, y, z, vx, vy, vz] (meters, m/s)
    state_eme2000 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])

    # Transform to GCRF
    state_gcrf = bh.state_eme2000_to_gcrf(state_eme2000)
    print(f"GCRF state: {state_gcrf}")
    ```

=== "Rust"

    ``` rust
    use brahe as bh;
    use nalgebra as na;

    fn main() {
        // State vector in EME2000 [x, y, z, vx, vy, vz] (meters, m/s)
        let state_eme2000 = na::SVector::<f64, 6>::new(
            bh::R_EARTH + 500e3, 0.0, 0.0,
            0.0, 7600.0, 0.0
        );

        // Transform to GCRF
        let state_gcrf = bh::state_eme2000_to_gcrf(state_eme2000);
        println!("GCRF state: {:?}", state_gcrf);
    }
    ```

!!! note "Velocity Transformation"
    Because the transformation does not vary with time, velocity vectors are directly rotated without additional correction terms. There is no time-varying rotation rate to account for.

### Position Vector

Transform a position vector from EME2000 to GCRF:

=== "Python"

    ``` python
    import brahe as bh
    import numpy as np

    # Position vector in EME2000 (meters)
    r_eme2000 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])

    # Transform to GCRF
    r_gcrf = bh.position_eme2000_to_gcrf(r_eme2000)
    print(f"GCRF position: {r_gcrf}")
    ```

=== "Rust"

    ``` rust
    use brahe as bh;
    use nalgebra as na;

    fn main() {
        // Position vector in EME2000 (meters)
        let r_eme2000 = na::Vector3::new(bh::R_EARTH + 500e3, 0.0, 0.0);

        // Transform to GCRF
        let r_gcrf = bh::position_eme2000_to_gcrf(r_eme2000);
        println!("GCRF position: {:?}", r_gcrf);
    }
    ```

### Rotation Matrix

Get the constant rotation matrix from EME2000 to GCRF:

=== "Python"

    ``` python
    import brahe as bh

    # Get rotation matrix from EME2000 to GCRF
    R = bh.rotation_eme2000_to_gcrf()
    print(f"Rotation matrix shape: {R.shape}")
    # Output: Rotation matrix shape: (3, 3)
    ```

=== "Rust"

    ``` rust
    use brahe as bh;

    fn main() {
        // Get rotation matrix from EME2000 to GCRF
        let r = bh::rotation_eme2000_to_gcrf();
        println!("Rotation matrix: {:?}", r);
    }
    ```

## GCRF to EME2000

Transform coordinates from the modern GCRF to the legacy EME2000 frame.

### State Vector

Transform a complete state vector (position and velocity) from GCRF to EME2000:

=== "Python"

    ``` python
    import brahe as bh
    import numpy as np

    # State vector in GCRF [x, y, z, vx, vy, vz] (meters, m/s)
    state_gcrf = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])

    # Transform to EME2000
    state_eme2000 = bh.state_gcrf_to_eme2000(state_gcrf)
    print(f"EME2000 state: {state_eme2000}")
    ```

=== "Rust"

    ``` rust
    use brahe as bh;
    use nalgebra as na;

    fn main() {
        // State vector in GCRF [x, y, z, vx, vy, vz] (meters, m/s)
        let state_gcrf = na::SVector::<f64, 6>::new(
            bh::R_EARTH + 500e3, 0.0, 0.0,
            0.0, 7600.0, 0.0
        );

        // Transform to EME2000
        let state_eme2000 = bh::state_gcrf_to_eme2000(state_gcrf);
        println!("EME2000 state: {:?}", state_eme2000);
    }
    ```

### Position Vector

Transform a position vector from GCRF to EME2000:

=== "Python"

    ``` python
    import brahe as bh
    import numpy as np

    # Position vector in GCRF (meters)
    r_gcrf = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])

    # Transform to EME2000
    r_eme2000 = bh.position_gcrf_to_eme2000(r_gcrf)
    print(f"EME2000 position: {r_eme2000}")
    ```

=== "Rust"

    ``` rust
    use brahe as bh;
    use nalgebra as na;

    fn main() {
        // Position vector in GCRF (meters)
        let r_gcrf = na::Vector3::new(bh::R_EARTH + 500e3, 0.0, 0.0);

        // Transform to EME2000
        let r_eme2000 = bh::position_gcrf_to_eme2000(r_gcrf);
        println!("EME2000 position: {:?}", r_eme2000);
    }
    ```

### Rotation Matrix

Get the constant rotation matrix from GCRF to EME2000:

=== "Python"

    ``` python
    import brahe as bh

    # Get rotation matrix from GCRF to EME2000
    R = bh.rotation_gcrf_to_eme2000()
    print(f"Rotation matrix shape: {R.shape}")
    # Output: Rotation matrix shape: (3, 3)
    ```

=== "Rust"

    ``` rust
    use brahe as bh;

    fn main() {
        // Get rotation matrix from GCRF to EME2000
        let r = bh::rotation_gcrf_to_eme2000();
        println!("Rotation matrix: {:?}", r);
    }
    ```

## Frame Bias Matrix

The underlying frame bias transformation can also be accessed directly:

=== "Python"

    ``` python
    import brahe as bh

    # Get the bias matrix
    B = bh.bias_eme2000()
    print(f"Bias matrix shape: {B.shape}")
    # Output: Bias matrix shape: (3, 3)
    ```

=== "Rust"

    ``` rust
    use brahe as bh;

    fn main() {
        // Get the bias matrix
        let b = bh::bias_eme2000();
        println!("Bias matrix: {:?}", b);
    }
    ```

The bias matrix is identical to `rotation_gcrf_to_eme2000()` and represents the constant transformation from GCRF to EME2000.

## When to Use EME2000

EME2000 should primarily be used when:

- Working with legacy systems or datasets that use J2000.0 coordinates
- Interfacing with software that requires J2000.0 input/output
- Comparing results with historical analyses performed in J2000.0

For new applications, **use GCRF as your standard inertial frame**. GCRF is the current IAU/IERS standard and provides the most accurate representation of an inertial reference frame.

## See Also

- [GCRF ↔ ITRF Transformations](gcrf_itrf.md) - Time-dependent transformations between inertial and Earth-fixed frames
- [Reference Frames Overview](index.md) - Complete overview of all reference frames in Brahe
