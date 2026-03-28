# Measurement Models

Measurement models define $h(\mathbf{x}, t)$ — the function mapping a filter state to a
predicted observation — along with a noise covariance $R$. Brahe provides six built-in
models for GNSS-like observations in ECEF and inertial frames. All assume the filter state
is Cartesian ECI: $\mathbf{x} = [x, y, z, v_x, v_y, v_z, \ldots]$ in meters and m/s.

The most common starting point is an ECEF position model consuming raw GNSS receiver
outputs:

=== "Python"
    ``` python
    --8<-- "./examples/estimation/ecef_gnss_tracking.py:11"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/estimation/ecef_gnss_tracking.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/estimation/ecef_gnss_tracking.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/estimation/ecef_gnss_tracking.rs.txt"
        ```

## ECEF Models

ECEF models process **GNSS receiver outputs** reported in the Earth-fixed frame. The
filter state remains in ECI — these models internally rotate the predicted state from ECI
to ECEF at each observation epoch. Jacobians are computed via central finite differences
because the rotation is epoch-dependent.

### ECEFPositionMeasurementModel

- **State**: $\mathbf{x} = [x, y, z, v_x, v_y, v_z, \ldots]_{\text{ECI}}$ (meters, m/s)
- **Measurement**: $\mathbf{z} = R_{\text{ECI} \to \text{ECEF}}(t) \cdot [x, y, z]_{\text{ECI}}$ — 3 values in meters
- **Jacobian**: Numerical (finite difference)

### ECEFVelocityMeasurementModel

Converts the full ECI state to ECEF and extracts velocity, properly accounting for Earth
rotation effects.

- **State**: $\mathbf{x} = [x, y, z, v_x, v_y, v_z, \ldots]_{\text{ECI}}$ (meters, m/s)
- **Measurement**: $\mathbf{z} = [v_x, v_y, v_z]_{\text{ECEF}}$ — 3 values in m/s
- **Jacobian**: Numerical (finite difference)

### ECEFStateMeasurementModel

Full 6D position + velocity in ECEF. Useful when a GNSS receiver provides both solutions
simultaneously.

- **State**: $\mathbf{x} = [x, y, z, v_x, v_y, v_z, \ldots]_{\text{ECI}}$ (meters, m/s)
- **Measurement**: $\mathbf{z} = [x, y, z, v_x, v_y, v_z]_{\text{ECEF}}$ — 6 values in meters + m/s
- **Jacobian**: Numerical (finite difference)

## Inertial Models

Inertial models directly extract components from the ECI state vector. The mapping is a
simple selection (identity sub-matrix), so Jacobians are **analytical** — fast and exact.
Use these when measurements are already in ECI, for simulation, or when the frame
conversion is handled externally.

### InertialPositionMeasurementModel

- **State**: $\mathbf{x} = [x, y, z, v_x, v_y, v_z, \ldots]_{\text{ECI}}$ (meters, m/s)
- **Measurement**: $\mathbf{z} = [x, y, z]_{\text{ECI}}$ — 3 values in meters
- **Jacobian**: $H = [I_{3 \times 3} \mid 0_{3 \times (n-3)}]$ (analytical)

### InertialVelocityMeasurementModel

- **State**: $\mathbf{x} = [x, y, z, v_x, v_y, v_z, \ldots]_{\text{ECI}}$ (meters, m/s)
- **Measurement**: $\mathbf{z} = [v_x, v_y, v_z]_{\text{ECI}}$ — 3 values in m/s
- **Jacobian**: $H = [0_{3 \times 3} \mid I_{3 \times 3} \mid 0_{3 \times (n-6)}]$ (analytical)

### InertialStateMeasurementModel

- **State**: $\mathbf{x} = [x, y, z, v_x, v_y, v_z, \ldots]_{\text{ECI}}$ (meters, m/s)
- **Measurement**: $\mathbf{z} = [x, y, z, v_x, v_y, v_z]_{\text{ECI}}$ — 6 values in meters + m/s
- **Jacobian**: $H = [I_{6 \times 6} \mid 0_{6 \times (n-6)}]$ (analytical)

## Noise Specification

All models accept noise as a scalar sigma (isotropic), per-axis sigmas, a full covariance
matrix, or upper-triangular packed elements:

=== "Python"
    ``` python
    --8<-- "./examples/estimation/measurement_noise.py:11"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/estimation/measurement_noise.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/estimation/measurement_noise.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/estimation/measurement_noise.rs.txt"
        ```

## Custom Measurement Models

For observations beyond the built-in models — range, range-rate, angles, Doppler, or any
nonlinear function — define a custom measurement model. Subclass `MeasurementModel` in
Python or implement the `MeasurementModel` trait in Rust.

The full pattern, including analytical Jacobians and mixing custom models with built-in
models in a single filter, is covered in the [Custom Models](custom_models.md) guide.

---

## See Also

- [Custom Models](custom_models.md) -- Writing custom measurement models with examples
- [Extended Kalman Filter](extended_kalman_filter.md) -- Using models with the EKF
- [Unscented Kalman Filter](unscented_kalman_filter.md) -- Using models with the UKF
- [Measurement Models API Reference](../../library_api/estimation/measurement_models.md) -- Complete class documentation
