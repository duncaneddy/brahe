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

## Azimuth/Elevation/Range

`AzElRangeMeasurementModel` handles **ground-based radar/tracking sensor observations** in
the station's local topocentric (SEZ) frame. Unlike the ECEF and inertial models above, the
measurement is angular plus range, not a Cartesian sub-vector of the state:

- **State**: $\mathbf{x} = [x, y, z, v_x, v_y, v_z, \ldots]_{\text{ECI}}$ (meters, m/s)
- **Measurement**: $\mathbf{z} = [\text{azimuth}, \text{elevation}, \text{range}] +
  \mathbf{b}$ -- azimuth clockwise from north, elevation from the local horizon, in the
  units given by `AngleFormat` at construction (degrees or radians); range in meters
- **Jacobian**: Numerical (finite difference) -- the ECI-to-ECEF rotation is epoch-dependent

The model accepts a constant bias `[bias_az, bias_el, bias_range]`, applied inside
`predict()`. This models a calibrated sensor (e.g. Vallado Table 4-4 az/el/range bias
values): a filter built with the same bias as the measurement source stays consistent,
rather than needing to estimate the bias away as an unmodeled error.

`residual()` wraps the azimuth component into $(-180°, 180°]$ (or $(-\pi, \pi]$ in radians)
so a pass crossing the 0/360° boundary does not produce a spurious ~360° residual:

=== "Python"
    ``` python
    --8<-- "./examples/estimation/ssn_tracking.py:11"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/estimation/ssn_tracking.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/estimation/ssn_tracking.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/estimation/ssn_tracking.rs.txt"
        ```

The azimuth wrap in `residual()` only fixes the final residual computation. When this
model is used with the Unscented Kalman Filter, the sigma points propagated through
`predict()` are averaged directly to form the predicted measurement mean; if a pass carries
sigma-point azimuths that straddle the wrap (e.g. some near 359°, others near 1°), that
mean is computed without wrap-awareness and can be biased toward the middle of the circle
rather than the correct side. This is a documented limitation for UKF use with wrap-prone
angular measurements -- the EKF, which linearizes rather than averaging samples, is not
affected.

`SimpleSSNSensor` pairs a sensor site (location, field-of-view limits, bias/noise
calibration -- see the [SSN Sensor Datasets](../datasets/ssn_sensors.md) guide) with
measurement generation, and its `measurement_model()` method returns an
`AzElRangeMeasurementModel` built from the same bias and noise, so simulated measurements
and the filter's model stay consistent by construction. See the
[SSN Radar Tracking example](../../examples/ssn_tracking.md) for a full EKF/UKF/BLS
walkthrough built on this dataset.

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
- [SSN Sensor Datasets](../datasets/ssn_sensors.md) -- Sensor sites backing `SimpleSSNSensor`
- [Measurement Models API Reference](../../library_api/estimation/measurement_models.md) -- Complete class documentation
- [AzElRangeMeasurementModel API Reference](../../library_api/estimation/azelrange_measurement_model.md) -- Complete class documentation
- [SimpleSSNSensor API Reference](../../library_api/estimation/simple_ssn_sensor.md) -- Complete class documentation
