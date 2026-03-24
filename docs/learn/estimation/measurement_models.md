# Measurement Models

A measurement model defines $h(\mathbf{x}, t)$ -- the function that maps a state vector to
a predicted observation. Each model also specifies its noise covariance $R$ and measurement
dimension. The EKF uses these to compute innovations, Kalman gain, and state updates.

Brahe provides six built-in models covering the most common GPS-like observation types.
They are organized by the reference frame of the measurement: **inertial** (ECI) models
observe state components directly, while **ECEF** models account for Earth rotation.

## Inertial Models

Inertial models directly extract components from the ECI state vector. Because the mapping
is a simple selection (identity sub-matrix), their Jacobians are analytical -- no numerical
differentiation is needed, making them fast and exact.

**InertialPositionMeasurementModel** observes $\mathbf{z} = [x, y, z]$ from the state.
The measurement Jacobian is $H = [I_{3\times3} \mid 0_{3\times(n-3)}]$.

```python
import brahe as bh

# Isotropic 10 m noise on all position axes
pos_model = bh.InertialPositionMeasurementModel(10.0)

# Per-axis noise (different accuracy in each direction)
pos_model = bh.InertialPositionMeasurementModel.per_axis(5.0, 10.0, 15.0)
```

**InertialVelocityMeasurementModel** observes $\mathbf{z} = [v_x, v_y, v_z]$ from the
state. The Jacobian picks out velocity elements: $H = [0_{3\times3} \mid I_{3\times3} \mid 0_{3\times(n-6)}]$.

**InertialStateMeasurementModel** observes the full 6D state
$\mathbf{z} = [x, y, z, v_x, v_y, v_z]$. It takes separate position and velocity noise
standard deviations.

```python
# 10 m position noise, 0.1 m/s velocity noise
state_model = bh.InertialStateMeasurementModel(10.0, 0.1)
```

## ECEF Models

ECEF models are designed for GNSS receiver outputs, where measurements are reported in the
Earth-fixed frame. The estimator state is still in ECI, so these models internally rotate
the predicted state from ECI to ECEF at the observation epoch before computing the
measurement.

Because the ECI-to-ECEF rotation is epoch-dependent and involves Earth orientation
parameters, the Jacobians are computed via finite differences using the same perturbation
infrastructure as the propagator Jacobians (`DifferenceMethod::Central` with adaptive
perturbation sizing).

**ECEFPositionMeasurementModel** converts ECI position to ECEF:
$\mathbf{z} = R_{ECI \to ECEF}(t) \cdot [x, y, z]_{ECI}$.

**ECEFVelocityMeasurementModel** converts the full ECI state to ECEF and extracts
velocity, properly accounting for Earth rotation effects on the velocity transformation.

**ECEFStateMeasurementModel** converts the full 6D state from ECI to ECEF.

```python
# Typical GNSS accuracy: 5 m position, 0.05 m/s velocity
ecef_model = bh.ECEFStateMeasurementModel(5.0, 0.05)
```

## Choosing a Model

**Inertial models** are appropriate when measurements are already in the ECI frame or when
the ECI-to-ECEF rotation is handled externally. They are computationally cheaper (analytical
Jacobians) and simpler to reason about.

**ECEF models** are appropriate when processing raw GNSS receiver outputs that report
position and velocity in ECEF. They handle the frame conversion internally so the user does
not need to pre-rotate observations.

For measurements that do not fit either category -- range, range-rate, angles, Doppler, or
any other nonlinear observation -- define a [custom measurement model](custom_models.md).

---

## See Also

- [Custom Models](custom_models.md) -- Defining measurement models in Python
- [Extended Kalman Filter](extended_kalman_filter.md) -- Using models with the EKF
- [Measurement Models API Reference](../../library_api/estimation/measurement_models.md) -- Complete class documentation
