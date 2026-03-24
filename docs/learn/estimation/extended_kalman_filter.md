# Extended Kalman Filter

The Extended Kalman Filter (EKF) is a sequential estimator that linearizes dynamics and
measurement models around the current state estimate. At each observation, it propagates
state and covariance forward in time (predict step), then incorporates the measurement
(update step). Brahe's EKF leverages the propagator's built-in State Transition Matrix
(STM) for covariance prediction, so no separate covariance propagation is needed.

## Setting Up

The EKF constructor takes the initial conditions, measurement models, and propagation
configuration. It internally builds a numerical orbit propagator with STM enabled.

```python
import brahe as bh
import numpy as np

bh.initialize_eop()

epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7612.0, 0.0])
p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

ekf = bh.ExtendedKalmanFilter(
    epoch, state, p0,
    measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
    propagation_config=bh.NumericalPropagationConfig.default(),
    force_config=bh.ForceModelConfig.two_body(),
)
```

The constructor ensures STM propagation is enabled regardless of the propagation config
passed in. The initial covariance matrix dimensions must match the state vector length.

## Processing Observations

Each observation pairs a measurement vector with a timestamp and a model index indicating
which measurement model to use.

**One at a time** with `process_observation()`, which enforces strict chronological ordering
and returns a `FilterRecord` with full diagnostics:

```python
obs = bh.Observation(epoch + 60.0, np.array([6878e3, 100.0, 50.0]), model_index=0)
record = ekf.process_observation(obs)
```

**In batch** with `process_observations()`, which auto-sorts by epoch before processing:

```python
observations = [
    bh.Observation(epoch + 60.0 * i, truth_pos, model_index=0)
    for i in range(1, 31)
]
ekf.process_observations(observations)
```

Different measurement types can be interleaved by setting different `model_index` values.
For example, position observations at index 0 and range observations at index 1 can arrive
in any order when using `process_observations()`.

## Accessing Results

After processing, the filter state is available through pass-through methods:

```python
state = ekf.current_state()         # numpy 1D array
cov = ekf.current_covariance()      # numpy 2D array or None
epoch = ekf.current_epoch()         # Epoch
records = ekf.records()             # list of FilterRecord
```

Each `FilterRecord` captures the complete diagnostic state of a single update:

| Field | Description |
|-------|-------------|
| `state_predicted` | State after propagation, before measurement update |
| `covariance_predicted` | Covariance after propagation, before update |
| `state_updated` | State after measurement incorporation |
| `covariance_updated` | Covariance after measurement incorporation |
| `prefit_residual` | $\mathbf{z} - h(\mathbf{x}_{pred})$ -- innovation |
| `postfit_residual` | $\mathbf{z} - h(\mathbf{x}_{upd})$ -- should be small |
| `kalman_gain` | $K$ matrix used for the update |
| `measurement_name` | Name of the measurement model used |

Pre-fit residuals indicate how well the predicted state matches the observation. Post-fit
residuals should be smaller, confirming the update improved the estimate. Monitoring these
over time is the primary tool for assessing filter health.

## Process Noise

Process noise accounts for unmodeled dynamics (drag variations, gravity model truncation,
etc.) by inflating the predicted covariance at each step. Without process noise, the
covariance monotonically decreases and the filter eventually ignores new measurements.

```python
q = np.diag([1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8])
pn = bh.ProcessNoiseConfig(q, scale_with_dt=True)
config = bh.EKFConfig(process_noise=pn)

ekf = bh.ExtendedKalmanFilter(
    epoch, state, p0,
    measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
    propagation_config=bh.NumericalPropagationConfig.default(),
    force_config=bh.ForceModelConfig.two_body(),
    config=config,
)
```

When `scale_with_dt=True`, the effective process noise is $Q \cdot \Delta t$ (continuous-time
model). When `False`, $Q$ is applied as-is at each step (discrete-time model).

## Using Custom Dynamics

For systems beyond standard orbital mechanics, use `ExtendedKalmanFilter.from_dynamics()`
with a user-defined dynamics function:

```python
def my_dynamics(t, state, params):
    r = state[:3]
    v = state[3:6]
    r_mag = np.linalg.norm(r)
    a = -bh.GM_EARTH / r_mag**3 * r
    return np.concatenate([v, a])

ekf = bh.ExtendedKalmanFilter.from_dynamics(
    epoch, state, p0,
    measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
    dynamics=my_dynamics,
    propagation_config=bh.NumericalPropagationConfig.default(),
)
```

This constructs a generic `NumericalPropagator` internally. See
[General Dynamics Propagation](../orbit_propagation/numerical_propagation/generic_dynamics.md)
for details on the dynamics function signature.

---

## See Also

- [Measurement Models](measurement_models.md) -- Built-in and custom measurement types
- [Custom Models](custom_models.md) -- Defining measurement models in Python
- [EKF API Reference](../../library_api/estimation/extended_kalman_filter.md) -- Complete method documentation
- [Covariance and Sensitivity](../orbit_propagation/numerical_propagation/covariance_sensitivity.md) -- STM propagation details
