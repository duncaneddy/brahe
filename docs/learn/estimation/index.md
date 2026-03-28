# Estimation

Estimation processes measurements to refine a spacecraft's state -- position, velocity, and
optionally additional parameters -- beyond what the dynamics model alone can predict. Brahe
provides an Extended Kalman Filter (EKF) with built-in and custom measurement models. The
primary workflow is: create a filter with an initial state estimate, feed it observations,
and read the refined state.

## The Core Workflow

Set up an EKF with a propagator, measurement model, and initial covariance, then process
observations sequentially. The filter propagates state and covariance to each observation
epoch and incorporates the measurement to produce an updated estimate.

=== "Python"
    ``` python
    --8<-- "./examples/estimation/ekf_position_tracking.py:11"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/estimation/ekf_position_tracking.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/estimation/ekf_position_tracking.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/estimation/ekf_position_tracking.rs.txt"
        ```

The EKF constructor accepts an initial epoch, state, covariance, one or more measurement
models, and propagation configuration. Internally it builds a numerical orbit propagator
with STM (State Transition Matrix) propagation enabled -- the STM drives covariance
prediction between observations. Each call to `process_observation()` performs a predict
step (propagate to observation time) followed by an update step (incorporate the
measurement via Kalman gain).

## How the Pieces Connect

The estimation module has three main components:

**Measurement models** define the observation function $h(\mathbf{x}, t)$ that maps a
state vector to a predicted measurement. Built-in models handle GPS-like position and
velocity observations in both inertial and ECEF frames. Custom models can be defined in
Python by subclassing `MeasurementModel`.

**The Extended Kalman Filter** orchestrates the estimation loop. It owns a numerical
propagator for state and covariance prediction and a list of measurement models for
incorporating observations. Different measurement types can arrive at different times --
each `Observation` carries a `model_index` indicating which model to use.

**Filter records** capture the full diagnostic state at each update: pre-fit state,
post-fit state, pre-fit and post-fit residuals, covariance, and Kalman gain. These enable
analysis of filter performance, consistency checks, and residual monitoring.

## What's Available

The current release includes two sequential filters:

- **Extended Kalman Filter (EKF)** -- linearizes dynamics and measurements using STM and
  Jacobians. Efficient (one propagation per step) and well-suited for mildly nonlinear
  problems.

- **Unscented Kalman Filter (UKF)** -- propagates sigma points through true nonlinear
  functions without linearization. More robust for strongly nonlinear problems, at the cost
  of 2n+1 propagations per step.

- **Batch Least Squares (BLS)** -- processes all observations simultaneously, iterating to
  minimize the weighted sum of squared residuals. Best for offline orbit determination with
  complete observation arcs. Supports two solver formulations and consider parameters.

All estimators share the same measurement models, observation types, and Python API.

---

## See Also

- [Measurement Models](measurement_models.md) -- Built-in GPS-like measurement types
- [Extended Kalman Filter](extended_kalman_filter.md) -- EKF setup, processing, and diagnostics
- [Unscented Kalman Filter](unscented_kalman_filter.md) -- UKF sigma points and EKF comparison
- [Custom Models](custom_models.md) -- Defining measurement models in Python
- [Batch Least Squares](batch_least_squares.md) -- BLS offline orbit determination
- [Estimation API Reference](../../library_api/estimation/index.md) -- Complete type and method documentation
- [Covariance and Sensitivity](../orbit_propagation/numerical_propagation/covariance_sensitivity.md) -- STM propagation used by the EKF
