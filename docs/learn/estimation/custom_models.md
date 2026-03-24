# Custom Measurement Models

Subclass `MeasurementModel` in Python to define any nonlinear measurement function. The
Rust EKF calls your Python `predict()` method during filtering, and computes the
measurement Jacobian via finite differences automatically unless you provide an analytical
override.

## The Pattern

=== "Python"
    ``` python
    --8<-- "./examples/estimation/custom_range_model.py:11"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/estimation/custom_range_model.py.txt"
        ```

The `RangeModel` above measures the Euclidean distance from a ground station to the
satellite. It implements four required methods and relies on the default finite-difference
Jacobian. The EKF processes range observations alongside built-in position observations by
assigning different `model_index` values.

## Required Methods

Every custom model must implement these four methods:

**`predict(epoch, state) -> numpy.ndarray`** -- compute the predicted measurement
$h(\mathbf{x}, t)$. The `epoch` is a `brahe.Epoch` and `state` is a 1D numpy array.
Return a 1D numpy array of length `measurement_dim()`.

**`noise_covariance() -> numpy.ndarray`** -- return the measurement noise covariance
matrix $R$ as a 2D numpy array of shape `(m, m)`. This is called once at construction and
cached, so it must not depend on epoch or state.

**`measurement_dim() -> int`** -- return the dimension of the measurement vector. Also
called once and cached.

**`name() -> str`** -- return a human-readable name. This appears in `FilterRecord` entries
and is useful for filtering residuals by model type.

## Analytical Jacobian (Optional)

By default, the measurement Jacobian $H = \partial h / \partial \mathbf{x}$ is computed via
central finite differences using the same perturbation strategy as the propagator Jacobians.
This calls your `predict()` method $2n$ times (where $n$ is the state dimension), which
works but adds Python function-call overhead.

To provide an analytical Jacobian, override `jacobian()` and return a 2D numpy array:

```python
class RangeModelAnalytical(bh.MeasurementModel):
    def __init__(self, station_eci, sigma):
        super().__init__()
        self.station_eci = np.array(station_eci)
        self.sigma = sigma

    def predict(self, epoch, state):
        return np.array([np.linalg.norm(state[:3] - self.station_eci)])

    def jacobian(self, epoch, state):
        diff = state[:3] - self.station_eci
        r = np.linalg.norm(diff)
        h = np.zeros((1, len(state)))
        h[0, :3] = diff / r
        return h

    def noise_covariance(self):
        return np.array([[self.sigma**2]])

    def measurement_dim(self):
        return 1

    def name(self):
        return "RangeAnalytical"
```

Return `None` from `jacobian()` to explicitly request the finite-difference fallback.

## Mixing Models

A single EKF can use multiple measurement models -- both built-in and custom. Each
`Observation` carries a `model_index` that selects which model processes it:

```python
ekf = bh.ExtendedKalmanFilter(
    epoch, state, p0,
    measurement_models=[
        bh.InertialPositionMeasurementModel(10.0),  # index 0
        RangeModel(station, 100.0),                  # index 1
    ],
    propagation_config=bh.NumericalPropagationConfig.default(),
    force_config=bh.ForceModelConfig.two_body(),
)

# Position observation uses model 0
obs_pos = bh.Observation(t1, position_measurement, model_index=0)

# Range observation uses model 1
obs_range = bh.Observation(t2, np.array([range_km]), model_index=1)
```

Built-in models passed to the EKF execute entirely in Rust with no Python overhead. Custom
Python models incur GIL acquisition on each `predict()` and `jacobian()` call. For
performance-critical applications with many observations, consider implementing custom
models in Rust via the `MeasurementModel` trait.

---

## See Also

- [Measurement Models](measurement_models.md) -- Built-in GPS-like measurement types
- [Extended Kalman Filter](extended_kalman_filter.md) -- EKF setup and processing
- [MeasurementModel API Reference](../../library_api/estimation/common_types.md) -- Base class documentation
