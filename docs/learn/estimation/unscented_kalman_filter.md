# Unscented Kalman Filter

The Unscented Kalman Filter (UKF) is a sequential estimator that propagates deterministic
sigma points through the true nonlinear dynamics and measurement models, avoiding
linearization entirely. It captures the mean and covariance of the state distribution to
second order without computing Jacobians or a State Transition Matrix.

## Setting Up

The UKF constructor takes the same arguments as the EKF, plus UKF-specific tuning
parameters (alpha, beta, kappa) via `UKFConfig`. It does not require STM propagation.

=== "Python"
    ``` python
    --8<-- "./examples/estimation/ukf_position_tracking.py:11"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/estimation/ukf_position_tracking.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/estimation/ukf_position_tracking.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/estimation/ukf_position_tracking.rs.txt"
        ```

The constructor internally builds a numerical propagator, generates sigma point weights
from the `UKFConfig` parameters, and validates that the initial covariance matches the
state dimension.

## How It Works

At each observation, the UKF performs two steps:

**Predict**: generate 2n+1 sigma points from the current state and covariance (where n is
the state dimension), propagate each through the dynamics to the observation epoch, then
reconstruct the predicted mean and covariance from the propagated points.

**Update**: transform the sigma points through the measurement model, compute the
innovation covariance and cross-covariance, then apply a Kalman-like gain to incorporate
the measurement.

The sigma points are chosen to exactly match the first and second moments of the state
distribution. The `alpha` parameter controls how far the sigma points spread from the mean
(smaller values keep them closer), `beta` encodes prior distribution knowledge (2.0 is
optimal for Gaussian), and `kappa` provides an additional scaling degree of freedom.

## Processing Observations

The UKF has the same API as the EKF:

```python
# One at a time
record = ukf.process_observation(obs)

# Batch (auto-sorts by epoch)
ukf.process_observations(observations)
```

`FilterRecord` fields are identical to the EKF: pre/post-fit states, covariances,
residuals, and gain matrix. The `kalman_gain` field contains the UKF gain analog
$K = P_{xz} S^{-1}$, which has the same interpretation as the EKF gain (maps innovation
to state correction).

## UKF Configuration

```python
config = bh.UKFConfig(
    alpha=1e-3,    # Sigma point spread (default: 1e-3)
    beta=2.0,      # Distribution parameter (default: 2.0, optimal for Gaussian)
    kappa=0.0,     # Scaling parameter (default: 0.0)
)
```

The defaults work well for most orbital mechanics problems. Increase `alpha` if the filter
diverges due to sigma points being too close to the mean.

## EKF vs UKF

The EKF linearizes dynamics and measurements around the current state, computing the State
Transition Matrix and measurement Jacobian. The UKF instead propagates 2n+1 sigma points
through the true nonlinear functions.

**EKF** computes one dynamics propagation per step plus Jacobian evaluations. It requires
STM propagation, which adds overhead to the dynamics integration. Linearization errors
accumulate when dynamics or measurements are strongly nonlinear.

**UKF** computes 2n+1 dynamics propagations per step (13 for a 6D state) but each
propagation is simpler (no STM or variational equations). It captures nonlinearity to
second order and works with non-differentiable measurement models. The trade-off is more
function evaluations per step.

Both filters use the same measurement models, observation types, and filter record format.
Switching between them requires only changing the constructor.

## Filter Equations

The UKF uses deterministic sigma points to propagate the mean and covariance through
nonlinear functions without linearization.

### Sigma Point Generation

Given state dimension $n$, form $2n + 1$ sigma points from the current state
$\mathbf{x}^{+}$ and covariance $P^{+}$:

$$
\boldsymbol{\chi}_0 = \mathbf{x}^{+}
$$

$$
\boldsymbol{\chi}_i = \mathbf{x}^{+} + \left(\sqrt{(n + \lambda)\, P^{+}}\right)_i, \quad i = 1, \ldots, n
$$

$$
\boldsymbol{\chi}_{i+n} = \mathbf{x}^{+} - \left(\sqrt{(n + \lambda)\, P^{+}}\right)_i, \quad i = 1, \ldots, n
$$

where $\lambda = \alpha^2 (n + \kappa) - n$ and $\left(\sqrt{M}\right)_i$ denotes the
$i$-th column of the matrix square root.

### Weights

$$
W_0^{(m)} = \frac{\lambda}{n + \lambda}, \qquad
W_0^{(c)} = \frac{\lambda}{n + \lambda} + (1 - \alpha^2 + \beta)
$$

$$
W_i^{(m)} = W_i^{(c)} = \frac{1}{2(n + \lambda)}, \quad i = 1, \ldots, 2n
$$

### Predict

Propagate each sigma point through the dynamics:

$$
\boldsymbol{\chi}_{i,k}^{-} = f(\boldsymbol{\chi}_{i,k-1}^{+},\; t_{k-1} \to t_k)
$$

Reconstruct the predicted mean and covariance:

$$
\mathbf{x}_{k}^{-} = \sum_{i=0}^{2n} W_i^{(m)} \, \boldsymbol{\chi}_{i,k}^{-}
$$

$$
P_{k}^{-} = \sum_{i=0}^{2n} W_i^{(c)} \left(\boldsymbol{\chi}_{i,k}^{-} - \mathbf{x}_{k}^{-}\right)\left(\boldsymbol{\chi}_{i,k}^{-} - \mathbf{x}_{k}^{-}\right)^T + Q_k
$$

### Update

Transform the predicted sigma points through the measurement model:

$$
\mathbf{z}_{i,k} = h(\boldsymbol{\chi}_{i,k}^{-})
$$

$$
\hat{\mathbf{z}}_k = \sum_{i=0}^{2n} W_i^{(m)} \, \mathbf{z}_{i,k}
$$

**Innovation covariance:**

$$
S_k = \sum_{i=0}^{2n} W_i^{(c)} \left(\mathbf{z}_{i,k} - \hat{\mathbf{z}}_k\right)\left(\mathbf{z}_{i,k} - \hat{\mathbf{z}}_k\right)^T + R
$$

**Cross-covariance:**

$$
P_{xz} = \sum_{i=0}^{2n} W_i^{(c)} \left(\boldsymbol{\chi}_{i,k}^{-} - \mathbf{x}_{k}^{-}\right)\left(\mathbf{z}_{i,k} - \hat{\mathbf{z}}_k\right)^T
$$

**Kalman gain:**

$$
K_k = P_{xz} \, S_k^{-1}
$$

**State and covariance update:**

$$
\mathbf{x}_{k}^{+} = \mathbf{x}_{k}^{-} + K_k \left(\mathbf{z}_k - \hat{\mathbf{z}}_k\right)
$$

$$
P_{k}^{+} = P_{k}^{-} - K_k \, S_k \, K_k^T
$$

---

## See Also

- [Extended Kalman Filter](extended_kalman_filter.md) -- EKF setup and processing
- [Measurement Models](measurement_models.md) -- Built-in and custom measurement types
- [UKF API Reference](../../library_api/estimation/unscented_kalman_filter.md) -- Complete method documentation
