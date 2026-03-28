# Batch Least Squares

Batch Least Squares (BLS) processes an entire arc of observations simultaneously, iterating
to minimize the weighted sum of squared residuals. Unlike the EKF and UKF which update
sequentially, BLS re-linearizes the full problem at each iteration, making it the standard
approach for offline orbit determination when the complete observation set is available.

## Golden-Path Example

=== "Python"
    ``` python
    --8<-- "./examples/estimation/bls_position_tracking.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/estimation/bls_position_tracking.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/estimation/bls_position_tracking.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/estimation/bls_position_tracking.rs.txt"
        ```

## How It Works

BLS uses Gauss-Newton iteration to refine the state estimate. Starting from an initial
guess $\mathbf{x}_0$, the algorithm propagates the state to each observation epoch,
computes predicted measurements $h(\mathbf{x}, t_i)$, and forms the residual vector
$\mathbf{y}_i = \mathbf{z}_i - h(\mathbf{x}, t_i)$. The State Transition Matrix $\Phi$
maps state perturbations from the reference epoch to each observation time, allowing the
measurement Jacobians to be expressed in terms of the state correction at the reference
epoch.

At each iteration, the algorithm solves a linear least squares problem for the state
correction $\delta\mathbf{x}$ that minimizes the weighted residual cost function:

$$
J = \frac{1}{2} \left[ \delta\mathbf{x}^T P_0^{-1} \delta\mathbf{x} + \sum_i (\mathbf{y}_i - \tilde{H}_i \, \delta\mathbf{x})^T R_i^{-1} (\mathbf{y}_i - \tilde{H}_i \, \delta\mathbf{x}) \right]
$$

where $\tilde{H}_i = H_i \Phi(t_i, t_0)$ maps the state correction at the reference epoch
to the observation space at time $t_i$, and $P_0$ is the a priori covariance. The state is
updated, the trajectory is re-propagated from the corrected initial conditions, and the
process repeats until convergence.

## Solver Formulations

Brahe provides two equivalent formulations for solving the linear system at each iteration.

**Normal Equations** (default) accumulates the information matrix
$\Lambda = P_0^{-1} + \sum_i \tilde{H}_i^T R_i^{-1} \tilde{H}_i$ and normal vector
$\mathbf{N} = -P_0^{-1} \delta\mathbf{x}_0 + \sum_i \tilde{H}_i^T R_i^{-1} \mathbf{y}_i$,
then solves $\Lambda \, \delta\mathbf{x} = \mathbf{N}$ via Cholesky decomposition. This
requires only $O(n^2)$ memory where $n$ is the state dimension, making it efficient for
large observation sets.

**Stacked Observation Matrix** builds the full $\tilde{H}$ matrix and residual vector
across all observations, then solves via QR decomposition. This uses $O(m \times n)$ memory
where $m$ is the total measurement dimension, but provides better numerical conditioning
when the problem is poorly scaled or near-singular.

## Consider Parameters

When some state elements are uncertain but should not be estimated (e.g., drag coefficient,
sensor biases), the consider parameter formulation accounts for their effect on the
solution covariance without including them in the solve. The state vector is partitioned
into solve-for parameters (first $n_s$ elements) and consider parameters (remaining $n_c$
elements). The solve only corrects the solve-for parameters, but the final covariance
reflects the additional uncertainty from the consider parameters through the sensitivity
matrix $S_c$.

Configure consider parameters via `ConsiderParameterConfig`, specifying `n_solve` (the
number of solve-for parameters) and `consider_covariance` (the $n_c \times n_c$ a priori
covariance for the consider parameters).

## Convergence Criteria

BLS supports two convergence criteria, configured via `BLSConfig`:

**State correction norm** (`state_correction_threshold`) declares convergence when
$\|\delta\mathbf{x}\|$ falls below the threshold. This is enabled by default with a
threshold of $10^{-8}$. It directly measures whether the state estimate has stabilized.

**Relative cost change** (`cost_convergence_threshold`) declares convergence when
$|\Delta J| / |J|$ falls below the threshold. This is useful when the absolute state
correction is large but the cost function is no longer improving meaningfully.

If both are set, the solver converges when either criterion is satisfied. The solver also
terminates when `max_iterations` is reached, regardless of convergence.

## Diagnostics

After calling `solve()`, iteration-level diagnostics are available through
`iteration_records()`, which returns a `BLSIterationRecord` for each iteration containing
the cost function value, state correction norm, and RMS residuals. Monitoring the cost
across iterations confirms that the solver is converging (decreasing cost) and has not
stalled.

Per-observation residuals are available through `observation_residuals()`, returning
`BLSObservationResidual` entries with pre-fit and post-fit residuals for each measurement.
Large post-fit residuals can indicate outlier observations or an incorrect measurement model.
Systematic patterns in residuals across time may reveal unmodeled dynamics.

The `converged()`, `iterations_completed()`, and `final_cost()` accessors provide summary
status without inspecting individual records.

---

## See Also

- [Extended Kalman Filter](extended_kalman_filter.md) -- Sequential EKF for real-time estimation
- [Unscented Kalman Filter](unscented_kalman_filter.md) -- UKF sigma-point estimation
- [Measurement Models](measurement_models.md) -- Built-in and custom measurement types
- [BLS API Reference](../../library_api/estimation/batch_least_squares.md) -- Complete method documentation
- [Covariance and Sensitivity](../orbit_propagation/numerical_propagation/covariance_sensitivity.md) -- STM propagation details
