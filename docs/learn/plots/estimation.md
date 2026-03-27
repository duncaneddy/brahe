# Estimation Plots

Brahe provides estimation-specific plotting functions for visualizing filter performance: state estimation errors with $n\sigma$ covariance bounds, state values with uncertainty patches, measurement residuals (prefit, postfit, and RMS), and marginal distributions with covariance ellipses. All functions support multiple solver overlays for comparing filters, configurable grid layouts, and both matplotlib and plotly backends.

## State Error Grid

The state error grid shows the difference between estimated and true state values across all state components. When a sigma level is provided, covariance-derived uncertainty bands indicate the filter's confidence — if the error stays within the bounds, the filter is consistent.

=== "Matplotlib"

    ``` python
    --8<-- "./examples/estimation/plot_state_error_grid_matplotlib.py:11"
    ```

=== "Plotly"

    ``` python
    --8<-- "./examples/estimation/plot_state_error_grid_plotly.py:11"
    ```

??? example "Output"
    === "Matplotlib"
        ```
        --8<-- "./docs/outputs/estimation/plot_state_error_grid_matplotlib.py.txt"
        ```

    === "Plotly"
        ```
        --8<-- "./docs/outputs/estimation/plot_state_error_grid_plotly.py.txt"
        ```

The 2x3 grid layout (configurable via `ncols`) shows each state component in its own subplot. The error line should converge toward zero as the filter processes more observations. Covariance bands that shrink over time indicate the filter is gaining confidence in its estimate.

To compare multiple filters on the same grid, pass a list of solvers:

```python
fig = bh.plot_estimator_state_error_grid(
    solvers=[ekf, ukf],
    true_trajectory=truth_traj,
    sigma=3,
    labels=["EKF", "UKF"],
    colors=["blue", "red"],
)
```

## State Value Grid

The state value grid plots the actual estimated state values with a dashed truth reference line. Optional uncertainty patches show the $\pm n\sigma$ envelope around the estimate — useful for seeing how the estimated trajectory tracks the truth.

=== "Matplotlib"

    ``` python
    --8<-- "./examples/estimation/plot_state_value_grid_matplotlib.py:11"
    ```

=== "Plotly"

    ``` python
    --8<-- "./examples/estimation/plot_state_value_grid_plotly.py:11"
    ```

??? example "Output"
    === "Matplotlib"
        ```
        --8<-- "./docs/outputs/estimation/plot_state_value_grid_matplotlib.py.txt"
        ```

    === "Plotly"
        ```
        --8<-- "./docs/outputs/estimation/plot_state_value_grid_plotly.py.txt"
        ```

## Measurement Residuals

Residual plots show how well the estimated state explains the observations. Pre-fit residuals ($\mathbf{z} - h(\hat{\mathbf{x}}^-)$) reflect the prediction quality; post-fit residuals ($\mathbf{z} - h(\hat{\mathbf{x}}^+)$) show how much unexplained measurement error remains after the update. When `residual_type="both"`, prefit and postfit are overlaid with distinct marker styles.

=== "Matplotlib"

    ``` python
    --8<-- "./examples/estimation/plot_residuals_matplotlib.py:11"
    ```

=== "Plotly"

    ``` python
    --8<-- "./examples/estimation/plot_residuals_plotly.py:11"
    ```

??? example "Output"
    === "Matplotlib"
        ```
        --8<-- "./docs/outputs/estimation/plot_residuals_matplotlib.py.txt"
        ```

    === "Plotly"
        ```
        --8<-- "./docs/outputs/estimation/plot_residuals_plotly.py.txt"
        ```

### RMS Residuals

The RMS residual view compresses per-component residuals into a single scalar per epoch — the root mean square across all measurement components. This is useful for tracking overall measurement fit quality over time.

=== "Matplotlib"

    ``` python
    --8<-- "./examples/estimation/plot_residual_rms_matplotlib.py:11"
    ```

=== "Plotly"

    ``` python
    --8<-- "./examples/estimation/plot_residual_rms_plotly.py:11"
    ```

??? example "Output"
    === "Matplotlib"
        ```
        --8<-- "./docs/outputs/estimation/plot_residual_rms_matplotlib.py.txt"
        ```

    === "Plotly"
        ```
        --8<-- "./docs/outputs/estimation/plot_residual_rms_plotly.py.txt"
        ```

## Marginal Distributions

The marginal distribution plot shows the joint uncertainty between two state components as a covariance ellipse, with optional marginal density curves on the top and right axes. This visualization is useful for understanding correlation structure and comparing uncertainty representations from different estimation methods.

=== "Matplotlib"

    ``` python
    --8<-- "./examples/estimation/plot_marginal_matplotlib.py:11"
    ```

=== "Plotly"

    ``` python
    --8<-- "./examples/estimation/plot_marginal_plotly.py:11"
    ```

??? example "Output"
    === "Matplotlib"
        ```
        --8<-- "./docs/outputs/estimation/plot_marginal_matplotlib.py.txt"
        ```

    === "Plotly"
        ```
        --8<-- "./docs/outputs/estimation/plot_marginal_plotly.py.txt"
        ```

The `scatter_points` parameter overlays Monte Carlo samples for visual comparison against the analytical covariance ellipse. The `state_indices` parameter selects which pair of state components to visualize — for example, `(0, 1)` for X-Y position or `(3, 4)` for Vx-Vy velocity.

## Raw Array Interface

All solver-API functions have `_from_arrays` counterparts that accept pre-computed numpy arrays instead of solver objects. This is useful when working with custom estimators or external tools:

```python
fig = bh.plot_estimator_state_error_grid_from_arrays(
    times=[t_ekf, t_ukf],         # list of 1D time arrays
    errors=[err_ekf, err_ukf],     # list of (N, n_states) error arrays
    sigmas=[sig_ekf, sig_ukf],     # list of (N, n_states) sigma arrays
    labels=["EKF", "UKF"],
    state_labels=["X [m]", "Y [m]", "Z [m]", "Vx [m/s]", "Vy [m/s]", "Vz [m/s]"],
    ncols=3,
)
```

## Time Axis Options

All functions support flexible time axis configuration:

```python
# Elapsed seconds (default)
fig = bh.plot_estimator_state_error_grid(solvers=[ekf], ..., time_units="seconds")

# Other presets
fig = bh.plot_estimator_state_error_grid(solvers=[ekf], ..., time_units="minutes")
fig = bh.plot_estimator_state_error_grid(solvers=[ekf], ..., time_units="hours")

# Orbital periods (requires orbital_period in seconds)
fig = bh.plot_estimator_state_error_grid(
    solvers=[ekf], ...,
    time_units="orbits",
    orbital_period=bh.orbital_period(bh.R_EARTH + 500e3),
)

# Custom transform via callable
fig = bh.plot_estimator_state_error_grid(
    solvers=[ekf], ...,
    time_units=lambda epochs: (np.array([(e - epochs[0]) / 3600 for e in epochs]), "Time [hr]"),
)
```

---

## See Also

- [Estimation State Plots API](../../library_api/plots/estimation_state.md) -- Full function signatures and parameters
- [Measurement Residual Plots API](../../library_api/plots/estimation_residuals.md) -- Residual plot reference
- [Marginal Distribution Plots API](../../library_api/plots/estimation_marginal.md) -- Marginal plot reference
- [Extended Kalman Filter](../estimation/extended_kalman_filter.md) -- EKF setup and usage
- [Batch Least Squares](../estimation/batch_least_squares.md) -- BLS diagnostics and residuals
- [Plotting Overview](index.md) -- Backend system and general plotting guide
