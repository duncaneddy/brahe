# Estimation Plots

Brahe provides estimation-specific plotting functions for visualizing filter performance: state estimation errors with $n\sigma$ covariance bounds, state values with uncertainty patches, measurement residuals (prefit, postfit, and RMS), and marginal distributions with covariance ellipses. All functions support multiple solver overlays for comparing filters, configurable grid layouts, and both matplotlib and plotly backends.

!!! tip "Switching Backends"
    All estimation plot functions accept a `backend=` parameter. Use `backend="plotly"` for interactive exploration and `backend="matplotlib"` for publication-quality static figures.

## State Error Grid

The state error grid shows the difference between estimated and true state values across all state components. When a sigma level is provided, covariance-derived uncertainty bands indicate the filter's confidence — if the error stays within the bounds, the filter is consistent.

### Interactive State Error Grid (Plotly)

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/estimation_state_error_plotly_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/estimation_state_error_plotly_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="estimation_state_error_plotly.py"
    --8<-- "./plots/learn/plots/estimation_state_error_plotly.py"
    ```

### Static State Error Grid (Matplotlib)

<figure markdown="span">
    ![State Error Grid](../../figures/estimation_state_error_matplotlib_light.svg#only-light)
    ![State Error Grid](../../figures/estimation_state_error_matplotlib_dark.svg#only-dark)
</figure>

??? "Plot Source"

    ``` python title="estimation_state_error_matplotlib.py"
    --8<-- "./plots/learn/plots/estimation_state_error_matplotlib.py"
    ```

The 2x3 grid layout (configurable via `ncols`) shows each state component in its own subplot. The error line should converge toward zero as the filter processes more observations. Covariance bands that shrink over time indicate the filter is gaining confidence in its estimate.

### Comparing Filters (EKF vs UKF)

To compare multiple filters on the same grid, pass a list of solvers. This example runs both an EKF and UKF on identical observation data and overlays their state errors:

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/estimation_ekf_ukf_comparison_plotly_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/estimation_ekf_ukf_comparison_plotly_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="estimation_ekf_ukf_comparison_plotly.py"
    --8<-- "./plots/learn/plots/estimation_ekf_ukf_comparison_plotly.py"
    ```

<figure markdown="span">
    ![EKF vs UKF Comparison](../../figures/estimation_ekf_ukf_comparison_matplotlib_light.svg#only-light)
    ![EKF vs UKF Comparison](../../figures/estimation_ekf_ukf_comparison_matplotlib_dark.svg#only-dark)
</figure>

??? "Plot Source"

    ``` python title="estimation_ekf_ukf_comparison_matplotlib.py"
    --8<-- "./plots/learn/plots/estimation_ekf_ukf_comparison_matplotlib.py"
    ```

## State Value Grid

The state value grid plots the actual estimated state values with a dashed truth reference line. Optional uncertainty patches show the $\pm n\sigma$ envelope around the estimate — useful for seeing how the estimated trajectory tracks the truth.

### Interactive State Value Grid (Plotly)

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/estimation_state_value_plotly_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/estimation_state_value_plotly_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="estimation_state_value_plotly.py"
    --8<-- "./plots/learn/plots/estimation_state_value_plotly.py"
    ```

### Static State Value Grid (Matplotlib)

<figure markdown="span">
    ![State Value Grid](../../figures/estimation_state_value_matplotlib_light.svg#only-light)
    ![State Value Grid](../../figures/estimation_state_value_matplotlib_dark.svg#only-dark)
</figure>

??? "Plot Source"

    ``` python title="estimation_state_value_matplotlib.py"
    --8<-- "./plots/learn/plots/estimation_state_value_matplotlib.py"
    ```

## Measurement Residuals

Residual plots show how well the estimated state explains the observations. Pre-fit residuals ($\mathbf{z} - h(\hat{\mathbf{x}}^-)$) reflect the prediction quality; post-fit residuals ($\mathbf{z} - h(\hat{\mathbf{x}}^+)$) show how much unexplained measurement error remains after the update. When `residual_type="both"`, prefit and postfit are overlaid with distinct colors and marker styles for direct visual comparison. Sequential filters (EKF, UKF) show the clearest prefit/postfit separation since each observation has a distinct predict→update step.

### Interactive Residual Plot (Plotly)

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/estimation_residuals_plotly_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/estimation_residuals_plotly_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="estimation_residuals_plotly.py"
    --8<-- "./plots/learn/plots/estimation_residuals_plotly.py"
    ```

### Static Residual Plot (Matplotlib)

<figure markdown="span">
    ![Measurement Residuals](../../figures/estimation_residuals_matplotlib_light.svg#only-light)
    ![Measurement Residuals](../../figures/estimation_residuals_matplotlib_dark.svg#only-dark)
</figure>

??? "Plot Source"

    ``` python title="estimation_residuals_matplotlib.py"
    --8<-- "./plots/learn/plots/estimation_residuals_matplotlib.py"
    ```

### RMS Residuals

The RMS residual view compresses per-component residuals into a single scalar per epoch — the root mean square across all measurement components. This is useful for tracking overall measurement fit quality over time.

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/estimation_residual_rms_plotly_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/estimation_residual_rms_plotly_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="estimation_residual_rms_plotly.py"
    --8<-- "./plots/learn/plots/estimation_residual_rms_plotly.py"
    ```

<figure markdown="span">
    ![Residual RMS](../../figures/estimation_residual_rms_matplotlib_light.svg#only-light)
    ![Residual RMS](../../figures/estimation_residual_rms_matplotlib_dark.svg#only-dark)
</figure>

??? "Plot Source"

    ``` python title="estimation_residual_rms_matplotlib.py"
    --8<-- "./plots/learn/plots/estimation_residual_rms_matplotlib.py"
    ```

## Marginal Distributions

The marginal distribution plot shows the joint uncertainty between two state components as a covariance ellipse, with optional marginal density curves on the top and right axes. This visualization is useful for understanding correlation structure and comparing uncertainty representations from different estimation methods. The `scatter_points` parameter overlays Monte Carlo samples for visual comparison against the analytical covariance ellipse.

### Interactive Marginal Plot (Plotly)

<div class="plotly-embed tall">
  <iframe class="only-light" src="../../figures/estimation_marginal_plotly_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/estimation_marginal_plotly_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="estimation_marginal_plotly.py"
    --8<-- "./plots/learn/plots/estimation_marginal_plotly.py"
    ```

### Static Marginal Plot (Matplotlib)

<figure markdown="span">
    ![Marginal Distribution](../../figures/estimation_marginal_matplotlib_light.svg#only-light)
    ![Marginal Distribution](../../figures/estimation_marginal_matplotlib_dark.svg#only-dark)
</figure>

??? "Plot Source"

    ``` python title="estimation_marginal_matplotlib.py"
    --8<-- "./plots/learn/plots/estimation_marginal_matplotlib.py"
    ```

The `state_indices` parameter selects which pair of state components to visualize — for example, `(0, 1)` for X-Y position or `(3, 4)` for Vx-Vy velocity.

---

## See Also

- [Estimation State Plots API](../../library_api/plots/estimation_state.md) -- Full function signatures and parameters
- [Measurement Residual Plots API](../../library_api/plots/estimation_residuals.md) -- Residual plot reference
- [Marginal Distribution Plots API](../../library_api/plots/estimation_marginal.md) -- Marginal plot reference
- [Extended Kalman Filter](../estimation/extended_kalman_filter.md) -- EKF setup and usage
- [Batch Least Squares](../estimation/batch_least_squares.md) -- BLS diagnostics and residuals
- [Plotting Overview](index.md) -- Backend system and general plotting guide
