# Integrator Configuration

The `NumericalPropagationConfig` controls the numerical integration method, step sizes, and error tolerances. Brahe provides preset configurations for common scenarios and allows custom configurations for specific requirements.

For API details, see the [NumericalPropagationConfig API Reference](../../../library_api/propagators/numerical_propagation_config.md). For detailed information about integrator theory and low-level usage, see the [Numerical Integration](../../integrators/index.md) guide.

## Full Example

Here is a complete example creating a `NumericalPropagationConfig` exercising all available configuration options:

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/integrator_overview.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/integrator_overview.rs:4"
    ```

## Architecture Overview

### Configuration Hierarchy

`NumericalPropagationConfig` is the top-level container that aggregates all integrator settings. Each component has its own configuration struct:

``` .no-linenums
NumericalPropagationConfig
├── method: IntegratorMethod
│   ├── RK4 (fixed step)
│   ├── RKF45 (adaptive)
│   ├── DP54 (adaptive, default)
│   └── RKN1210 (adaptive, high precision)
├── integrator: IntegratorConfig
│   ├── abs_tol, rel_tol
│   ├── initial_step, min_step, max_step
│   ├── step_safety_factor
│   ├── min/max_step_scale_factor
│   └── fixed_step_size (for RK4)
└── variational: VariationalConfig
    ├── enable_stm, enable_sensitivity
    ├── store_stm_history, store_sensitivity_history
    └── jacobian_method, sensitivity_method
```

The configuration is captured at propagator construction time and remains immutable during propagation.

## Integration Methods

Four integration methods are available:

<div class="center-table" markdown="1">
| Method | Order | Adaptive | Function Evals | Description |
|------|-----|--------|--------------|-----------|
| RK4 | 4 | No | 4 | Classic fixed-step Runge-Kutta |
| RKF45 | 4(5) | Yes | 6 | Runge-Kutta-Fehlberg adaptive |
| DP54 | 5(4) | Yes | 6-7 | Dormand-Prince (MATLAB ode45) |
| RKN1210 | 12(10) | Yes | 17 | High-precision Runge-Kutta-Nystrom |
</div>

### RK4 (Fixed Step)

Classic 4th-order Runge-Kutta with fixed step size. No error control - requires careful step size selection.

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/integrator_method_rk4.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/integrator_method_rk4.rs:4"
    ```

### DP54 (Default)

Dormand-Prince 5(4) adaptive method. Uses FSAL (First-Same-As-Last) optimization for efficiency. MATLAB's `ode45` uses this method.

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/integrator_method_dp54.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/integrator_method_dp54.rs:4"
    ```

### RKN1210 (High Precision)

12th-order Runge-Kutta-Nystrom optimized for second-order ODEs like orbital mechanics. Achieves extreme accuracy with tight tolerances.

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/integrator_method_rkn1210.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/integrator_method_rkn1210.rs:4"
    ```

## Error Tolerances

Adaptive integrators adjust step size to keep error within:

$$
\text{error} < \text{abs\_tol} + \text{rel\_tol} \times |\text{state}|
$$

- **`abs_tol`**: Bounds error when state components are small (default: 1e-6)
- **`rel_tol`**: Bounds error proportional to state magnitude (default: 1e-3)

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/integrator_tolerances.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/integrator_tolerances.rs:4"
    ```

## Customizing Configuration

### Python Builder Pattern

Python supports method chaining to customize from a preset:

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/integrator_builder_pattern.py:8"
    ```

### Rust Struct Syntax

In Rust, use struct update syntax (`..`) to customize from defaults:

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/integrator_builder_pattern.rs:4"
    ```

## Preset Configurations

Brahe provides preset configurations for common use cases:

<div class="center-table" markdown="1">
| Preset | Method | abs_tol | rel_tol | Description |
|------|------|-------|-------|-----------|
| `default()` | DP54 | 1e-6 | 1e-3 | General purpose |
| `high_precision()` | RKN1210 | 1e-10 | 1e-8 | Maximum accuracy |
| `with_method(M)` | M | 1e-6 | 1e-3 | Custom method with defaults |
</div>

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/integrator_presets.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/integrator_presets.rs:4"
    ```

## Variational Equations

The propagator can optionally integrate variational equations to compute the State Transition Matrix (STM) and sensitivity matrices. This is enabled via `VariationalConfig`:

- **`enable_stm`**: Compute the State Transition Matrix
- **`enable_sensitivity`**: Compute parameter sensitivity matrix
- **`store_*_history`**: Store matrices at output times in trajectory
- **`jacobian_method`/`sensitivity_method`**: Finite difference method (Forward, Backward, Central)

The STM maps initial state perturbations to final state perturbations: $\delta\mathbf{x}(t) = \Phi(t, t_0) \cdot \delta\mathbf{x}(t_0)$

See [Covariance and Sensitivity](covariance_sensitivity.md) for detailed usage.

---

## See Also

- [Numerical Propagation Overview](index.md) - Architecture and concepts
- [Force Models](force_models.md) - Configuring force models
- [Covariance and Sensitivity](covariance_sensitivity.md) - Variational equations
- [Integrators](../../integrators/index.md) - Detailed integrator documentation
- [NumericalPropagationConfig API Reference](../../../library_api/propagators/numerical_propagation_config.md)
