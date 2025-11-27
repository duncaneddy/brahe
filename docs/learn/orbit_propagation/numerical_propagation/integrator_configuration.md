# Integrator Configuration

The `NumericalPropagationConfig` controls the numerical integration method, step sizes, and error tolerances. Choosing appropriate settings balances accuracy against computational cost.

For API details, see the [NumericalPropagationConfig API Reference](../../../library_api/propagators/numerical_propagation_config.md). For detailed information about integrator theory and low-level usage, see the [Numerical Integration](../../integrators/index.md) guide.

## NumericalPropagationConfig

The `NumericalPropagationConfig` class contains three components:

- **method**: The integration method to use (`IntegratorMethod`)
- **integrator**: Tolerances and step size settings (`IntegratorConfig`)
- **variational**: STM and sensitivity matrix settings (`VariationalConfig`)

### Constructor

=== "Python"

    ```python
    config = bh.NumericalPropagationConfig(
        method=bh.IntegratorMethod.DP54,
        integrator=bh.IntegratorConfig.adaptive(1e-9, 1e-6),
        variational=bh.VariationalConfig.default(),
    )
    ```

=== "Rust"

    ```rust
    let config = NumericalPropagationConfig {
        method: IntegratorMethod::DP54,
        integrator: IntegratorConfig::adaptive(1e-9, 1e-6),
        variational: VariationalConfig::default(),
    };
    ```

## IntegratorConfig

The `IntegratorConfig` struct contains settings for error control and step size limits.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `abs_tol` | f64 | 1e-6 | Absolute error tolerance |
| `rel_tol` | f64 | 1e-3 | Relative error tolerance |
| `initial_step` | Option | None | Initial step size (auto if None) |
| `min_step` | Option | 1e-12 | Minimum allowed step size |
| `max_step` | Option | 900.0 | Maximum allowed step size (15 min) |
| `step_safety_factor` | Option | 0.9 | Safety factor for step control |
| `min_step_scale_factor` | Option | 0.2 | Minimum step reduction factor |
| `max_step_scale_factor` | Option | 10.0 | Maximum step growth factor |
| `max_step_attempts` | usize | 10 | Max attempts per step |
| `fixed_step_size` | Option | None | Fixed step for non-adaptive methods |

For comprehensive discussion of these parameters and how they affect integration, see the [Numerical Integration](../../integrators/index.md) guide.

### Factory Methods

**default()**: Standard settings for general use.

**fixed_step(step_size)**: Configure for fixed-step integration.

**adaptive(abs_tol, rel_tol)**: Configure with custom tolerances.

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/integrator_tolerances.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/integrator_tolerances.rs:4"
    ```

## IntegratorMethod

Four integration methods are available:

| Method | Order | Adaptive | Function Evals | Description |
|--------|-------|----------|----------------|-------------|
| RK4 | 4 | No | 4 | Classic fixed-step Runge-Kutta |
| RKF45 | 4(5) | Yes | 6 | Runge-Kutta-Fehlberg adaptive |
| DP54 | 5(4) | Yes | 6-7 | Dormand-Prince (MATLAB ode45) |
| RKN1210 | 12(10) | Yes | 17 | High-precision Runge-Kutta-Nystrom |

### RK4 (Fixed Step)

Classic 4th-order Runge-Kutta with fixed step size. No error control.

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/integrator_method_rk4.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/integrator_method_rk4.rs:4"
    ```

### DP54 (Default)

Dormand-Prince 5(4) adaptive method. The default choice for most applications.

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/integrator_method_dp54.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/integrator_method_dp54.rs:4"
    ```

### RKN1210 (High Precision)

12th-order Runge-Kutta-Nystrom for maximum accuracy.

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/integrator_method_rkn1210.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/integrator_method_rkn1210.rs:4"
    ```

## Builder Pattern

The Python API supports method chaining to customize configuration:

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/integrator_builder_pattern.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/integrator_builder_pattern.rs:4"
    ```

## Preset Configurations

Brahe provides preset configurations for common use cases:

| Preset | Method | abs_tol | rel_tol | Description |
|--------|--------|---------|---------|-------------|
| `default()` | DP54 | 1e-6 | 1e-3 | General purpose |
| `high_precision()` | RKN1210 | 1e-10 | 1e-8 | Maximum accuracy |
| `with_method(M)` | M | 1e-6 | 1e-3 | Custom method with defaults |

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/integrator_presets.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/integrator_presets.rs:4"
    ```

## Variational Equations

The propagator can optionally integrate variational equations to compute the State Transition Matrix (STM). This is enabled via `VariationalConfig`:

```python
var_config = bh.VariationalConfig(
    enable_stm=True,
    enable_sensitivity=False,
    store_stm_history=True,
    store_sensitivity_history=False,
    jacobian_method=bh.DifferenceMethod.CENTRAL,
    sensitivity_method=bh.DifferenceMethod.CENTRAL,
)
```

The STM is useful for:

- Covariance propagation
- Sensitivity analysis
- Orbit determination
- Maneuver optimization

See [Covariance and Sensitivity](covariance_sensitivity.md) for details.

## Performance Impact

Integration method choice significantly affects performance:

| Method | Relative Speed | Relative Accuracy |
|--------|----------------|-------------------|
| RK4 (60s step) | Fastest | Lowest |
| DP54 (default) | Medium | Good |
| RKN1210 (high precision) | Slowest | Highest |

For most applications, DP54 with default tolerances provides the best balance. Use RKN1210 only when sub-millimeter precision is required over extended periods.

---

## See Also

- [Numerical Propagation Overview](index.md) - Architecture and concepts
- [Force Models](force_models.md) - Configuring force models
- [Covariance and Sensitivity](covariance_sensitivity.md) - Variational equations
- [Integrators](../../integrators/index.md) - Detailed integrator documentation
- [NumericalPropagationConfig API Reference](../../../library_api/propagators/numerical_propagation_config.md)
