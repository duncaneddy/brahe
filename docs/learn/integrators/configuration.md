# Integrator Configuration

Proper integrator configuration is essential for balancing accuracy, performance, and reliability. This guide explains all configuration parameters and how to choose appropriate values.

## IntegratorConfig Class

The [`IntegratorConfig`](../../library_api/integrators/config.md) class encapsulates all settings for adaptive-step integrators. Key parameters include:

- **Error Tolerances**: `abs_tol`, `rel_tol`
- **Step Size Limits**: `min_step`, `max_step`
- **Step Size Control**: `step_safety_factor`, `min_step_scale_factor`, `max_step_scale_factor`
- **Maximum Step Attempts**: `max_step_attempts`

## Configuration Parameters

### Error Tolerances

**`abs_tol`** (float): Absolute error tolerance

- Controls maximum absolute error allowed per step
- Units match state units (meters for position, m/s for velocity)
- Prevents excessively small steps when state approaches zero

**`rel_tol`** (float): Relative error tolerance

- Controls maximum relative error as fraction of state magnitude
- Dimensionless
- Scales with state magnitude

**Combined tolerance:**

$$\text{tol}_i = \text{abs\_tol} + \text{rel\_tol} \times |x_i|$$

### Step Size Limits

**`min_step`** (float): Minimum allowed step size (seconds)

- Safety limit preventing infinitesimally small steps
- If integrator hits this limit repeatedly, tolerances may be too tight

**`max_step`** (float): Maximum allowed step size (seconds)

- Prevents missing important dynamics by taking too-large steps
- Critical for problems with events or discontinuities

### Step Size Control

**`step_safety_factor`** (float): Safety margin for step size adjustment

- Multiplier applied to calculated optimal step size
- Makes step size more conservative
- Default: 0.9 (use 90% of optimal)
- Range: 0.8 to 0.95

**Formula:**

$$
h_{\text{new}} = \text{safety\_factor} \times h \times \left(\frac{\text{tol}}{\varepsilon}\right)^{1/p}
$$

Decreasing the safety factor results in smaller steps and higher accuracy but more function evaluations. Increasing it yields larger steps and faster performance but risks exceeding error tolerances and more rejections, which in turn results in wasted computation.

**`min_step_scale_factor`** (float): Minimum step size change ratio

- Prevents dramatic step size reductions
- Ensures step doesn't shrink too rapidly
- Default: 0.2 (can shrink to 20% of current)

**`max_step_scale_factor`** (float): Maximum step size change ratio

- Prevents dramatic step size increases
- Ensures gradual adaptation
- Default: 10.0 (can grow to 10× current)

**Why limit step changes:**
- Prevents oscillating step sizes
- Smooths adaptation

### Step Attempts

**`max_step_attempts`** (int): Maximum retry attempts before error

- If step rejected more than this many times, raise error
- Prevents infinite loops with pathological problems
- Default: 10

**Typical causes of many rejections:**
1. Tolerances too tight for integrator order
2. Stiff differential equations
3. Discontinuity in dynamics
4. Bug in dynamics function

## Configuration Examples

These examples illustrate different parameter combinations representing different points on the accuracy-performance spectrum:

### Conservative Configuration

Tight tolerances and restrictive step size controls:

=== "Python"

    ``` python
    --8<-- "./examples/integrators/configuration_examples.py:conservative"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/integrators/configuration_examples.rs:conservative"
    ```

### Balanced Configuration

Moderate settings suitable for many applications:

=== "Python"

    ``` python
    --8<-- "./examples/integrators/configuration_examples.py:balanced"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/integrators/configuration_examples.rs:balanced"
    ```

### Aggressive Configuration

Relaxed tolerances for faster computation:

=== "Python"

    ``` python
    --8<-- "./examples/integrators/configuration_examples.py:aggressive"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/integrators/configuration_examples.rs:aggressive"
    ```

### High-Precision Configuration

Very tight tolerances for high-accuracy needs:

=== "Python"

    ``` python
    --8<-- "./examples/integrators/configuration_examples.py:high_precision_config"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/integrators/configuration_examples.rs:high_precision"
    ```

## Tuning Strategy

### 1. Start with Defaults

```python
config = bh.IntegratorConfig.adaptive(abs_tol=1e-10, rel_tol=1e-9)
integrator = bh.DP54Integrator(dynamics, config)
```

### 2. Assess Performance

Run test integration and monitor:
- Number of steps taken
- Number of rejected steps (should be < 1%)
- Error estimates
- Step size variation

### 3. Adjust Based on Observations

**If steps too small:**
```python
# Relax tolerances by 10×
config.abs_tol = 1e-9
config.rel_tol = 1e-8
```

**If missing features:**
```python
# Reduce max step
config.max_step = orbital_period / 50
```

**If many rejections:**
```python
# Decrease safety factor
config.step_safety_factor = 0.7

# Or reduce step scale factors
config.max_step_scale_factor = 5.0
```

**If hitting min_step:**
```python
# Switch to higher-order integrator or relax tolerances
integrator = bh.RKN1210Integrator(dynamics, config)
```

### 4. Validate

Compare against:

- Analytical solution (if available)
- Same integration with 10× tighter tolerances
- Energy/momentum conservation
- Independent integration software

### 5. Document

Record final configuration with rationale:
```python
# Configuration tuned for LEO orbit propagation
# - Tolerances provide ~5m position accuracy over 1 day
# - Max step prevents missing station-keeping maneuvers
# - Validated against analytical two-body solution
config = bh.IntegratorConfig.adaptive(
    abs_tol=1e-11,
    rel_tol=1e-10,
    max_step=300.0  # 5 minutes
)
```

## See Also

- **[Adaptive-Step Integrators](adaptive_step.md)** - How adaptive integration works
- **[Fixed-Step Integrators](fixed_step.md)** - Fixed-step integration guide
- **[Configuration API Reference](../../library_api/integrators/config.md)** - Complete API documentation
- **[Integrators Overview](index.md)** - Comparison of all integrators
