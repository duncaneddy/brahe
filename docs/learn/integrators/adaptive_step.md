# Adaptive-Step Integrators

Adaptive-step integrators automatically adjust their step size to maintain a specified error tolerance. This makes them efficient and reliable for problems where the optimal step size isn't known in advance or varies during integration. They are particularly useful for orbital mechanics, where dynamics can change rapidly due to close encounters or perturbations. In elliptical orbits, for example, smaller steps are needed near periapsis to capture rapid motion, while taking larger steps near apoapsis is acceptable and saves computation.

## How Adaptive Stepping Works

Adaptive methods estimate the local truncation error at each step by computing two solutions of different orders:

1. **Higher-order solution** (order $p$): More accurate, used for propagation
2. **Lower-order solution** (order $p-1$): Less accurate, used for error estimation

The **error estimate** is the difference between these solutions:

$$
\varepsilon \approx \|\mathbf{x}_p - \mathbf{x}_{p-1}\|
$$

This is compared against a **tolerance**:

$$
\text{tol} = \text{abs\_tol} + \text{rel\_tol} \times \|\mathbf{x}\|
$$

- **If $\varepsilon < \text{tol}$**: Step accepted, state advances
- **If $\varepsilon \geq \text{tol}$**: Step rejected, retry with smaller $h$

## Available Adaptive Integrators

### RKF45: Runge-Kutta-Fehlberg 4(5)

An embedded Runge-Kutta method using 5th-order solution for propagation and 4th-order solution for error estimation.

### DP54: Dormand-Prince 5(4)

An embedded Runge-Kutta method widely used in scientific computing (e.g., MATLAB's `ode45`).

### RKN1210: Runge-Kutta-Nyström 12(10)

A high-order method specialized for second-order differential equations, particularly well-suited to orbital mechanics.

**Requirements:**

- State dimension must be even (position and velocity components)
- Best suited for problems naturally expressed as second-order systems (e.g., $\mathbf{F} = m\mathbf{a}$)

## Basic Usage

=== "Python"

    ``` python
    --8<-- "./examples/integrators/adaptive_stepping.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/integrators/adaptive_stepping.rs:4"
    ```

## Step Size Control Algorithm

After computing error estimate $\varepsilon$, the integrator calculates a new step size:

$$h_{\text{new}} = \text{safety\_factor} \times h \times \left(\frac{\text{tol}}{\varepsilon}\right)^{1/p}$$

where:
- **safety_factor**: Conservative multiplier (default 0.9)
- **$p$**: Order of error estimate
- **$h$**: Current step size

This is clamped to reasonable bounds:

$$h_{\text{new}} = \text{clip}(h_{\text{new}}, \text{min\_scale} \times h, \text{max\_scale} \times h)$$

and absolute limits:

$$h_{\text{new}} = \text{clip}(h_{\text{new}}, h_{\text{min}}, h_{\text{max}})$$

### Control Parameters

From `IntegratorConfig`:

- `abs_tol`: Absolute error tolerance (default 1e-10)
- `rel_tol`: Relative error tolerance (default 1e-9)
- `min_step`: Minimum allowed step size (default 1e-12 s)
- `max_step`: Maximum allowed step size (default 900 s)
- `step_safety_factor`: Safety margin (default 0.9)
- `min_step_scale_factor`: Min step change ratio (default 0.2)
- `max_step_scale_factor`: Max step change ratio (default 10.0)
- `max_step_attempts`: Max tries before error (default 10)

## Highly Elliptical Orbit Example

The following example demonstrates propagating a highly elliptical orbit (HEO) using the RKN1210 adaptive-step integrator with tight tolerances for high precision.

=== "Python"

    ``` python
    --8<-- "./examples/integrators/high_precision.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/integrators/high_precision.rs:4"
    ```

## See Also

- **[Configuration Guide](configuration.md)** - Detailed parameter tuning
- **[Fixed-Step Integrators](fixed_step.md)** - For comparison
- **[RKF45 API Reference](../../library_api/integrators/rkf45.md)** - RKF45 documentation
- **[DP54 API Reference](../../library_api/integrators/dp54.md)** - DP54 documentation
- **[RKN1210 API Reference](../../library_api/integrators/rkn1210.md)** - RKN1210 documentation
- **[Integrators Overview](index.md)** - Comparison of all integrators
