# Numerical Integration

Numerical integration is fundamental to spacecraft trajectory propagation, orbit determination, and mission planning. Brahe provides multiple integration methods optimized for different accuracy and performance requirements.

!!! warning "Experimental API"
    The integrators module is currently experimental. While the core functionality should be stable, the API may change in future releases as we refine the design and add features.

## What is Numerical Integration?

Numerical integration solves ordinary differential equations (ODEs) of the form:

$$\dot{\mathbf{x}} = \mathbf{f}(t, \mathbf{x})$$

$\mathbf{x}$ is the state vector, typically position and velocity $\mathbf{x} = \begin{bmatrix} \mathbf{p} \\ \mathbf{v} \end{bmatrix}$, and $\mathbf{f}$ defines the dynamics (gravity, perturbations, thrust, etc.). The integrator advances the state forward in time from an initial condition $\mathbf{x}_0$ at time $t_0$ to $\mathbf{x}(t)$ at some future time $t$.

In an ideal world, we would have closed-form analytical solutions for these equations. However, real-world dynamics are often too complex for exact solutions, necessitating numerical methods that approximate the solution. It is often much easier to write down the equations for the dynamics (force models) than to derive analytical solutions for them. Numerical integrators provide a way to compute these approximations efficiently and accurately.

## Available Integrators

Brahe provides four integration methods with different accuracy and performance characteristics:

<div class="center-table" markdown="1">
| Integrator | Order | Type | Stages |
|------------|-------|------|--------|
| **RK4** | 4 | Fixed | 4 |
| **RKF45** | 4(5) | Adaptive | 6 |
| **DP54** | 5(4) | Adaptive | 7 (6 effective) |
| **RKN1210** | 12(10) | Adaptive | 17 |
</div>

**Key Properties:**

- **Order**: Higher order methods achieve better accuracy for a given step size
- **Type**: Fixed-step uses constant time step; Adaptive automatically adjusts step size based on error estimates
- **Stages**: Number of function evaluations per step (more stages = higher computational cost)

## Common Interfaces

All integrators implement a consistent interface, making it easy to switch between methods.

### Core Types

**`IntegratorConfig`**: Configuration controlling integration behavior

- `abs_tol`, `rel_tol`: Error tolerances (adaptive mode)
- `min_step`, `max_step`: Step size bounds
- `step_safety_factor`: Conservative factor for step size adjustment (default 0.9)

**`AdaptiveStepResult`**: Result from adaptive integration step

- `state`: New state vector after integration
- `dt_used`: Actual time step taken (may differ from requested)
- `error_estimate`: Estimated error in the step
- `dt_next`: Recommended step size for next integration

### Integration Methods

All integrators provide these methods:

**`step(t, state, dt)`**: Advance state by one time step

- For fixed-step integrators: Returns new state
- For adaptive integrators: Returns `AdaptiveStepResult`

**`step_with_varmat(t, state, phi, dt)`**: Advance state and state transition matrix

- Propagates both state and variational equations
- Requires a Jacobian provider
- Essential for orbit determination and uncertainty propagation

## Comparing Integrator Accuracy

The plot below shows position error vs. time for different integrators propagating a highly elliptical orbit (HEO) compared to analytical Keplerian propagation. All adaptive integrators use the same tolerances (abs_tol=1e-10, rel_tol=1e-9). In the figure, we can see that after one orbit, RKN1210 achieves sub-millimeter accuracy, while DP54 and RKF45 reach meter-level accuracy. The fixed-step RK4 with a 60s step has the most error, reaching about 1000m after one orbit.

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/integrator_accuracy_comparison_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/integrator_accuracy_comparison_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="integrator_accuracy_comparison.py"
    --8<-- "./plots/integrator_accuracy_comparison.py:12"
    ```

## Basic Usage Patterns

### Fixed-Step Integration

To use a fixed-step integrator like RK4, you create an instance with the desired step size and call `step` in a loop:

=== "Python"

    ``` python
    --8<-- "./examples/integrators/pattern_fixed_step.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/integrators/pattern_fixed_step.rs:7"
    ```

### Adaptive Integration

To use an adaptive-step integrator like RKF45, you create it with an `IntegratorConfig` specifying tolerances, then call `step`. The adaptive integrator returns an [`AdaptiveStepResult`](../../library_api/integrators/config.md) containing the new state and recommended next step size.


=== "Python"

    ``` python
    --8<-- "./examples/integrators/pattern_adaptive_step.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/integrators/pattern_adaptive_step.rs:7"
    ```

To take multiple steps until a final time, you can use a loop that updates the time and state based on the `dt_used` and `dt_next` values from the result.

=== "Python"

    ``` python
    --8<-- "./examples/integrators/pattern_multi_step.py:10"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/integrators/pattern_multi_step.rs:8"
    ```

## State Transition Matrix Propagation

For orbit determination and covariance propagation, you often need to propagate the state transition matrix (STM) alongside the state. The STM $\Phi(t, t_0)$ maps perturbations in initial state to perturbations in final state:

$$\delta\mathbf{x}(t) = \Phi(t, t_0) \cdot \delta\mathbf{x}(t_0)$$

State transition matrices are needed for a few key aspects of astrodynamics including:

- **Covariance propagation**: $P(t) = \Phi(t, t_0) P(t_0) \Phi(t, t_0)^T$
- **Sensitivity analysis**: How errors in initial conditions affect trajectory
- **Orbit determination**: Computing measurement sensitivities

They can be propagated by integrating the variational equations alongside the state, which requires computing the Jacobian of the dynamics. Brahe's integrators support this via the `step_with_varmat` method. You can learn more about defining Jacobians in the [Jacobian Computation](../mathematics/jacobian.md) guide.

=== "Python"

    ``` python
    --8<-- "./examples/integrators/pattern_stm.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/integrators/pattern_stm.rs:7"
    ```

## Module Contents

- **[Fixed-Step Integrators](fixed_step.md)** - RK4 and fixed-step integration
- **[Adaptive-Step Integrators](adaptive_step.md)** - RKF45, DP54, and RKN1210
- **[Variational Equations](variational_equations.md)** - State Transition Matrix propagation and theory
- **[Generic Functions](generic_functions.md)** - Test functions and integrator validation
- **[Configuration Guide](configuration.md)** - Choosing tolerances and tuning parameters

## See Also

- **[Comparing Integrator Performance](../../examples/using_different_integrators.md)** - Complete example comparing all integrators on a 7-day orbit propagation
- **[Integrators API Reference](../../library_api/integrators/index.md)** - Complete API documentation
- **[Jacobian Computation](../mathematics/jacobian.md)** - Required for variational equations
- **[Keplerian Propagation](../orbit_propagation/keplerian_propagation.md)** - Analytical propagation alternative
