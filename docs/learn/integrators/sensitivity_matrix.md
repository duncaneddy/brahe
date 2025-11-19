# Sensitivity Matrix

The sensitivity matrix extends variational equations to include the effect of uncertain parameters on the state. While the State Transition Matrix (STM) maps initial state uncertainties to final state uncertainties, the sensitivity matrix maps parameter uncertainties to state uncertainties.

## What is the Sensitivity Matrix?

For a dynamical system that depends on both state and parameters:

$$\dot{\mathbf{x}} = \mathbf{f}(t, \mathbf{x}, \mathbf{p})$$

where $\mathbf{p}$ is a vector of "consider parameters" (parameters that affect dynamics but aren't estimated), the sensitivity matrix $\mathbf{S}$ describes how state evolves with respect to parameter changes:

$$\mathbf{S}(t) = \frac{\partial \mathbf{x}(t)}{\partial \mathbf{p}}$$

The sensitivity matrix satisfies the differential equation:

$$\dot{\mathbf{S}} = \frac{\partial \mathbf{f}(t, \mathbf{x}, \mathbf{p})}{\partial \mathbf{x}} \mathbf{S} + \frac{\partial \mathbf{f}(t, \mathbf{x}, \mathbf{p})}{\partial \mathbf{p}}$$

Sensitivity matrices are essential for accounting for parameter uncertainties in orbit determination.

## Relationship to STM

The sensitivity matrix and STM serve related but distinct purposes:

<div class="center-table" markdown="1">
| Matrix | Equation | Maps |
|--------|----------|------|
| STM $\Phi$ | $\dot{\Phi} = \mathbf{A}\Phi$ | Initial state → Final state |
| Sensitivity $\mathbf{S}$ | $\dot{\mathbf{S}} = \mathbf{A}\mathbf{S} + \mathbf{B}$ | Parameters → Final state |
</div>

## Propagating the Sensitivity Matrix

Brahe integrators provide `step_with_sensmat()` for propagating the sensitivity matrix alongside the state:

```rust
// Result: (new_state, new_sensitivity, dt_used, error, dt_next)
let (state, sens, dt_used, error, dt_next) =
    integrator.step_with_sensmat(t, state, sensitivity, &params, dt);
```

For combined STM and sensitivity propagation:

```rust
// Result: (new_state, new_stm, new_sensitivity, dt_used, error, dt_next)
let (state, phi, sens, dt_used, error, dt_next) =
    integrator.step_with_varmat_sensmat(t, state, phi, sensitivity, &params, dt);
```

## Using Sensitivity Providers

Brahe provides two approaches for computing the sensitivity matrix $\partial \mathbf{f}/\partial \mathbf{p}$---`NumericalSensitivity` and `AnalyticSensitivity` classes. The `NumericalSensitivity` provider computes sensitivities automatically by perturbing parameters, while `AnalyticSensitivity` allows you to supply analytical derivatives for better performance. When you know the analytical form of $\partial \mathbf{f}/\partial \mathbf{p}$, use `AnalyticSensitivity` for better accuracy and performance. The example files below demonstrate both numerical and analytical approaches.

=== "Python"

    ``` python
    --8<-- "./examples/integrators/sensitivity.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/integrators/sensitivity.rs:6"
    ```

### When to Use Analytical Sensitivity

Use analytical sensitivity when:

- Derivatives $\partial \mathbf{f}/\partial \mathbf{p}$ are simple to derive (e.g., drag coefficient, solar radiation pressure coefficient)
- Maximum accuracy is required (no finite difference errors)
- Sensitivity will be computed many times (performance critical)
- Working with well-understood parameter dependencies

!!! tip "Recommendation"
    For common parameters like atmospheric drag coefficient or solar radiation pressure coefficient, the analytical derivatives are often straightforward. When analytical forms are available, they provide better accuracy and performance than numerical approximations.

### Perturbation Strategies

The `NumericalSensitivity` provider uses the same perturbation strategies as `NumericalJacobian`:

- **Fixed perturbation**: Constant step size for all parameters
- **Percentage perturbation**: Scale by parameter magnitude
- **Adaptive perturbation**: Balance accuracy and robustness

See the [Jacobian Computation](../mathematics/jacobian.md#perturbation-strategies) guide for detailed information on configuring perturbation strategies.


## Integrating Sensitivity Matrices

=== "Python"

    ``` python
    --8<-- "./examples/integrators/pattern_sensitivity_matrix.py:14"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/integrators/pattern_sensitivity_matrix.rs:6"
    ```


## See Also

- [Variational Equations](variational_equations.md) - State Transition Matrix theory
- [Configuration Guide](configuration.md) - Integrator tuning
- [Jacobian Computation](../mathematics/jacobian.md) - Computing the A matrix
