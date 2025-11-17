# Variational Equations and State Transition Matrix

Variational equations enable propagating not just the statye dynamics, but also how small perturbations would affect the state over time. They help relate how changes in state at one time map to changes at a later time which is critical for orbit determination and control.

## What are Variational Equations?

For a dynamical system $\dot{\mathbf{x}} = \mathbf{f}(t, \mathbf{x})$, variational equations describe how small deviations from the nominal trajectory evolve. Consider two nearby initial conditions:

- Nominal: $\mathbf{x}_0$
- Perturbed: $\mathbf{x}_0 + \delta\mathbf{x}_0$

The difference in trajectories can be approximated by the **State Transition Matrix** (STM) $\Phi(t, t_0)$:

$$\delta\mathbf{x}(t) \approx \Phi(t, t_0) \cdot \delta\mathbf{x}_0$$

This relationship is exact for linear systems and accurate for nonlinear systems when $||\delta\mathbf{x}_0||$ is small.

## The State Transition Matrix

The State Transition Matrix (STM) satisfies the **matrix differential equation**:

$$\dot{\Phi}(t, t_0) = \mathbf{J}(t, \mathbf{x}(t)) \cdot \Phi(t, t_0)$$

where $\mathbf{J}$ is the **Jacobian matrix** of the dynamics:

$$\mathbf{J}_{ij} = \frac{\partial f_i}{\partial x_j}$$

The STM has a few key properites. First, the intial condition of the STM is always the indentity matrix. That is:

$\Phi(t_0, t_0) = \mathbf{I}$ (identity matrix)

For linear systems: $\Phi(t, t_0)$ is the matrix exponential $e^{\mathbf{A}(t-t_0)}$.

## STM Propagation in Brahe

Brahe integrators can propagate the state and STM simultaneously using `step_with_varmat()`. What happens under the hood is:

1. The integrator advances both the state ($\mathbf{x}$) and the STM ($\Phi$) using the same time step
2. At each stage of the Runge-Kutta method, the Jacobian is evaluated at the current state
3. The variational equations $\dot{\Phi} = \mathbf{J} \cdot \Phi$ are integrated alongside the state equations
<!-- 4. Both are subject to the same error control and step size selection -->

=== "Python"

    ``` python
    --8<-- "./examples/integrators/with_jacobian.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/integrators/with_jacobian.rs:4"
    ```

## Equivalence to Direct Perturbation

The power of the STM is that it allows predicting many perturbed trajectories efficiently. Instead of integrating each perturbed initial condition separately, we can integrate the nominal trajectory once (with STM) and map any initial perturbation through the STM.

The following example demonstrates this equivalence:

=== "Python"

    ``` python
    --8<-- "./examples/integrators/stm_vs_direct_perturbation.py:13"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/integrators/stm_vs_direct_perturbation.rs:14"
    ```

## See Also

- [Fixed Step Integration](fixed_step.md) - Constant time step methods
- [Adaptive Step Integration](adaptive_step.md) - Automatic step size control
- [Mathematics: Jacobian](../../library_api/mathematics/jacobian.md) - Jacobian provider API reference
