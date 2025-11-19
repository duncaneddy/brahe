# Fixed-Step Integrators

Fixed-step integrators use a constant time step throughout the integration. Unlike adaptive methods, they don't automatically adjust step size based on error estimates. They are simpler to implement and have predictable computational costs, but require careful step size selection to ensure accuracy. They provide regular output at fixed intervals, making them suitable for applications needing uniform sampling.

## RK4: Classical Runge-Kutta

Brahe implements the classical 4th-order Runge-Kutta (RK4) method as it's primary fixed-step integrator. The 4th-order Runge-Kutta method (RK4) is the most popular fixed-step integrator, offering an excellent balance of accuracy and simplicity.

### Algorithm

For $\dot{\mathbf{x}} = \mathbf{f}(t, \mathbf{x})$, the RK4 method computes:

$$\begin{align}
\mathbf{k}_1 &= \mathbf{f}(t, \mathbf{x}) \\
\mathbf{k}_2 &= \mathbf{f}(t + h/2, \mathbf{x} + h\mathbf{k}_1/2) \\
\mathbf{k}_3 &= \mathbf{f}(t + h/2, \mathbf{x} + h\mathbf{k}_2/2) \\
\mathbf{k}_4 &= \mathbf{f}(t + h, \mathbf{x} + h\mathbf{k}_3)
\end{align}$$

The next state is then given by:

$$\mathbf{x}(t + h) = \mathbf{x}(t) + \frac{h}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)$$

## Choosing Step Size

The step size h must balance accuracy and computational cost. Too large causes unacceptable errors; too small wastes computation. A decent starting point is to relate h to the characteristic time scale of the dynamics. For orbital dynamics, a common heuristic is 

$$
h \approx \frac{T}{100 \text{ \textemdash } 1000}
$$

where T is the orbital period.

Since fixed-step methods lack automatic error control, it is critical to validate that the step-size choice achieves the desired level of accuracy. Common validation approaches include:

1. **Analytical solution**: Compare against closed-form solution (when available)
2. **Step Size Comparison**: Run with both $h$ and $h/2$, compare results to confirm convergence
3. **Energy/momentum conservation**: If you have a conserative system (in astrodynamics, this would be a gravitional-only system), check that total energy and angular momentum remain constant over time.
4. **Reference integrator**: Compare against adaptive integrator with tight tolerances

### Basic Integration Example

The following example demonstrates using the RK4 fixed-step integrator to  integrate a simple harmonic oscillator.

=== "Python"

    ``` python
    --8<-- "./examples/integrators/fixed_step_demo.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/integrators/fixed_step_demo.rs:4"
    ```


## See Also

- **[Adaptive-Step Integrators](adaptive_step.md)** - For automatic error control
- **[Configuration Guide](configuration.md)** - Detailed configuration options
- **[RK4 API Reference](../../library_api/integrators/rk4.md)** - Complete RK4 documentation
- **[Integrators Overview](index.md)** - Comparison of all integrators
