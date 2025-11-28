# Jacobian Computation

Jacobian matrices are fundamental to many advanced astrodynamics computations. This guide explains how to compute and use Jacobians in Brahe for both analytical and numerical approaches.

## Understanding Jacobians

A Jacobian matrix describes how a function's outputs change with respect to changes in its inputs. For a vector function $\mathbf{f}: \mathbb{R}^n \rightarrow \mathbb{R}^n$ that maps state $\mathbf{x}$ to derivative $\dot{\mathbf{x}}$:

$$\dot{\mathbf{x}} = \mathbf{f}(t, \mathbf{x})$$

The Jacobian matrix $\mathbf{J}$ is:

$$J_{ij} = \frac{\partial f_i}{\partial x_j}$$

$\mathbf{J}$ is a real-valued $n \times n$ matrix ($J \in \mathbb{R}^{n \times n}$). In astrodynamics, this describes how the rate of change of each state component depends on all other state components.

In astrodynamics, Jacobians are crucial for:

- **State Transition Matrices**: Describing how small changes in initial conditions affect future states
- **Orbit Determination**: Propagating covariance matrices and computing measurement sensitivities
- **Trajectory Optimization**: Computing gradients for optimization algorithms
- **Uncertainty Propagation**: Tracking how uncertainties (covariances) evolve over time

### Analytical vs. Numerical Jacobians

Brahe supports both analytical and numerical Jacobian computation. Analytical Jacobians, represented by the `AnalyticJacobian` class require you to provide closed-form derivative expressions, while numerical Jacobians, provided by `NumericalJacobian` use finite difference methods to approximate derivatives automatically when given only the dynamics function.

## Analytical Jacobians

When you know the closed-form derivatives $\frac{\partial f_i}{\partial x_j}$, analytical Jacobians provide the most accurate and efficient computation.

### Simple Harmonic Oscillator Example

For a simple example, let's consider a 2D harmonic oscillator with state vector $\mathbf{x} = \begin{bmatrix} x \\ v \end{bmatrix}$ where $x$ is position and $v$ is velocity. The dynamics are:

$$\begin{bmatrix}
\dot{x} \\
\dot{v}
\end{bmatrix} = \begin{bmatrix}
v \\
-x
\end{bmatrix}$$

The analytical Jacobian is:

$$\mathbf{J} = \begin{bmatrix}
0 & 1 \\
-1 & 0
\end{bmatrix}$$

We can implement this analytical Jacobian in Brahe as follows:

=== "Python"

    ``` python
    --8<-- "./examples/mathematics/analytical_jacobian.py:10"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/mathematics/analytical_jacobian.rs:4"
    ```

### When to Use Analytical Jacobians

- Derivatives are simple to compute (e.g., linear systems, Keplerian dynamics)
- Maximum accuracy is required (no finite difference errors)
- Jacobian will be evaluated many times (performance critical)
- Working with well-studied systems (two-body problem, etc.)

## Numerical Jacobians

Numerical Jacobians use finite differences to approximate derivatives automatically. This is essential when analytical derivatives are complex or unknown. Brahe supports three difference methods: forward, central, and backward differences.

#### Forward Difference

The forward difference method approximates the derivative by perturbing the input state positively along each dimension $e_j$ and measuring the change in output, as follows:

$$
J_{ij} \approx \frac{f_i(\mathbf{x} + h \cdot \mathbf{e}_j) - f_i(\mathbf{x})}{h}
$$

Forward differences have first-order accuracy with an error on the order of $O(h)$, where $h$ is the perturbation size. This method requires $n + 1$ function evaluations for an $n$-dimensional state vector.

#### Central Difference

The central difference method improves accuracy by perturbing the input state both positively and negatively along each dimension $e_j$:

$$
J_{ij} \approx \frac{f_i(\mathbf{x} + h \cdot \mathbf{e}_j) - f_i(\mathbf{x} - h \cdot \mathbf{e}_j)}{2h}
$$

Central differences have second-order accuracy with an error on the order of $O(h^2)$. This method requires $2n$ function evaluations for an $n$-dimensional state vector.

!!! tip "Recommendation"
    Use central differences unless computational cost is prohibitive. The ~2x increase in function evaluations is usually worth the improved accuracy.

#### Backward Difference

Finally, the backward difference method approximates the derivative by perturbing the input state negatively along each dimension:

$$
J_{ij} \approx \frac{f_i(\mathbf{x}) - f_i(\mathbf{x} - h \cdot \mathbf{e}_j)}{h}
$$

Similar to forward differences, backward differences have first-order accuracy with an error on the order of $O(h)$ and require $n + 1$ function evaluations. They are less commonly used but implemented for completeness.

### Basic Numerical Jacobian

We can implement the same 2D harmonic oscillator example using a numerical Jacobian with central differences:

=== "Python"

    ``` python
    --8<-- "./examples/mathematics/numerical_jacobian.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/mathematics/numerical_jacobian.rs:4"
    ```


## Comparing Methods

=== "Python"

    ``` python
    --8<-- "./examples/mathematics/jacobian_methods.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/mathematics/jacobian_methods.rs:4"
    ```

## Perturbation Strategies

The choice of perturbation size $h$ significantly affects numerical Jacobian accuracy. Too large does not provide an accurate approximation of the local derivative; too small causes roundoff errors. Brahe provides several perturbation strategies to provide options for choosing $h$. However it is ultimately up to the user to select the best strategy for their specific application.

### Fixed Perturbation

One simple approach is to use a fixed perturbation size for all state components. This generally works well when all state components have similar magnitudes.

$$
h = \text{constant}
$$

```python
jacobian = bh.NumericalJacobian.central(dynamics) \\
    .with_fixed_offset(1e-6)
```

### Percentage Perturbation

Another approach is to use a percentage of each state component's magnitude as the perturbation size.

$$
h_j = \text{percentage} \times |x_j|
$$

```python
jacobian = bh.NumericalJacobian.central(dynamics) \\
    .with_percentage(1e-5)  # 0.001% perturbation
```

### Adaptive Perturbation

The adaptive perturbation strategy combines both absolute and relative scaling to choose an appropriate perturbation size for each state component. It multiples the component scale factor $s$ by $\sqrt(\epsilon)$ where $\espilon$ is machine epsilon for double precision ($\approx 2.22e-16$) and enforces a minimum value $h_{min}$ to avoid excessively small perturbations.

$$
h_j = s \times \sqrt(\epsilon) \times \max(|x_j|, h_{min})
$$

```python
jacobian = bh.NumericalJacobian.central(dynamics) \\
    .with_adaptive(scale_factor=1e-8, min_value=1e-6)
```

!!! tip "Recommendation"
    Adaptive perturbation is will generally best choice for most applications, as it balances accuracy and robustness across a wide range of state magnitudes, but percentage-based perturbations can also work well without much tuning.

### Comparing Strategies

=== "Python"

    ``` python
    --8<-- "./examples/mathematics/perturbation_strategies.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/mathematics/perturbation_strategies.rs:4"
    ```

## Using with Integrators

Jacobians are primarily used with numerical integrators for variational equation propagation see the [Numerical Integration](../integrators/index.md) guide for more details.

## See Also

- **[Mathematics Module](index.md)** - Mathematics module overview
- **[Jacobian API Reference](../../library_api/mathematics/jacobian.md)** - Complete API documentation
- **[Numerical Integration](../integrators/index.md)** - Using Jacobians with integrators
