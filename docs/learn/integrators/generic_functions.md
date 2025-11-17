# Integrating Generic Functions

While Brahe's integrators are optimized for orbital mechanics, they work on **any ordinary differential equation (ODE)**. This makes them useful for testing, validation, and other applications beyond spacecraft dynamics.

## Why Use Test Functions?

Test functions with known analytical solutions serve several purposes:

1. **Validation**: Verify integrator implementation correctness
2. **Accuracy Assessment**: Measure numerical error against ground truth
3. **Performance Comparison**: Benchmark different methods
4. **Understanding Behavior**: Explore how integrators handle different problem types

This page demonsrtrates using Brahe integrators on series of standard test functions with known solutions.

## 1. Exponential Decay

A simple ODE with an analytical solution:

$$
\frac{dx}{dt} = -kx \quad \Rightarrow \quad x(t) = x_0 e^{-kt}
$$

=== "Python"

    ``` python
    --8<-- "./examples/integrators/exponential_decay.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/integrators/exponential_decay.rs:15"
    ```

## 2. Simple Harmonic Oscillator

The archetypal second-order ODE, converted to a first-order system:

$$
\frac{d^2x}{dt^2} = -\omega^2 x \quad \Rightarrow \quad \begin{cases} \dot{x}_1 = x_2 \\ \dot{x}_2 = -\omega^2 x_1 \end{cases}
$$

**Analytical solution**:

$$x(t) = x_0 \cos(\omega t) + \frac{v_0}{\omega} \sin(\omega t)$$

=== "Python"

    ``` python
    --8<-- "./examples/integrators/harmonic_oscillator.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/integrators/harmonic_oscillator.rs:15"
    ```

## 3. Linear 2D System with Analytical STM

A coupled linear system where the STM can be computed exactly via matrix exponential. This type of system is common in control theory and linearized dynamics.

$$
\frac{d\mathbf{x}}{dt} = \mathbf{A} \mathbf{x} \quad \Rightarrow \quad \mathbf{x}(t) = e^{\mathbf{A}t} \mathbf{x}_0
$$

For example, with $\mathbf{A} = \begin{bmatrix} -0.1 & 0.2 \\ 0 & -0.3 \end{bmatrix}$:


=== "Python"

    ``` python
    --8<-- "./examples/integrators/linear_system_stm.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/integrators/linear_system_stm.rs:15"
    ```

## 4. Van der Pol Oscillator

A nonlinear oscillator with limit cycle behavior:

$$
\frac{d^2x}{dt^2} - \mu(1 - x^2)\frac{dx}{dt} + x = 0 \quad \Rightarrow \quad \begin{cases} \dot{x}_1 = x_2 \\ \dot{x}_2 = \mu(1 - x_1^2)x_2 - x_1 \end{cases}$$

=== "Python"

    ``` python
    --8<-- "./examples/integrators/van_der_pol.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/integrators/van_der_pol.rs:15"
    ```

## See Also

- [Adaptive Step Integration](adaptive_step.md) - Detailed guide to adaptive methods
- [Fixed Step Integration](fixed_step.md) - When and how to use fixed-step methods
- [Variational Equations](variational_equations.md) - STM propagation theory
- [Library API: Integrators](../../library_api/integrators/index.md) - Complete API reference
