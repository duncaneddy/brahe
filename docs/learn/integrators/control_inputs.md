# Control Inputs

Control inputs allow you to add external forcing functions to your dynamics without modifying the core dynamics function. This separation is useful for modeling thrust, drag, or other perturbations that can be toggled on and off.

## What are Control Inputs?

In control theory, a dynamical system with control inputs is written as:

$$\dot{\mathbf{x}} = \mathbf{f}(t, \mathbf{x}) + \mathbf{u}(t, \mathbf{x})$$

where:

- $\mathbf{f}(t, \mathbf{x})$ is the nominal dynamics (e.g., gravitational acceleration)
- $\mathbf{u}(t, \mathbf{x})$ is the control input (e.g., thrust acceleration)

The control input function $\mathbf{u}$ takes the current time and state and returns a vector that is added to the state derivative. This additive structure makes it easy to:

- Switch control on/off without changing the dynamics function
- Combine different control strategies
- Test different control laws with the same dynamics


## Control Input Function Signature

The control input function must follow specific signatures depending on the language:

=== "Python"

    ```python
    def control_function(t: float, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Args:
            t: Current time
            state: Current state vector
            params: Additional parameters
            
        Returns:
            Control vector of same dimension as state
        """
        pass
    ```

=== "Rust"

    Must be a closure or function with the signature, that either uses dynamic or static sized vectors:

    ```rust
    Fn(f64, DVector<f64>, DVector<f64>) -> DVector<f64>
    ```

    or

    ```rust
    Fn(f64, SVector<f64, S>, SVector<f64, P>) -> SVector<f64, S>
    ```

The function receives:
- Current time as a scalar
- Current state vector
- Additional parameters as a vector

And returns a control vector of the same dimension as the state, which is added to the derivative computed by the dynamics function. The additional parameters can be ignored if not needed, but the signature must be maintained.

## Using Control Inputs

Control inputs are passed as a separate parameter to the integrator constructor. The dynamics function computes the nominal state derivative, and the control function computes the perturbation that is added to it.

=== "Python"

    ``` python
    --8<-- "./examples/integrators/control_inputs.py:14"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/integrators/control_inputs.rs:6"
    ```

## Applications

Control inputs are particularly useful for:

- **Orbit raising/lowering**: Frequent thrusting to get to desired orbit
- **Station keeping**: Small corrections to maintain orbit and compensate for drag
- **Redezvous and Proximity Operations**: Relative motion control between satellites
- **Spacecraft Collision Avoidance**: Maneuvering to avoid debris

## See Also

- [Adaptive Step Integration](adaptive_step.md) - Recommended for control problems
- [Variational Equations](variational_equations.md) - For control sensitivity analysis
- [Configuration Guide](configuration.md) - Tuning integrator parameters
