# General Dynamics Propagation

The `NumericalPropagator` provides a general-purpose numerical integrator for arbitrary ordinary differential equations (ODEs). Unlike `NumericalOrbitPropagator` which has built-in orbital force models, the generic propagator accepts user-defined dynamics functions.

For API details, see the [NumericalPropagator API Reference](../library_api/propagators/numerical_propagator.md).

## When to Use General Dynamics

Use `NumericalPropagator` when:

- Propagating non-orbital systems (attitude dynamics, relative motion, etc.)
- Implementing custom force models not available in `NumericalOrbitPropagator`
- Extending the state vector beyond position/velocity (mass, attitude, etc.)
- Integrating coupled systems (orbit + attitude, multiple bodies, etc.)

For mass tracking during thrust maneuvers, see [Extending Spacecraft State](orbit_propagation/numerical_propagation/extending_state.md).

## Dynamics Function

The dynamics function defines the system's equations of motion. It receives the current time, state, and optional parameters, returning the state derivative:

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/generic_dynamics.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/generic_dynamics.rs:4"
    ```

### Function Signature

The dynamics function signature:

```python
def dynamics(t: float, state: np.ndarray, params: np.ndarray | None) -> np.ndarray:
    """
    Args:
        t: Time (seconds from reference epoch)
        state: Current state vector
        params: Optional parameter vector

    Returns:
        State derivative vector (same dimension as state)
    """
    # Compute derivatives
    dstate = np.zeros(len(state))
    # ...
    return dstate
```

For orbital mechanics, the standard 6-element state is:

$$\mathbf{x} = [x, y, z, v_x, v_y, v_z]^T$$

With derivative:

$$\dot{\mathbf{x}} = [v_x, v_y, v_z, a_x, a_y, a_z]^T$$

## Extended State Vectors

The generic propagator supports arbitrary state dimensions. Common extensions:

### Attitude Dynamics

Include quaternion and angular velocity for 6-DOF simulation:

```python
def six_dof_dynamics(t, state, params):
    # State: [pos(3), vel(3), quat(4), omega(3)] = 13 elements
    pos = state[:3]
    vel = state[3:6]
    quat = state[6:10]
    omega = state[10:13]

    # Translational dynamics
    r = np.linalg.norm(pos)
    acc = -bh.GM_EARTH * pos / r**3

    # Attitude dynamics (quaternion derivative)
    q_dot = 0.5 * quaternion_multiply(quat, [0, omega[0], omega[1], omega[2]])

    # Angular acceleration (Euler's equations)
    I = np.diag(params[:3])  # Inertia tensor diagonal
    torque = np.zeros(3)     # External torques
    omega_dot = np.linalg.inv(I) @ (torque - np.cross(omega, I @ omega))

    return np.concatenate([vel, acc, q_dot, omega_dot])
```

### Relative Motion

Propagate relative position/velocity for formation flying:

```python
def hill_clohessy_wiltshire(t, state, params):
    # State: [x, y, z, vx, vy, vz] in Hill frame (RTN)
    x, y, z, vx, vy, vz = state
    n = params[0]  # Mean motion of reference orbit

    # HCW equations (linearized relative motion)
    ax = 3*n**2*x + 2*n*vy
    ay = -2*n*vx
    az = -n**2*z

    return np.array([vx, vy, vz, ax, ay, az])
```

## Control Inputs

Like `NumericalOrbitPropagator`, the generic propagator supports control input functions:

```python
def control_input(t, state, params):
    # Return state derivative contribution from control
    dx = np.zeros(len(state))
    # Add control acceleration, torque, etc.
    return dx

prop = bh.NumericalPropagator(
    epoch, initial_state,
    dynamics_fn=dynamics,
    control_input=control_input,
    ...
)
```

## Event Detection

The generic propagator supports the same event detection system:

```python
# Detect when relative distance crosses threshold
distance_event = bh.ValueEvent(
    "Proximity",
    lambda t, state, params: np.linalg.norm(state[:3]) - threshold,
    0.0,
    bh.EventDirection.DECREASING
)

prop.add_event_detector(distance_event)
```

## Performance Considerations

Custom dynamics functions are called at every integration step, so efficiency matters:

1. **Minimize function calls**: Cache expensive computations
2. **Avoid allocations**: Reuse arrays where possible
3. **Use NumPy vectorization**: Avoid Python loops for numerical operations
4. **Profile your dynamics**: The dynamics function dominates runtime

For Rust, ensure the dynamics closure captures minimal state and avoids unnecessary cloning.

---

## See Also

- [Numerical Propagation Overview](orbit_propagation/numerical_propagation/index.md) - Architecture and concepts
- [Extending Spacecraft State](orbit_propagation/numerical_propagation/extending_state.md) - Extended state vectors
- [Maneuvers](orbit_propagation/numerical_propagation/maneuvers.md) - Control inputs for thrust
- [Event Detection](orbit_propagation/numerical_propagation/event_detection.md) - Detecting conditions
- [NumericalPropagator API Reference](../library_api/propagators/numerical_propagator.md)
