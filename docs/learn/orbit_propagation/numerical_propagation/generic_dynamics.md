# General Dynamics Propagation

The `NumericalPropagator` provides a general-purpose numerical integrator for arbitrary ordinary differential equations (ODEs). Unlike `NumericalOrbitPropagator` which has built-in orbital force models, the generic propagator accepts user-defined dynamics functions for any dynamical system.

For API details, see the [NumericalPropagator API Reference](../../../library_api/propagators/numerical_propagator.md).

## When to Use General Dynamics

Use `NumericalPropagator` when:

- Propagating non-orbital systems (simple harmonic oscillators, population models, etc.)
- Implementing custom force models not available in `NumericalOrbitPropagator`
- Integrating coupled systems (orbit + attitude, multiple bodies, etc.)
- Prototyping custom dynamics before committing to the orbital framework

For orbital mechanics with extended state (mass, battery, temperature tracking), prefer `NumericalOrbitPropagator` with `additional_dynamics`. See [Extending Spacecraft State](extending_state.md).

## Full Example

The following example demonstrates propagating a simple harmonic oscillator (SHO), a canonical test case for numerical integrators:

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/generic_dynamics.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/generic_dynamics.rs:4"
    ```

### SHO Visualization

The following plot shows the position and velocity of the SHO over 3 periods, comparing numerical and analytical solutions:

<div class="plotly-embed">
  <iframe class="only-light" src="../../../figures/generic_dynamics_sho_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../../figures/generic_dynamics_sho_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="generic_dynamics_sho.py"
    --8<-- "./plots/learn/numerical_propagation/generic_dynamics_sho.py:12"
    ```

## Architecture Overview

### NumericalPropagator vs NumericalOrbitPropagator

<div class="center-table" markdown="1">
| Feature | NumericalOrbitPropagator | NumericalPropagator |
|---------|-------------------------|---------------------|
| Orbital dynamics | Built-in via ForceModelConfig | Must implement in dynamics_fn |
| State dimension | 6+ (orbital + extended) | Any dimension |
| Extended state | Via `additional_dynamics` | Include in dynamics_fn |
| Control | Via `control_input` | Via `control_input` |
| Trajectory type | `(D)OrbitTrajectory` with interpolation | `(D)Trajectory` |
| Use case | Orbital mechanics | Any ODE system |
</div>

## Dynamics Function

The dynamics function defines the system's equations of motion. It receives the current time (seconds from epoch), state vector, and optional parameters, returning the state derivative.

### Function Signature

=== "Python"

    ```python
    def dynamics(t: float, state: np.ndarray, params: np.ndarray | None) -> np.ndarray:
        """
        Compute state derivative for given time and state.

        Args:
            t: Time in seconds from reference epoch
            state: Current state vector (N-dimensional)
            params: Optional parameter vector

        Returns:
            State derivative vector (same dimension as state)
        """
        dstate = np.zeros(len(state))
        # Compute derivatives based on your equations of motion
        # ...
        return dstate
    ```

=== "Rust"

    ```rust
    let dynamics_fn: bh::DStateDynamics = Box::new(
        |t: f64, state: &na::DVector<f64>, params: Option<&na::DVector<f64>>| -> na::DVector<f64> {
            // t: Time in seconds from reference epoch
            // state: Current state vector (N-dimensional)
            // params: Optional parameter vector

            let mut dstate = na::DVector::zeros(state.len());
            // Compute derivatives based on your equations of motion
            // ...
            dstate
        }
    );
    ```

### Mathematical Form

For a general system, the dynamics function computes:

$$\dot{\mathbf{x}} = f(t, \mathbf{x}, \mathbf{p})$$

where $\mathbf{x}$ is the state vector, $t$ is time, and $\mathbf{p}$ is the parameter vector.

For orbital mechanics, the standard 6-element state is:

$$\mathbf{x} = [x, y, z, v_x, v_y, v_z]^T$$

With derivative:

$$\dot{\mathbf{x}} = [v_x, v_y, v_z, a_x, a_y, a_z]^T$$

## Parameter Handling

Parameters allow passing constants to the dynamics function without hardcoding them:

=== "Python"

    ```python
    # Define parameters
    params = np.array([omega**2, damping_coeff, mass])

    # Access in dynamics function
    def dynamics(t, state, params):
        omega_sq = params[0]
        damping = params[1]
        mass = params[2]
        # Use parameters in computation
        ...

    # Create propagator with parameters
    prop = bh.NumericalPropagator(
        epoch, initial_state, dynamics,
        bh.NumericalPropagationConfig.default(),
        params  # Pass parameters here
    )
    ```

=== "Rust"

    ```rust
    // Define parameters
    let params = na::DVector::from_vec(vec![omega * omega, damping_coeff, mass]);

    // Access in dynamics function
    let dynamics_fn: bh::DStateDynamics = Box::new(
        |_t, state, params| {
            let p = params.unwrap();
            let omega_sq = p[0];
            let damping = p[1];
            let mass = p[2];
            // Use parameters in computation
            ...
        }
    );

    // Create propagator with parameters
    let prop = bh::DNumericalPropagator::new(
        epoch, initial_state, dynamics_fn,
        bh::NumericalPropagationConfig::default(),
        Some(params),  // Pass parameters here
        None, None
    ).unwrap();
    ```

## Control Inputs

`NumericalPropagator` supports a separate `control_input` function that adds control contributions to the state derivative at each integration step. This separates the natural dynamics from control logic, making it easier to enable/disable control or swap control strategies.

The following example shows a damped harmonic oscillator where damping is implemented via `control_input`:

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/generic_dynamics_control.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/generic_dynamics_control.rs:4"
    ```

## Event Detection

The generic propagator supports the same event detection system as `NumericalOrbitPropagator`. Events can detect when computed quantities cross threshold values:

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/generic_dynamics_events.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/generic_dynamics_events.rs:4"
    ```

## Extended State Vectors

The generic propagator supports arbitrary state dimensions. The following pseudocode illustrates common extensions:

!!! note "Illustrative Pseudocode"
    The examples below are simplified pseudocode to illustrate the concepts. For complete, runnable examples of extended state propagation, see [Extending Spacecraft State](extending_state.md).

### Attitude Dynamics

Include quaternion and angular velocity for 6-DOF simulation (13-element state):

=== "Python"

    ```python
    def six_dof_dynamics(t, state, params):
        # State: [pos(3), vel(3), quat(4), omega(3)] = 13 elements
        pos = state[:3]
        vel = state[3:6]
        quat = state[6:10]   # [q0, q1, q2, q3]
        omega = state[10:13]  # Angular velocity [rad/s]

        # Translational dynamics (two-body gravity)
        r = np.linalg.norm(pos)
        acc = -bh.GM_EARTH * pos / r**3

        # Attitude kinematics (quaternion derivative)
        omega_quat = np.array([0, omega[0], omega[1], omega[2]])
        q_dot = 0.5 * quaternion_multiply(quat, omega_quat)

        # Angular dynamics (Euler's equations)
        I = np.diag(params[:3])  # Inertia tensor diagonal [kg*m^2]
        torque = np.zeros(3)     # External torques [N*m]
        omega_dot = np.linalg.inv(I) @ (torque - np.cross(omega, I @ omega))

        return np.concatenate([vel, acc, q_dot, omega_dot])
    ```

=== "Rust"

    ```rust
    let six_dof_dynamics: bh::DStateDynamics = Box::new(
        |_t, state, params| {
            // State: [pos(3), vel(3), quat(4), omega(3)] = 13 elements
            let pos = na::Vector3::new(state[0], state[1], state[2]);
            let vel = na::Vector3::new(state[3], state[4], state[5]);
            let quat = na::Vector4::new(state[6], state[7], state[8], state[9]);
            let omega = na::Vector3::new(state[10], state[11], state[12]);

            // Translational dynamics (two-body gravity)
            let r = pos.norm();
            let acc = -bh::GM_EARTH * pos / r.powi(3);

            // Attitude kinematics (quaternion derivative)
            let omega_matrix = na::Matrix4::new(
                0.0, -omega[0], -omega[1], -omega[2],
                omega[0], 0.0, omega[2], -omega[1],
                omega[1], -omega[2], 0.0, omega[0],
                omega[2], omega[1], -omega[0], 0.0,
            );
            let q_dot = 0.5 * omega_matrix * quat;

            // Angular dynamics (Euler's equations)
            let p = params.unwrap();
            let inertia = na::Matrix3::from_diagonal(&na::Vector3::new(p[0], p[1], p[2]));
            let torque = na::Vector3::zeros();
            let omega_dot = inertia.try_inverse().unwrap()
                * (-omega.cross(&(inertia * omega)) + torque);

            // Assemble derivative vector
            let mut dx = na::DVector::zeros(13);
            dx.fixed_rows_mut::<3>(0).copy_from(&vel);
            dx.fixed_rows_mut::<3>(3).copy_from(&acc);
            dx[6] = q_dot[0]; dx[7] = q_dot[1]; dx[8] = q_dot[2]; dx[9] = q_dot[3];
            dx[10] = omega_dot[0]; dx[11] = omega_dot[1]; dx[12] = omega_dot[2];
            dx
        }
    );
    ```

### Relative Motion (Hill-Clohessy-Wiltshire)

Propagate relative position/velocity for formation flying in the Hill frame:

=== "Python"

    ```python
    def hill_clohessy_wiltshire(t, state, params):
        # State: [x, y, z, vx, vy, vz] in Hill frame (RTN)
        x, y, z, vx, vy, vz = state
        n = params[0]  # Mean motion of reference orbit [rad/s]

        # HCW equations (linearized relative motion)
        ax = 3*n**2*x + 2*n*vy
        ay = -2*n*vx
        az = -n**2*z

        return np.array([vx, vy, vz, ax, ay, az])
    ```

=== "Rust"

    ```rust
    let hcw_dynamics: bh::DStateDynamics = Box::new(
        |_t, state, params| {
            // State: [x, y, z, vx, vy, vz] in Hill frame (RTN)
            let (x, y, z) = (state[0], state[1], state[2]);
            let (vx, vy, vz) = (state[3], state[4], state[5]);
            let n = params.unwrap()[0];  // Mean motion [rad/s]

            // HCW equations (linearized relative motion)
            let ax = 3.0 * n * n * x + 2.0 * n * vy;
            let ay = -2.0 * n * vx;
            let az = -n * n * z;

            na::DVector::from_vec(vec![vx, vy, vz, ax, ay, az])
        }
    );
    ```

## Quick Reference

### NumericalPropagator Constructor Parameters

<div class="center-table" markdown="1">
| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `epoch` | Epoch | Initial epoch | Yes |
| `initial_state` | DVector / ndarray | Initial state vector (N-dimensional) | Yes |
| `dynamics_fn` | Closure / Callable | State derivative function | Yes |
| `config` | NumericalPropagationConfig | Integrator settings | Yes |
| `params` | DVector / ndarray | Optional parameter vector | No |
| `control_input` | Closure / Callable | Optional control input function | No |
| `initial_covariance` | DMatrix / ndarray | Optional initial covariance (enables STM) | No |
</div>

## Performance Considerations

Custom dynamics functions are called at every integration step, so efficiency matters:

1. **Minimize function calls**: Cache expensive computations
2. **Avoid allocations**: Reuse arrays where possible
3. **Use NumPy vectorization**: Avoid Python loops for numerical operations
4. **Profile your dynamics**: The dynamics function dominates runtime

For Rust, ensure the dynamics closure captures minimal state and avoids unnecessary cloning.

---

## See Also

- [Numerical Propagation Overview](index.md) - Architecture and concepts
- [Extending Spacecraft State](extending_state.md) - Extended state for orbital propagation
- [Maneuvers](maneuvers.md) - Control inputs for thrust
- [Event Detection](event_detection.md) - Detecting conditions
- [NumericalPropagator API Reference](../../../library_api/propagators/numerical_propagator.md)
