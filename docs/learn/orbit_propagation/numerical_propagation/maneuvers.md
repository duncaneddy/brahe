# Impulsive and Continuous Control

The numerical propagator supports both impulsive and continuous thrust maneuvers, enabling orbit transfer, station-keeping, and trajectory optimization studies.

## Impulsive Maneuvers

Impulsive maneuvers model instantaneous velocity changes ($\Delta v$). They're implemented using event callbacks that modify the state at specific conditions.

### Using Event Callbacks

Impulsive maneuvers combine event detection with state modification. For callback details, see [Event Callbacks](event_callbacks.md).

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/impulsive_maneuver.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/impulsive_maneuver.rs:4"
    ```

### Hohmann Transfer Visualization

The following plots show the altitude and velocity changes during the Hohmann transfer example above.

#### Orbit Geometry

A top-down view showing the initial circular orbit, Hohmann transfer ellipse, and final circular orbit:

<div class="plotly-embed">
  <iframe class="only-light" src="../../../figures/hohmann_transfer_orbit_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../../figures/hohmann_transfer_orbit_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="hohmann_transfer_orbit.py"
    --8<-- "./plots/learn/numerical_propagation/hohmann_transfer_orbit.py:12"
    ```

#### Altitude Profile

The spacecraft altitude increases from 400 km to 800 km through two impulsive burns:

<div class="plotly-embed">
  <iframe class="only-light" src="../../../figures/impulsive_maneuver_altitude_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../../figures/impulsive_maneuver_altitude_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="impulsive_maneuver_altitude.py"
    --8<-- "./plots/learn/numerical_propagation/impulsive_maneuver_altitude.py:12"
    ```

#### Velocity Components

The velocity components show the discrete jumps from each impulsive burn:

<div class="plotly-embed">
  <iframe class="only-light" src="../../../figures/impulsive_maneuver_velocity_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../../figures/impulsive_maneuver_velocity_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="impulsive_maneuver_velocity.py"
    --8<-- "./plots/learn/numerical_propagation/impulsive_maneuver_velocity.py:12"
    ```

### Common Impulsive Maneuvers

<div class="center-table" markdown="1">
| Maneuver | Implementation |
|----------|----------------|
| Hohmann transfer | Two burns at apoapsis/periapsis |
| Plane change | Burn perpendicular to velocity at ascending/descending node |
| Orbit raising | Prograde burn at periapsis/apoapsis |
| Circularization | Burn at target altitude |
</div>

## Continuous Thrust

Continuous thrust maneuvers apply acceleration over extended periods. They're implemented via control input functions that add acceleration at each integration step.

### Control Input Functions

The control input function is called at each integration step and returns a state derivative contribution:

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/continuous_control.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/continuous_control.rs:4"
    ```

### Control Function Signature

The control function receives the epoch, current state, and optional parameters. It returns a state derivative vector (same dimension as state):

=== "Python"

    ```python
    def control_input(epoch, state, params):
        # Create derivative vector (zeros for positions, acceleration for velocities)
        dx = np.zeros(len(state))

        # Compute acceleration
        acceleration = compute_thrust_acceleration(epoch, state, params)

        # Apply to velocity derivatives only
        dx[3:6] = acceleration

        return dx
    ```

=== "Rust"

    ```rust
    let control_fn: bh::DControlInput = Some(Box::new(
        |t: f64, state: &na::DVector<f64>, params: Option<&na::DVector<f64>>| {
            // Create derivative vector (zeros for positions, acceleration for velocities)
            let mut dx = na::DVector::zeros(state.len());

            // Compute acceleration
            let acceleration = compute_thrust_acceleration(t, state, params);

            // Apply to velocity derivatives only
            dx[3] = acceleration[0];
            dx[4] = acceleration[1];
            dx[5] = acceleration[2];

            dx
        },
    ));
    ```

The returned vector is added to the equations of motion:

$$\dot{\mathbf{x}} = f(\mathbf{x}, t) + \mathbf{u}(t, \mathbf{x})$$

where $f$ is the natural dynamics and $\mathbf{u}$ is the control input.

### Variable Thrust

The control function can implement time-varying or state-dependent thrust:

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/variable_thrust.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/variable_thrust.rs:4"
    ```

#### Thrust Profile Visualization

The following plot shows the trapezoidal thrust profile with ramp-up and ramp-down phases:

<div class="plotly-embed">
  <iframe class="only-light" src="../../../figures/variable_thrust_profile_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../../figures/variable_thrust_profile_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="variable_thrust_profile.py"
    --8<-- "./plots/learn/numerical_propagation/variable_thrust_profile.py:12"
    ```

## Fuel Consumption Tracking

Neither maneuver type automatically tracks fuel consumption. To track propellant:

1. Extend the state vector to include mass
2. Add mass derivative to control input or additional dynamics

See [Extending Spacecraft State](extending_state.md) for complete examples.

---

## See Also

- [Event Detection](event_detection.md) - Event system fundamentals
- [Event Callbacks](event_callbacks.md) - Callback function details
- [Extending Spacecraft State](extending_state.md) - Extended state vectors
- [General Dynamics Propagation](generic_dynamics.md) - Extended state vectors
- [Numerical Orbit Propagator](numerical_orbit_propagator.md) - Propagator fundamentals
