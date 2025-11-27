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

#### Altitude Profile

The spacecraft altitude increases from 400 km to 800 km through two impulsive burns:

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/impulsive_maneuver_altitude_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/impulsive_maneuver_altitude_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="impulsive_maneuver_altitude.py"
    --8<-- "./plots/learn/numerical_propagation/impulsive_maneuver_altitude.py:12"
    ```

#### Velocity Components

The velocity components show the discrete jumps from each impulsive burn:

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/impulsive_maneuver_velocity_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/impulsive_maneuver_velocity_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="impulsive_maneuver_velocity.py"
    --8<-- "./plots/learn/numerical_propagation/impulsive_maneuver_velocity.py:12"
    ```

### Common Impulsive Maneuvers

| Maneuver | Implementation |
|----------|----------------|
| Hohmann transfer | Two burns at apoapsis/periapsis |
| Plane change | Burn perpendicular to velocity at node |
| Orbit raising | Prograde burn at periapsis |
| Circularization | Burn at target altitude |

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

The returned vector is added to the equations of motion:

$$\dot{\mathbf{x}} = f(\mathbf{x}, t) + \mathbf{u}(t, \mathbf{x})$$

where $f$ is the natural dynamics and $\mathbf{u}$ is the control input.

### Thrust Directions

Common continuous thrust strategies:

| Strategy | Direction | Application |
|----------|-----------|-------------|
| Tangential | Along velocity vector | Orbit raising/lowering |
| Radial | Toward/away from Earth | Eccentricity control |
| Normal | Perpendicular to orbit plane | Inclination change |
| Optimal | Computed for objective | Fuel-optimal transfers |

### Variable Thrust

The control function can implement time-varying or state-dependent thrust:

```python
def variable_thrust(epoch, state, params):
    # Time since maneuver start
    t_maneuver = epoch - maneuver_start

    # Thrust magnitude profile (e.g., ramp up/down)
    if t_maneuver < ramp_time:
        magnitude = max_thrust * (t_maneuver / ramp_time)
    elif t_maneuver > burn_duration - ramp_time:
        magnitude = max_thrust * ((burn_duration - t_maneuver) / ramp_time)
    else:
        magnitude = max_thrust

    # Direction along velocity
    v = state[3:6]
    v_hat = v / np.linalg.norm(v)

    dx = np.zeros(6)
    dx[3:6] = (magnitude / mass) * v_hat
    return dx
```

## Combining Maneuver Types

Complex missions may combine impulsive and continuous maneuvers:

1. **Impulsive insertion burn** using event callback
2. **Continuous station-keeping** using control input
3. **Impulsive disposal burn** using another event

The propagator supports both mechanisms simultaneously.

## Fuel Consumption Tracking

Neither maneuver type automatically tracks fuel consumption. To track propellant:

1. Extend the state vector to include mass
2. Add mass derivative to control function
3. Use the generic `NumericalPropagator` for variable-dimension states

See [Extending Spacecraft State](extending_state.md) for complete examples.

## Maneuver Planning

Brahe's numerical propagator provides the dynamics engine for maneuver planning, but optimal maneuver computation requires additional analysis:

1. Define the objective (minimum fuel, minimum time, etc.)
2. Parameterize the maneuver (burn times, directions, magnitudes)
3. Propagate to evaluate the objective
4. Use optimization to find optimal parameters

The propagator handles step 3, while the optimization loop is typically implemented externally.

---

## See Also

- [Event Detection](event_detection.md) - Event system fundamentals
- [Event Callbacks](event_callbacks.md) - Callback function details
- [Extending Spacecraft State](extending_state.md) - Extended state vectors
- [General Dynamics Propagation](../../generic_dynamics.md) - Extended state vectors
- [Numerical Orbit Propagator](basic_propagation.md) - Propagator fundamentals
