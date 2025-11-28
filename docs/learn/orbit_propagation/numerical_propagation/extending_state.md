# Extending Spacecraft State

The `NumericalOrbitPropagator` supports extending state vectors beyond the standard 6-element orbital state, enabling modeling of additional state variables and dynamics like propellant mass, battery charge, or attiude alongside orbital dynamics. This is achieved through the `additional_dynamics` function.

## State Extension Approach

To extend the state vector with `NumericalOrbitPropagator`:

1. Define an extended state vector (e.g., 7 elements: `[pos, vel, mass]`)
2. Implement an `additional_dynamics` function that returns a full state-sized derivative vector, where the first 6 elements are zeros (orbital dynamics handled by the force model) and the remaining elements contain derivatives for the extended state
3. Optionally provide a `control_input` function for thrust accelerations
4. Create the propagator with these functions

The key advantage of using `NumericalOrbitPropagator` is that orbital dynamics (gravity, drag, SRP, etc.) are handled automatically by the force model configuration, while your `additional_dynamics` function adds derivatives for the extended state elements.

To showcase how to extend the spacecraft state, we present an example of tracking propellant mass during a thrust maneuver below.

## Mass Tracking Example

One common extension is tracking propellant mass during the mission. To model propelant mass we augment the state vector from 6 to 7 elements, by adding mass $m$ as the 7th element:

$$\mathbf{x} = [x, y, z, v_x, v_y, v_z, m]^T$$

### Mass Flow Dynamics

We model mass flow rate during thrust as:

$$\dot{m} = -\frac{F}{I_{sp} \cdot g_0}$$

where:

- $F$ is thrust force (N)
- $I_{sp}$ is specific impulse (s)
- $g_0$ is standard gravity (9.80665 m/s²)

### Implementation with NumericalOrbitPropagator

Both `additional_dynamics` and `control_input` functions return full state-sized vectors. The propagator adds these to the orbital dynamics computed from the force model.

The `additional_dynamics` function returns a state-sized vector with derivatives for extended elements:

```python
def additional_dynamics(t, state, params):
    """Return full state-sized vector with mass rate."""
    dx = np.zeros(len(state))  # Full state size
    if t < burn_duration:
        dx[6] = -mass_flow_rate  # dm/dt = -F/(Isp*g0)
    return dx
```

The `control_input` function returns a state-sized vector with acceleration in indices 3-5:

```python
def control_input(t, state, params):
    """Return full state-sized vector with thrust acceleration."""
    dx = np.zeros(len(state))
    if t < burn_duration:
        mass = state[6]  # Access mass from extended state
        vel = state[3:6]
        v_hat = vel / np.linalg.norm(vel)  # Prograde direction
        acc = (thrust_force / mass) * v_hat
        dx[3:6] = acc  # Add to velocity derivatives
    return dx
```

### Complete Example

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/mass_tracking.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/mass_tracking.rs:4"
    ```

### Orbital Elements Evolution

The following plot shows how orbital elements evolve during the thrust maneuver:

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/mass_tracking_elements_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/mass_tracking_elements_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="mass_tracking_elements.py"
    --8<-- "./plots/learn/numerical_propagation/mass_tracking_elements.py:12"
    ```

### Mass Depletion Profile

The mass decreases linearly during the thrust phase:

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/mass_tracking_mass_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/mass_tracking_mass_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="mass_tracking_mass.py"
    --8<-- "./plots/learn/numerical_propagation/mass_tracking_mass.py:12"
    ```

### Tsiolkovsky Verification

The mass ratio determines achievable $\Delta v$:

$$\Delta v = I_{sp} \cdot g_0 \cdot \ln\left(\frac{m_0}{m_f}\right)$$

This provides a useful validation check for mass tracking implementations.

## Other Common Extensions

### Battery/Power State

Track battery charge level during eclipse and sunlit periods:

```python
# State: [pos(3), vel(3), battery_charge]
def additional_dynamics_battery(t, state, params):
    """Battery dynamics only - orbital handled by force model."""
    dx = np.zeros(len(state))
    charge = state[6]

    # Simplified power dynamics
    if is_sunlit(state[:3]):
        dx[6] = solar_input - power_consumption
    else:
        dx[6] = -power_consumption

    return dx
```

### Thermal State

Track spacecraft temperature:

```python
# State: [pos(3), vel(3), temperature]
def additional_dynamics_thermal(t, state, params):
    """Thermal dynamics only - orbital handled by force model."""
    dx = np.zeros(len(state))
    temp = state[6]

    # Simplified radiation balance
    q_solar = solar_flux_absorbed(state[:3])
    q_radiated = emissivity * stefan_boltzmann * temp**4 * area
    dx[6] = (q_solar - q_radiated) / (mass * specific_heat)

    return dx
```

### Multiple Extensions

State vectors can include multiple extensions:

```python
# State: [pos(3), vel(3), mass, battery, temperature] = 9 elements
initial_state = np.array([
    *orbital_state,  # Position and velocity (6)
    1000.0,          # Mass (kg)
    100.0,           # Battery (Wh)
    293.0,           # Temperature (K)
])

def additional_dynamics_multi(t, state, params):
    """Return full state-sized vector with derivatives for extended elements."""
    dx = np.zeros(len(state))
    mass = state[6]
    charge = state[7]
    temp = state[8]

    dx[6] = -mass_flow_rate if thrusting else 0.0
    dx[7] = solar_input - power_consumption if is_sunlit(state[:3]) else -power_consumption
    dx[8] = (q_solar - q_radiated) / (mass * specific_heat)

    return dx
```

## Implementation Notes

### NumericalOrbitPropagator vs NumericalPropagator

| Feature | NumericalOrbitPropagator | NumericalPropagator |
|---------|------------------------|---------------------|
| Orbital dynamics | Built-in (force models) | Must implement manually |
| Extended state | Via `additional_dynamics` | Full dynamics function |
| Thrust | Via `control_input` | Include in dynamics |
| Trajectory | Orbital trajectory with interpolation | Generic state trajectory |

Use `NumericalOrbitPropagator` with `additional_dynamics` when you want:

- Automatic handling of gravity, drag, SRP, and other orbital perturbations
- To focus only on the extended state dynamics
- Access to orbital-specific features (STM, covariance, trajectory querying)

Use `NumericalPropagator` when you need:

- Complete control over all dynamics
- Non-orbital applications (attitude, relative motion, etc.)
- Custom force models not available in `ForceModelConfig`

### State Coupling

Extended state elements often couple with orbital dynamics:

- Mass affects thrust acceleration: $a = F/m(t)$
- Temperature affects drag coefficient: $C_D = f(T)$ for some materials
- Battery affects thruster availability

The `control_input` function can read the extended state elements to implement these couplings.

---

## See Also

- [General Dynamics Propagation](generic_dynamics.md) - Using `NumericalPropagator` for custom dynamics
- [Impulsive and Continuous Control](maneuvers.md) - Thrust implementation
- [Numerical Orbit Propagator](numerical_orbit_propagator.md) - Standard 6-DOF propagation
