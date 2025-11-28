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

=== "Python"

    ```python
    def additional_dynamics(t, state, params):
        """Return full state-sized vector with mass rate."""
        dx = np.zeros(len(state))  # Full state size
        if burn_start <= t < burn_end:
            dx[6] = -mass_flow_rate  # dm/dt = -F/(Isp*g0)
        return dx
    ```

=== "Rust"

    ```rust
    let additional_dynamics: bh::DAdditionalDynamics = Some(Box::new(
        |t: f64, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| {
            let mut dx = na::DVector::zeros(state.len());
            if burn_start <= t && t < burn_end {
                dx[6] = -mass_flow_rate;  // dm/dt = -F/(Isp*g0)
            }
            dx
        },
    ));
    ```

The `control_input` function returns a state-sized vector with acceleration in indices 3-5:

=== "Python"

    ```python
    def control_input(t, state, params):
        """Return full state-sized vector with thrust acceleration."""
        dx = np.zeros(len(state))
        if burn_start <= t < burn_end:
            mass = state[6]  # Access mass from extended state
            vel = state[3:6]
            v_hat = vel / np.linalg.norm(vel)  # Prograde direction
            acc = (thrust_force / mass) * v_hat
            dx[3:6] = acc  # Add to velocity derivatives
        return dx
    ```

=== "Rust"

    ```rust
    let control_input: bh::DControlInput = Some(Box::new(
        |t: f64, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| {
            let mut dx = na::DVector::zeros(state.len());
            if burn_start <= t && t < burn_end {
                let mass = state[6];  // Access mass from extended state
                let vel = na::Vector3::new(state[3], state[4], state[5]);
                let v_hat = vel.normalize();  // Prograde direction
                let acc = (thrust_force / mass) * v_hat;
                dx[3] = acc[0];
                dx[4] = acc[1];
                dx[5] = acc[2];
            }
            dx
        },
    ));
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
  <iframe class="only-light" src="../../../figures/mass_tracking_elements_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../../figures/mass_tracking_elements_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="mass_tracking_elements.py"
    --8<-- "./plots/learn/numerical_propagation/mass_tracking_elements.py:12"
    ```

### Mass Depletion Profile

The mass decreases linearly during the thrust phase:

<div class="plotly-embed">
  <iframe class="only-light" src="../../../figures/mass_tracking_mass_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../../figures/mass_tracking_mass_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="mass_tracking_mass.py"
    --8<-- "./plots/learn/numerical_propagation/mass_tracking_mass.py:12"
    ```

### Tsiolkovsky Verification

The mass ratio determines achievable $\Delta v$:

$$\Delta v = I_{sp} \cdot g_0 \cdot \ln\left(\frac{m_0}{m_f}\right)$$

This provides a useful validation check for mass tracking implementations.

## Battery Tracking Example

Another common extension is tracking battery state of charge during eclipse and sunlit periods. This models solar panel charging using the conical shadow model for accurate illumination calculation.

We augment the state vector with battery energy $E_{bat}$ in Watt-hours:

$$\mathbf{x} = [x, y, z, v_x, v_y, v_z, E_{bat}]^T$$

### Power Balance Dynamics

The battery state of charge changes based on the power balance:

$$\dot{E}_{bat} = \nu \cdot P_{solar} - P_{load}$$

where:

- $\nu$ is the illumination fraction (0 = full shadow, 1 = full sunlight)
- $P_{solar}$ is the solar panel output when fully illuminated (W)
- $P_{load}$ is the spacecraft power consumption (W)

### Implementation

The `additional_dynamics` function computes the illumination at each timestep using `eclipse_conical`:

=== "Python"

    ```python
    def additional_dynamics(t, state, params):
        """Battery dynamics with eclipse-aware solar charging."""
        dx = np.zeros(len(state))
        r_eci = state[:3]

        # Get sun position at current epoch
        current_epoch = epoch + t
        r_sun = bh.sun_position(current_epoch)

        # Get illumination fraction (0 = umbra, 0-1 = penumbra, 1 = sunlit)
        illumination = bh.eclipse_conical(r_eci, r_sun)

        # Battery dynamics (Wh/s = W / 3600)
        power_in = illumination * solar_panel_power  # W
        power_out = load_power  # W
        dx[6] = (power_in - power_out) / 3600.0  # Wh/s

        return dx
    ```

=== "Rust"

    ```rust
    let additional_dynamics: DStateDynamics = Box::new(move |t, state, _params| {
        let mut dx = na::DVector::zeros(state.len());
        let r_eci = na::Vector3::new(state[0], state[1], state[2]);

        // Get sun position at current epoch
        let current_epoch = *epoch_ref + t;
        let r_sun = bh::sun_position(current_epoch);

        // Get illumination fraction (0 = umbra, 0-1 = penumbra, 1 = sunlit)
        let illumination = bh::eclipse_conical(r_eci, r_sun);

        // Battery dynamics (Wh/s = W / 3600)
        let power_in = illumination * solar_panel_power;  // W
        let power_out = load_power;  // W
        dx[6] = (power_in - power_out) / 3600.0;  // Wh/s

        dx
    });
    ```

### Complete Example

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/battery_tracking.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/battery_tracking.rs:4"
    ```

### Battery Charge and Illumination Profile

The following plot shows battery state of charge over 3 orbits, with illumination fraction and eclipse periods clearly visible:

<div class="plotly-embed">
  <iframe class="only-light" src="../../../figures/battery_tracking_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../../figures/battery_tracking_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="battery_tracking.py"
    --8<-- "./plots/learn/numerical_propagation/battery_tracking.py:12"
    ```

The battery charges during sunlit periods (illumination = 1) and discharges during eclipse (illumination = 0). The penumbra regions show gradual transitions in illumination.

## Other Common Extensions

### Attitude Dynamics

Track spacecraft attitude alongside orbital motion. This example shows quaternion attitude propagation:

=== "Python"

    ```python
    # State: [pos(3), vel(3), q0, q1, q2, q3, wx, wy, wz] = 13 elements
    # Quaternion [q0, q1, q2, q3] + angular velocity [wx, wy, wz]
    def additional_dynamics_attitude(t, state, params):
        """Attitude dynamics only - orbital handled by force model."""
        dx = np.zeros(len(state))

        # Extract quaternion and angular velocity
        q = state[6:10]  # [q0, q1, q2, q3]
        omega = state[10:13]  # [wx, wy, wz] rad/s

        # Quaternion kinematics: dq/dt = 0.5 * Omega(omega) * q
        omega_matrix = np.array([
            [0, -omega[0], -omega[1], -omega[2]],
            [omega[0], 0, omega[2], -omega[1]],
            [omega[1], -omega[2], 0, omega[0]],
            [omega[2], omega[1], -omega[0], 0]
        ])
        dq = 0.5 * omega_matrix @ q
        dx[6:10] = dq

        # Angular velocity dynamics: I * domega/dt = -omega x (I * omega) + torque
        I = np.diag([10.0, 12.0, 8.0])  # Inertia tensor (kg*m^2)
        torque = np.zeros(3)  # External torques
        domega = np.linalg.solve(I, -np.cross(omega, I @ omega) + torque)
        dx[10:13] = domega

        return dx
    ```

=== "Rust"

    ```rust
    // State: [pos(3), vel(3), q0, q1, q2, q3, wx, wy, wz] = 13 elements
    let additional_dynamics_attitude: DStateDynamics = Box::new(
        |_t: f64, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| {
            let mut dx = na::DVector::zeros(state.len());

            // Extract quaternion and angular velocity
            let q = na::Vector4::new(state[6], state[7], state[8], state[9]);
            let omega = na::Vector3::new(state[10], state[11], state[12]);

            // Quaternion kinematics: dq/dt = 0.5 * Omega(omega) * q
            let omega_matrix = na::Matrix4::new(
                0.0, -omega[0], -omega[1], -omega[2],
                omega[0], 0.0, omega[2], -omega[1],
                omega[1], -omega[2], 0.0, omega[0],
                omega[2], omega[1], -omega[0], 0.0,
            );
            let dq = 0.5 * omega_matrix * q;
            dx[6] = dq[0]; dx[7] = dq[1]; dx[8] = dq[2]; dx[9] = dq[3];

            // Angular velocity dynamics: I * domega/dt = -omega x (I * omega) + torque
            let inertia = na::Matrix3::from_diagonal(&na::Vector3::new(10.0, 12.0, 8.0));
            let torque = na::Vector3::zeros();
            let domega = inertia.try_inverse().unwrap()
                * (-omega.cross(&(inertia * omega)) + torque);
            dx[10] = domega[0]; dx[11] = domega[1]; dx[12] = domega[2];

            dx
        },
    );
    ```

### Thermal State

Track spacecraft temperature:

=== "Python"

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

=== "Rust"

    ```rust
    // State: [pos(3), vel(3), temperature]
    let additional_dynamics_thermal: bh::DAdditionalDynamics = Some(Box::new(
        |_t: f64, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| {
            let mut dx = na::DVector::zeros(state.len());
            let temp = state[6];

            // Simplified radiation balance
            let q_solar = solar_flux_absorbed(&state.fixed_rows::<3>(0).into());
            let q_radiated = emissivity * stefan_boltzmann * temp.powi(4) * area;
            dx[6] = (q_solar - q_radiated) / (mass * specific_heat);

            dx
        },
    ));
    ```

### Multiple Extensions

State vectors can include multiple extensions:

=== "Python"

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

=== "Rust"

    ```rust
    // State: [pos(3), vel(3), mass, battery, temperature] = 9 elements
    let initial_state = na::DVector::from_vec(vec![
        orbital_state[0], orbital_state[1], orbital_state[2],  // Position
        orbital_state[3], orbital_state[4], orbital_state[5],  // Velocity
        1000.0,  // Mass (kg)
        100.0,   // Battery (Wh)
        293.0,   // Temperature (K)
    ]);

    let additional_dynamics_multi: bh::DAdditionalDynamics = Some(Box::new(
        |_t: f64, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| {
            let mut dx = na::DVector::zeros(state.len());
            let mass = state[6];
            let _charge = state[7];
            let _temp = state[8];

            dx[6] = if thrusting { -mass_flow_rate } else { 0.0 };
            dx[7] = if is_sunlit(&state.fixed_rows::<3>(0).into()) {
                solar_input - power_consumption
            } else {
                -power_consumption
            };
            dx[8] = (q_solar - q_radiated) / (mass * specific_heat);

            dx
        },
    ));
    ```

## Implementation Notes

Another way to implement extended state propagation is to use `NumericalPropagator`, which requires implementing the full dynamics function including orbital and extended state dynamics. However, using `NumericalOrbitPropagator` with `additional_dynamics` is often more convenient for orbital applications, as it handles standard orbital perturbations automatically.

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

---

## See Also

- [General Dynamics Propagation](generic_dynamics.md) - Using `NumericalPropagator` for custom dynamics
- [Impulsive and Continuous Control](maneuvers.md) - Thrust implementation
- [Numerical Orbit Propagator](numerical_orbit_propagator.md) - Standard 6-DOF propagation
