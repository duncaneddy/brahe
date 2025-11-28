# Low-Thrust Orbit Raising

This example demonstrates modeling a commercial electric propulsion system - the [Orbion Aurora](https://orbionspace.com/product/) Hall-effect thruster - for low-thrust orbit raising. We'll compare performance at the thruster's 100W and 300W power configurations, using extended state dynamics to track propellant mass depletion during the maneuver.

Electric propulsion offers significantly higher specific impulse $I_{sp}$ than chemical rockets, translating to dramatic reductions in propellant mass for a given delta-v. However, the trade-off is thrust level: while chemical engines produce newtons to kilonewtons, electric thrusters produce millinewtons. This example shows how to model this realistic scenario using Brahe's [`NumericalOrbitPropagator`](../learn/orbit_propagation/numerical_propagation/index.md) with extended state dynamics.

---

## The Orbion Aurora Thruster

The Aurora is a Hall-effect thruster designed for small satellites (70+ kg), with different powe configurations of 100W to 300W using xenon propellant.

``` python
--8<-- "./examples/examples/low_thrust_orbit_raising.py:preamble"
```

``` python
--8<-- "./examples/examples/low_thrust_orbit_raising.py:thruster_specs"
```

Key specifications from the [Aurora datasheet](https://orbionspace.com/product/):

<div class="center-table" markdown="1">
| Parameter | 100W | 300W |
|-----------|------|------|
| Thrust | 5.7 mN | 19.0 mN |
| Specific Impulse | 950 s | 1370 s |
| Mass Flow Rate | 0.53 mg/s | 1.3 mg/s |
</div>

The mass flow rate follows from the thrust equation:

$$\dot{m} = \frac{F}{I_{sp} \cdot g_0}$$

## Spacecraft Configuration

We model a 50 kg small satellite bus equipped with the Aurora system:

``` python
--8<-- "./examples/examples/low_thrust_orbit_raising.py:spacecraft_config"
```

The Aurora system has a dry mass of 8.3 kg and we model a maximum propellant capacity of 6.0 kg xenon.

## Initial Orbit

We start from a circular LEO orbit at 400 km altitude with ISS-like inclination:

``` python
--8<-- "./examples/examples/low_thrust_orbit_raising.py:initial_orbit"
```

## Extended State Dynamics

To track propellant mass during thrusting, we extend the state vector from 6 elements (position + velocity) to 7 elements by adding mass:

$$\mathbf{x} = [x, y, z, v_x, v_y, v_z, m]^T$$

### Dynamics Functions

The `NumericalOrbitPropagator` accepts two functions for extended state modeling:

1. **`control_input`**: Returns thrust acceleration (affects velocity derivatives)
2. **`additional_dynamics`**: Returns mass flow rate (affects mass derivative)

``` python
--8<-- "./examples/examples/low_thrust_orbit_raising.py:dynamics_functions"
```

The control input computes thrust acceleration using the current mass from the extended state, applying it in the prograde (velocity) direction. The additional dynamics function returns the negative mass flow rate to model propellant consumption.

## Propagation

We create two propagators - one for each power configuration - and propagate for 24 hours of continuous thrusting:

``` python
--8<-- "./examples/examples/low_thrust_orbit_raising.py:propagation"
```

## Results Analysis

After propagation, we extract the full trajectory including the mass state:

``` python
--8<-- "./examples/examples/low_thrust_orbit_raising.py:analysis"
```

The Tsiolkovsky rocket equation provides a theoretical check on our mass tracking:

$$\Delta v = I_{sp} \cdot g_0 \cdot \ln\left(\frac{m_0}{m_f}\right)$$

## Performance Comparison

The following plots compare the 100W and 300W configurations over 24 hours of continuous thrust:

<div class="plotly-embed tall">
  <iframe class="only-light" src="../figures/low_thrust_orbit_raising_comparison_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/low_thrust_orbit_raising_comparison_dark.html"  loading="lazy"></iframe>
</div>

``` python
--8<-- "./examples/examples/low_thrust_orbit_raising.py:visualization_comparison"
```

## Full Code Example

```python title="low_thrust_orbit_raising.py"
--8<-- "./examples/examples/low_thrust_orbit_raising.py:all"
```

---

## See Also

- [Extending Spacecraft State](../learn/orbit_propagation/numerical_propagation/extending_state.md) - Mass and battery tracking fundamentals
- [LEO to GEO Hohmann Transfer](geo_hohmann_transfer.md) - Impulsive maneuver comparison
- [Impulsive and Continuous Control](../learn/orbit_propagation/numerical_propagation/maneuvers.md) - Control implementation details
- [Numerical Orbit Propagation](../learn/orbit_propagation/numerical_propagation/index.md) - Propagator fundamentals
