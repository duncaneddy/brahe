# LEO to GEO Hohmann Transfer

In this example we'll compute and visualize a Hohmann transfer from a low Earth orbit (LEO) at 400 km altitude to geostationary orbit (GEO) at 35,786 km altitude. The Hohmann transfer is the most fuel-efficient two-impulse maneuver for transferring between coplanar circular orbits, making it fundamental to spacecraft mission design.

We'll use the `NumericalOrbitPropagator` with event callbacks to apply impulsive velocity changes at perigee (departure) and apogee (arrival). The example demonstrates the complete workflow from calculating the required delta-v values to visualizing the orbit geometry and trajectory profiles.

---

## Problem Setup

Our mission is to transfer a spacecraft from a circular LEO parking orbit at 400 km altitude to geostationary orbit at 35,786 km altitude. At GEO altitude, a satellite with zero inclination orbits the Earth once per sidereal day, appearing stationary relative to the ground - ideal for communications and weather satellites.

``` python
--8<-- "./examples/examples/geo_hohmann_transfer.py:preamble"
```

``` python
--8<-- "./examples/examples/geo_hohmann_transfer.py:orbital_parameters"
```

## Hohmann Transfer Theory

### Mathematical Background

The Hohmann transfer uses an elliptical transfer orbit that is tangent to both the initial and final circular orbits. The spacecraft performs two impulsive burns:

1. **First burn** at perigee (LEO altitude): Increases velocity to enter the transfer ellipse
2. **Second burn** at apogee (GEO altitude): Circularizes the orbit at the target altitude

The transfer orbit has its perigee at the initial orbit radius $r_1$ and apogee at the final orbit radius $r_2$.

**Transfer orbit semi-major axis:**

$$a_{transfer} = \frac{r_1 + r_2}{2}$$

**Circular orbit velocity** (from the vis-viva equation for $e = 0$):

$$v_{circ} = \sqrt{\frac{\mu}{r}}$$

**Velocity at perigee and apogee of transfer ellipse** (vis-viva equation):

$$v = \sqrt{\mu \left(\frac{2}{r} - \frac{1}{a}\right)}$$

**Delta-v for each burn:**

$$\Delta v_1 = v_{perigee,transfer} - v_{LEO} = \sqrt{\mu \left(\frac{2}{r_1} - \frac{1}{a_{transfer}}\right)} - \sqrt{\frac{\mu}{r_1}}$$

$$\Delta v_2 = v_{GEO} - v_{apogee,transfer} = \sqrt{\frac{\mu}{r_2}} - \sqrt{\mu \left(\frac{2}{r_2} - \frac{1}{a_{transfer}}\right)}$$

**Transfer time** (half the orbital period of the transfer ellipse):

$$T_{transfer} = \pi \sqrt{\frac{a_{transfer}^3}{\mu}}$$

### Delta-V Calculations

``` python
--8<-- "./examples/examples/geo_hohmann_transfer.py:hohmann_calculations"
```

The total delta-v of approximately 3.85 km/s is substantial - this is why geostationary satellites require large launch vehicles or are launched into geostationary transfer orbits (GTO) where the rocket provides the first burn and the satellite provides the circularization burn.

## Implementation

### Initial State

We create an initial circular orbit at LEO altitude. The spacecraft starts at the point that will become the perigee of the transfer orbit:

``` python
--8<-- "./examples/examples/geo_hohmann_transfer.py:initial_state"
```

### Event Callbacks for Impulsive Maneuvers

One way to implement the impulsive burns is by propagating to the burn times, extracting the state, applying the delta-v, and restarting propagation with a new state and propagator. However, brahe provides a cleaner approach using event callbacks. An event callback is a function that is invoked when a specified event occurs during propagation. In this case, we use `TimeEvent` detectors to trigger the burns at the calculated transfer times. Each callback receives the event epoch and current state, and returns the modified state along with an `EventAction` indicating whether to continue propagation:

``` python
--8<-- "./examples/examples/geo_hohmann_transfer.py:event_callbacks"
```

The callbacks apply delta-v in the prograde direction (along the velocity vector) to raise the orbit. Returning `EventAction.CONTINUE` tells the propagator to proceed with the modified state.

### Single Propagator with Events

We use a single propagator with `TimeEvent` detectors to trigger the burns at the appropriate times. This is cleaner than multi-stage propagation for simple maneuver sequences:

``` python
--8<-- "./examples/examples/geo_hohmann_transfer.py:single_propagator"
```

!!! note "Why Two-Body Dynamics?"
    We use the two-body force model (`ForceModelConfig.two_body()`) for this example because it provides the idealized Keplerian motion that matches the analytical Hohmann transfer theory. In practice, perturbations from Earth's oblateness, atmospheric drag (at LEO), and third-body effects would cause deviations that require trajectory correction maneuvers.

### Sampling the Trajectory

We sample the trajectory at regular intervals to create the visualization data. The single propagator stores the complete trajectory including the state discontinuities from the burns:

``` python
--8<-- "./examples/examples/geo_hohmann_transfer.py:sample_trajectory"
```

## Visualizations

### Orbit Geometry

The following plot shows a top-down view of the transfer geometry. The spacecraft departs from the LEO parking orbit (dashed blue), follows the transfer ellipse (solid red) for half an orbit, and arrives at GEO (dashed green):

<div class="plotly-embed medium">
  <iframe class="only-light" src="../figures/geo_hohmann_transfer_geometry_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/geo_hohmann_transfer_geometry_dark.html"  loading="lazy"></iframe>
</div>

``` python
--8<-- "./examples/examples/geo_hohmann_transfer.py:orbit_geometry_plot"
```

### Altitude Profile

The altitude profile shows the spacecraft climbing from 400 km to nearly 36,000 km over the ~5.3 hour transfer:

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/geo_hohmann_transfer_altitude_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/geo_hohmann_transfer_altitude_dark.html"  loading="lazy"></iframe>
</div>

``` python
--8<-- "./examples/examples/geo_hohmann_transfer.py:altitude_profile_plot"
```

### Velocity Profile

The velocity profile reveals the characteristic behavior of elliptical orbits - the spacecraft is fastest at perigee and slowest at apogee. The impulsive burns appear as discontinuities:

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/geo_hohmann_transfer_velocity_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/geo_hohmann_transfer_velocity_dark.html"  loading="lazy"></iframe>
</div>

``` python
--8<-- "./examples/examples/geo_hohmann_transfer.py:velocity_profile_plot"
```

## Transfer Summary

| Parameter | Value |
|-----------|-------|
| Initial altitude | 400 km (LEO) |
| Final altitude | 35,786 km (GEO) |
| First burn $\Delta v_1$ | 2.40 km/s |
| Second burn $\Delta v_2$ | 1.46 km/s |
| Total $\Delta v$ | 3.85 km/s |
| Transfer time | 5.29 hours |

## Full Code Example

```python title="geo_hohmann_transfer.py"
--8<-- "./examples/examples/geo_hohmann_transfer.py:all"
```

---

## See Also

- [Numerical Orbit Propagation](../learn/orbit_propagation/numerical_propagation/index.md) - Propagator fundamentals
- [Impulsive and Continuous Control](../learn/orbit_propagation/numerical_propagation/maneuvers.md) - Maneuver implementation details
- [Event Detection](../learn/orbit_propagation/numerical_propagation/event_detection.md) - Event-based propagation control
- [Force Models](../learn/orbit_propagation/numerical_propagation/force_models.md) - Configuring force models
