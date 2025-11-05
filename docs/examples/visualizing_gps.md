# Visualizing GPS Satellite Orbits

In this example we'll show how to visualize the orbits of GPS satellites using Brahe. We'll download the latest TLE data for the GPS constellation from CelesTrak, propagate each satellite for one orbit, and create an interactive 3D plot showing their trajectories around Earth.

This example is similar to the [Downloading & Visualizing TLE Data For Starlink Satellites](downloading_tle_data.md) example, but but adds in propagation for one full orbit before visualization.

---

## Initialize Earth Orientation Parameters

Before starting, we need to import brahe and ensure that we have Earth orientation parameters initialized. We'll use `initialize_eop()`, which provides a [CachingEOPProvider](../library_api/eop/caching_provider.md) to deliver up-to-date Earth orientation parameters.

``` python
--8<-- "./examples/examples/visualizing_gps.py:19:22"
```

## Download GPS TLEs

We'll use the [CelesTrak dataset](../library_api/datasets/celestrak.md) to fetch the latest TLE data for all GPS satellites. The `get_tles_as_propagators` function downloads the data and creates SGP4 propagators in one step:

``` python
--8<-- "./examples/examples/visualizing_gps.py:30:32"
```

## Propagate orbits

Next, we'll propagate each satellite for one full orbit based on its semi-major axis:

``` python
--8<-- "./examples/examples/visualizing_gps.py:39:42"
```

The line 
``` python
--8<-- "./examples/examples/visualizing_gps.py:40:40"
```
computes the or orbital period of the satellite by converting the semi-major axis associated with the `SGP4Propagator` into an orbital period using Brahe's `orbital_period` function.

It then propagates the satellite to one full orbit past its epoch using the `propagate_to` method to ensure that the trajectory contains position data for one complete orbit.

## Visualize in 3D

We'll create an interactive 3D visualization of the entire Starlink constellation using Plotly. We'll use the Natural Earth 50m texture for a realistic Earth representation:

``` python
--8<-- "./examples/examples/visualizing_gps.py:48:65"
```

The resulting plot shows the complete Starlink constellation orbiting Earth. The interactive visualization allows you to rotate, zoom, and pan to explore the satellite positions from different angles.

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/visualizing_gps_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/visualizing_gps_dark.html"  loading="lazy"></iframe>
</div>

## Full Code Example

```python title="visualizing_gps.py"
--8<-- "./examples/examples/visualizing_gps.py:19:67"
```

## See Also

- [CelesTrak Dataset](../learn/datasets/celestrak.md) - More details on using CelesTrak datasets
- [Two-Line Elements](../learn/orbits/two_line_elements.md) - Understanding TLE format and usage
- [SGP4 Propagator](../learn/orbit_propagation/sgp_propagation.md) - How SGP4 works for orbit propagation
- [3D Trajectory Plotting](../learn/plots/3d_trajectory.md) - Advanced options for trajectory visualization
