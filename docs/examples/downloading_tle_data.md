# Downloading & Visualizing TLE Data For Starlink Satellites

This example demonstrates how to download Two-Line Element (TLE) data from the CelesTrak dataset using the Brahe library, and then visualize the complete Starlink satellite constellation in an interactive 3D plot.

---

## Initialize Earth Orientation Parameters

Before starting, we need to import brahe and ensure that we have Earth orientation parameters initialized. We'll use `initialize_eop()`, which provides a [CachingEOPProvider](../library_api/eop/caching_provider.md) to deliver up-to-date Earth orientation parameters.

``` python
--8<-- "./examples/examples/downloading_tle_data.py:20:22"
```

## Download Starlink TLEs

We'll use the [CelesTrak dataset](../library_api/datasets/celestrak.md) to fetch the latest TLE data for all Starlink satellites. The `get_tles_as_propagators` function downloads the data and creates SGP4 propagators in one step:

``` python
--8<-- "./examples/examples/downloading_tle_data.py:30:35"
```

## Inspect Satellite Data

Let's examine the properties of the first satellite to understand the orbital parameters:

``` python
--8<-- "./examples/examples/downloading_tle_data.py:38:43"
```

## Visualize in 3D

We'll create an interactive 3D visualization of the entire Starlink constellation using Plotly. We'll use the Natural Earth 50m texture for a realistic Earth representation:

``` python
--8<-- "./examples/examples/downloading_tle_data.py:49:58"
```

Finally, we'll add points for all satellites at the current epoch:

``` python
--8<-- "./examples/examples/downloading_tle_data.py:64:77"
```


The resulting plot shows the complete Starlink constellation orbiting Earth. The interactive visualization allows you to rotate, zoom, and pan to explore the satellite positions from different angles.

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/downloading_tle_data_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/downloading_tle_data_dark.html"  loading="lazy"></iframe>
</div>

## Full Code Example

```python title="downloading_tle_data.py"
--8<-- "./examples/examples/downloading_tle_data.py:19:79"
```

## See Also

- [CelesTrak Dataset](../learn/datasets/celestrak.md) - More details on using CelesTrak datasets
- [Two-Line Elements](../learn/orbits/two_line_elements.md) - Understanding TLE format and usage
- [SGP4 Propagator](../learn/orbit_propagation/sgp_propagation.md) - How SGP4 works for orbit propagation
- [3D Trajectory Plotting](../learn/plots/3d_trajectory.md) - Advanced options for trajectory visualization
