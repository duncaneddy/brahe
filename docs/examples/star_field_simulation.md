# Star-Field Sensor Simulation

In this example we'll simulate a star tracker's field of view sweeping across the sky as a satellite orbits Earth. We'll propagate a sun-synchronous low Earth orbit for one orbital period, point a 30° full-angle sensor along the velocity vector (along-track), and test the Hipparcos catalog's naked-eye-bright stars against the sensor cone at each time step. The result is an animated 3D scene showing which stars enter and leave the field of view as the satellite moves.

---

## Setup

First, we import the required modules and define the sensor and orbit configuration. The sensor half-angle sets a 30° full field of view, and stars are displayed on a fixed shell at three Earth radii so they stay clear of the orbit and Earth sphere:

``` python
--8<-- "./plots/learn/star_field/star_field_simulation.py:scenario"
```

`state_koe_to_eci` converts the Keplerian elements `[a, e, i, raan, argp, mean_anomaly]` to a Cartesian ECI state, which seeds a `KeplerianPropagator`. Propagating to `epoch + period` with a 60 s step produces one frame per step for the full orbit. The sensor boresight is simply the normalized velocity vector at each step — the sensor points along-track.

## Load the Star Catalog

Next we download (or load from cache) the Hipparcos catalog and filter to naked-eye-bright stars. Each record's `unit_vector()` gives its direction in the same inertial frame as the propagated orbit, so no additional frame transformation is needed:

``` python
--8<-- "./plots/learn/star_field/star_field_simulation.py:catalog"
```

!!! note
    A star's `unit_vector()` is computed directly from its cataloged right ascension and declination — see [RA/Dec Transformations](../learn/coordinates/radec_transformations.md) for how `state_inertial_to_radec` and related functions convert between Cartesian directions and RA/Dec. Proper motion is not applied here since the catalog epoch (J1991.25) is close enough to the simulation epoch that the shift is negligible for a field-of-view demonstration.

## Visibility Test

A star is inside the field of view when the angle between the boresight and the star direction is smaller than the sensor half-angle, i.e. `dot(star_hat, boresight_hat) > cos(half_angle)`. Computing the full dot product matrix up front (stars × frames) lets us evaluate visibility for every frame in one vectorized operation:

``` python
--8<-- "./plots/learn/star_field/star_field_simulation.py:visibility"
```

## Field of View Cone

The sensor cone is rendered as a `Mesh3d` triangle fan: the apex sits at the satellite position, and a 24-segment base circle is built from two vectors orthogonal to the boresight, positioned `CONE_LENGTH` along the boresight direction:

``` python
--8<-- "./plots/learn/star_field/star_field_simulation.py:cone"
```

## Animation

The figure combines a static Earth sphere and orbit path with three traces that update every frame: the satellite marker, the FOV cone, and the currently visible stars. Marker size scales with each star's visual magnitude so brighter stars render larger. Only the three changing traces are included in each `go.Frame`, keeping the animation payload small even with nearly 100 frames.

The animation starts playing automatically; use the Play/Pause button or drag the slider to scrub through the orbit. Hover over any star marker to see its name:

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/star_field_simulation_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/star_field_simulation_dark.html"  loading="lazy"></iframe>
</div>

Across the 96 frames of this scenario, the number of visible stars ranges from 15 to 93 (median 27) as the 30° cone sweeps through regions of varying star density.

## Full Code Example

```python title="star_field_simulation.py"
--8<-- "./plots/learn/star_field/star_field_simulation.py:all"
```

!!! note "NETWORK"
    This script downloads the Hipparcos catalog on first use (cached permanently afterward at `~/.cache/brahe/star_catalog/`) and is flagged `NETWORK`, so it is skipped by default when generating documentation figures in bulk (`just make-plots`). Regenerate it directly with `just make-plot star_field_simulation`, or see [`just download-resources`](../development_guide.md) for populating the cache offline-first.

## See Also

- [Star Catalogs](../learn/datasets/star_catalogs.md)
- [RA/Dec Transformations](../learn/coordinates/radec_transformations.md)
- [KeplerianPropagator API Reference](../library_api/propagators/keplerian_propagator.md)
- [Star Catalog API Reference](../library_api/datasets/star_catalogs.md)
