# Gabbard Diagrams

A Gabbard diagram plots orbital period versus apogee and perigee altitude, providing a unique visualization for analyzing debris clouds, satellite breakups, and orbital constellations. Each object appears as two points: one for apogee altitude and one for perigee altitude, both at the same orbital period. This creates a characteristic pattern that reveals the distribution and evolution of orbital populations.

See also: [plot_gabbard_diagram API Reference](../../library_api/plots/gabbard.md)

## Interactive Gabbard Diagram (Plotly)

The plotly backend allows you to zoom into specific regions and hover over points to see exact values.

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/plot_gabbard_plotly_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/plot_gabbard_plotly_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="gabbard_plotly.py"
    --8<-- "./plots/learn/plots/gabbard_plotly.py"
    ```

This example shows a simulated debris cloud with 50 objects having varying semi-major axes and eccentricities. The scatter pattern shows:

- **Vertical spread** indicates eccentricity variation (distance between apogee and perigee for each object)
- **Horizontal spread** shows period distribution
- Objects with higher eccentricity have larger vertical separation between their apogee and perigee points

## Static Gabbard Diagram (Matplotlib)

The matplotlib backend produces publication-quality figures for research papers and technical reports.

![Gabbard Diagram](../../figures/plot_gabbard_matplotlib.png)

??? "Plot Source"

    ``` python title="gabbard_matplotlib.py"
    --8<-- "./plots/learn/plots/gabbard_matplotlib.py"
    ```

## Applications

### Debris Cloud Analysis

Gabbard diagrams are essential for analyzing satellite breakup events:

```python
import brahe as bh
import numpy as np

# Create epoch for analysis
epoch = bh.Epoch.from_datetime(2025, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Load debris population (example with generated data)
debris_props = []
for debris_tle in debris_tle_list:
    prop = bh.SGPPropagator.from_3le(
        debris_tle.name, debris_tle.line1, debris_tle.line2, 60.0
    )
    debris_props.append(prop)

# Create Gabbard diagram
fig = bh.plot_gabbard_diagram(
    debris_props,
    epoch,
    backend="plotly"
)
```

The diagram reveals:

- **Collision signature**: Debris from collisions typically shows a tight cluster
- **Explosion signature**: Explosions create wider scatter due to delta-V imparted
- **Evolution over time**: Atmospheric drag affects lower altitude debris first

### Constellation Design

Visualize constellation architecture and verify orbital distribution:

```python
# Walker constellation parameters
n_planes = 6
n_sats_per_plane = 11
altitude = bh.R_EARTH + 550e3

constellation = []
for i in range(n_planes):
    for j in range(n_sats_per_plane):
        raan = 2 * np.pi * i / n_planes
        anom = 2 * np.pi * j / n_sats_per_plane

        oe = np.array([altitude, 0.001, np.radians(53.0), raan, 0.0, anom])
        state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
        prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0)
        constellation.append(prop)

fig = bh.plot_gabbard_diagram(constellation, epoch)
```

For a well-designed constellation with near-circular orbits, all points cluster tightly, confirming uniform altitude and minimal eccentricity.

### Orbit Maintenance Monitoring

Track constellation health by monitoring eccentricity drift:

```python
# Compare constellation at two epochs
fig_initial = bh.plot_gabbard_diagram(constellation, epoch_initial)
fig_after_6months = bh.plot_gabbard_diagram(constellation, epoch_6mo_later)

# Increased vertical spread indicates eccentricity growth
# requiring station-keeping maneuvers
```

## Understanding the Diagram

### Reading the Plot

- **X-axis**: Orbital period (minutes or hours)
- **Y-axis**: Altitude (km)
- **Each object creates TWO points**:
    - Upper point: Apogee altitude
    - Lower point: Perigee altitude

### Interpreting Patterns

**Tight vertical pairs**: Low eccentricity (near-circular orbits)

**Wide vertical separation**: High eccentricity (elliptical orbits)

**Diagonal lines**: Natural grouping by semi-major axis

**Clustered points**: Objects with similar orbital characteristics

## Tips

- Use `backend="plotly"` to identify outliers and explore specific objects interactively
- For debris analysis, color-code by time since breakup to see evolution
- Compare pre- and post-event diagrams to quantify breakup energy
- Add reference lines for altitude constraints (e.g., ISS orbit, debris-heavy regions)

## See Also

- [plot_gabbard_diagram API Reference](../../library_api/plots/gabbard.md)
- [Keplerian Elements](../orbits/keplerian_elements.md) - Understanding orbital parameters
- [Propagators](../../library_api/propagators/index.md) - Creating propagators from TLEs
