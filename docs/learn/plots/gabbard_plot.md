# Gabbard Diagrams

A Gabbard diagram plots orbital period versus apogee and perigee altitude, providing a unique visualization for analyzing debris clouds, satellite breakups, and orbital constellations. Each object appears as two points: one for apogee altitude and one for perigee altitude, both at the same orbital period. This creates a characteristic pattern that reveals the distribution and evolution of orbital populations.

## Interactive Gabbard Diagram (Plotly)

The plotly backend allows you to zoom into specific regions and hover over points to see exact values.

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/gabbard_plotly_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/gabbard_plotly_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="gabbard_plotly.py"
    --8<-- "./plots/learn/plots/gabbard_plotly.py"
    ```

## Static Gabbard Diagram (Matplotlib)

The matplotlib backend produces publication-quality figures for research papers and technical reports.

![Gabbard Diagram](../../figures/gabbard_matplotlib_light.svg#only-light)
![Gabbard Diagram](../../figures/gabbard_matplotlib_dark.svg#only-dark)

??? "Plot Source"

    ``` python title="gabbard_matplotlib.py"
    --8<-- "./plots/learn/plots/gabbard_matplotlib.py"
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

**Clustered points**: Objects with similar orbital characteristics

## Tips

- Use `backend="plotly"` to identify outliers and explore specific objects interactively
- For debris analysis, color-code by time since breakup to see evolution
- Compare pre- and post-event diagrams to quantify breakup energy
- Add reference lines for altitude constraints (e.g., ISS orbit, debris-heavy regions)

## See Also

- [Keplerian Elements](../orbits/index.md) - Understanding orbital parameters
- [Propagators](../../library_api/propagators/index.md) - Creating propagators from TLEs
