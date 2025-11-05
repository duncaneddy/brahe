# Orbital Element Trajectories

Orbital element trajectory plots track how position, velocity, and orbital parameters evolve over time. Brahe provides two complementary views: Cartesian plots showing state vectors (x, y, z, vx, vy, vz) and Keplerian plots showing classical elements (a, e, i, Ω, ω, ν). These visualizations are essential for analyzing perturbations, verifying propagators, and understanding orbital dynamics.

## Cartesian State Vector Plots

Cartesian plots display position and velocity components in ECI coordinates, useful for debugging propagators and analyzing state evolution.

### Interactive Cartesian Plot (Plotly)

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/cartesian_trajectory_plotly_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/cartesian_trajectory_plotly_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="cartesian_trajectory_plotly.py"
    --8<-- "./plots/learn/plots/cartesian_trajectory_plotly.py"
    ```

### Static Cartesian Plot (Matplotlib)

<figure markdown="span">
    ![Cartesian Trajectory Plot](../../figures/plot_cartesian_trajectory_matplotlib_light.svg#only-light)
    ![Cartesian Trajectory Plot](../../figures/plot_cartesian_trajectory_matplotlib_dark.svg#only-dark)
</figure>

??? "Plot Source"

    ``` python title="cartesian_trajectory_matplotlib.py"
    --8<-- "./plots/learn/plots/cartesian_trajectory_matplotlib.py"
    ```

The 2×3 subplot layout shows:

- **Top row**: x, y, z position components (km)
- **Bottom row**: vx, vy, vz velocity components (km/s)

For circular orbits, you'll see sinusoidal patterns. Elliptical orbits show variations in velocity magnitude.

## Keplerian Orbital Element Plots

Keplerian plots display classical orbital elements, ideal for understanding long-term evolution and perturbation effects.

### Interactive Keplerian Plot (Plotly)

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/keplerian_trajectory_plotly_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/keplerian_trajectory_plotly_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="keplerian_trajectory_plotly.py"
    --8<-- "./plots/learn/plots/keplerian_trajectory_plotly.py"
    ```

### Static Keplerian Plot (Matplotlib)

<figure markdown="span">
    ![Keplerian Trajectory Plot](../../figures/plot_keplerian_trajectory_matplotlib_light.svg#only-light)
    ![Keplerian Trajectory Plot](../../figures/plot_keplerian_trajectory_matplotlib_dark.svg#only-dark)
</figure>

??? "Plot Source"

    ``` python title="keplerian_trajectory_matplotlib.py"
    --8<-- "./plots/learn/plots/keplerian_trajectory_matplotlib.py"
    ```

The 2×3 subplot layout shows:

- **Semi-major axis (a)**: Average orbital radius
- **Eccentricity (e)**: Orbit shape (0 = circular, >0 = elliptical)
- **Inclination (i)**: Orbital plane tilt
- **RAAN (Ω)**: Right ascension of ascending node
- **Argument of periapsis (ω)**: Orbit orientation in plane
- **Mean anomaly (M)**: Position along orbit

## Comparing Different Propagators

Compare different propagators to verify agreement or identify perturbation effects. These examples show how Keplerian (two-body) and SGP4 propagators diverge over time due to atmospheric drag and other perturbations.

The plots show how the two propagation methods diverge:

- **Keplerian (blue)**: Assumes pure two-body dynamics with no perturbations
- **SGP4 (red)**: Includes atmospheric drag and other perturbations

For near-circular LEO orbits, we notice there is significant variation in the argument of perigee (ω) and mean anomaly (M) over time due to numerical instability and ill-conditioning of these elements for near-circular orbits.

### Cartesian State Comparison

Comparing propagators in Cartesian space shows position and velocity component differences:

#### Interactive Cartesian Comparison (Plotly)

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/comparing_propagators_cartesian_plotly_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/comparing_propagators_cartesian_plotly_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="comparing_propagators_cartesian_plotly.py"
    --8<-- "./plots/learn/plots/comparing_propagators_cartesian_plotly.py"
    ```

#### Static Cartesian Comparison (Matplotlib)

<figure markdown="span">
    ![Comparing Propagators Cartesian Plot](../../figures/comparing_propagators_cartesian_matplotlib_light.svg#only-light)
    ![Comparing Propagators Cartesian Plot](../../figures/comparing_propagators_cartesian_matplotlib_dark.svg#only-dark)
</figure>

??? "Plot Source"

    ``` python title="comparing_propagators_cartesian_matplotlib.py"
    --8<-- "./plots/learn/plots/comparing_propagators_cartesian_matplotlib.py"
    ```

### Keplerian Element Comparison

Comparing propagators using Keplerian elements reveals how orbital parameters evolve differently:

#### Interactive Keplerian Comparison (Plotly)

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/comparing_propagators_keplerian_plotly_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/comparing_propagators_keplerian_plotly_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="comparing_propagators_keplerian_plotly.py"
    --8<-- "./plots/learn/plots/comparing_propagators_keplerian_plotly.py"
    ```

#### Static Keplerian Comparison (Matplotlib)

<figure markdown="span">
    ![Comparing Propagators Keplerian Plot](../../figures/comparing_propagators_keplerian_matplotlib_light.svg#only-light)
    ![Comparing Propagators Keplerian Plot](../../figures/comparing_propagators_keplerian_matplotlib_dark.svg#only-dark)
</figure>

??? "Plot Source"

    ``` python title="comparing_propagators_keplerian_matplotlib.py"
    --8<-- "./plots/learn/plots/comparing_propagators_keplerian_matplotlib.py"
    ```

## Unit Customization

### Cartesian Plots

```python
# Meters and m/s
fig = bh.plot_cartesian_trajectory(
    [{"trajectory": traj}],
    position_units="m",
    velocity_units="m/s"
)

# Kilometers and km/s (default)
fig = bh.plot_cartesian_trajectory(
    [{"trajectory": traj}],
    position_units="km",
    velocity_units="km/s"
)
```

### Keplerian Plots

```python
# Degrees (default)
fig = bh.plot_keplerian_trajectory(
    [{"trajectory": traj}],
    sma_units="km",
    angle_units="deg"
)

# Radians
fig = bh.plot_keplerian_trajectory(
    [{"trajectory": traj}],
    sma_units="km",
    angle_units="rad"
)
```

## See Also

- [plot_cartesian_trajectory API Reference](../../library_api/plots/orbital_trajectories.md)
- [plot_keplerian_trajectory API Reference](../../library_api/plots/orbital_trajectories.md)
- [3D Trajectories](3d_trajectory.md) - Spatial visualization
- [Orbital Anomalies](../orbits/anomalies.md) - Understanding orbital parameters
- [Propagators](../../library_api/propagators/index.md) - Orbit propagation
