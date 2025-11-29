# Walker Constellations

Walker constellations are satellite constellations designed for optimal coverage using symmetric orbital plane distributions. Named after John Walker, who formalized the notation in 1984, these patterns are fundamental to modern satellite system design.

## Walker Notation i:T/P/F

Walker constellations use **i:T/P/F** notation where:

- **i** (Inclination) - Orbital inclination in degrees
- **T** (Total) - Total number of satellites in the constellation
- **P** (Planes) - Number of equally-spaced orbital planes
- **F** (Phasing) - Relative phase difference between adjacent planes (0 to P-1)

All satellites share identical:

- Semi-major axis (altitude)
- Eccentricity (typically circular, e ≈ 0)
- Inclination

## Mathematical Formulation

### RAAN Distribution

For **P** orbital planes, the Right Ascension of Ascending Node (RAAN) for plane **k** is:

$$\Omega_k = \Omega_0 + k \cdot \frac{\Delta\Omega_\text{spread}}{P}$$

where:

- $\Omega_0$ is the reference RAAN
- $\Delta\Omega_\text{spread}$ is 360° for Walker Delta, 180° for Walker Star
- $k$ is the plane index (0 to $P-1$)

### Mean Anomaly Distribution

For **S = T/P** satellites per plane, the mean anomaly for satellite **j** in plane **k** is:

$$M_{k,j} = M_0 + j \cdot \frac{360°}{S} + k \cdot F \cdot \frac{360°}{T}$$

where:

- $M_0$ is the reference mean anomaly
- $j$ is the satellite index within the plane (0 to $S-1$)
- $F$ is the phasing factor

### Constraints

- $T$ must be evenly divisible by $P$
- $F$ must be in the range $[0, P-1]$

## Walker Delta vs Walker Star

Brahe supports two Walker patterns:

<div class="center-table" markdown="1">
| Pattern | RAAN Spread | Plane Spacing | Coverage |
|---------|-------------|---------------|----------|
| **Walker Delta** | 360° | $\Delta\Omega = \frac{360°}{P}$ | Global |
| **Walker Star** | 180° | $\Delta\Omega = \frac{180°}{P}$ | Polar |
</div>

**Walker Delta** distributes planes around the full 360° of RAAN, providing global coverage. This is the pattern used by GPS, Galileo, and GLONASS.

**Walker Star** distributes planes across only 180° of RAAN, concentrating coverage at polar regions. This pattern is used by Iridium for its polar LEO constellation.

## Generating Walker Constellations

### Basic Walker Delta (GPS-like)

=== "Python"

    ``` python
    --8<-- "./examples/orbits/walker_basic.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbits/walker_basic.rs:7"
    ```

### Walker Star (Iridium-like)

=== "Python"

    ``` python
    --8<-- "./examples/orbits/walker_star.py:12"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbits/walker_star.rs:8"
    ```

## Using Different Propagators

The `WalkerConstellationGenerator` can create propagators using different propagation methods:

### Keplerian Propagators

For analytical two-body propagation:

```python
propagators = walker.as_keplerian_propagators(step_size=60.0)  # 60 second steps
```

### SGP4 Propagators

For TLE-based propagation with perturbations:

```python
propagators = walker.as_sgp_propagators(
    step_size=60.0,
    bstar=0.0,      # Drag coefficient
    ndt2=0.0,       # Mean motion derivative / 2
    nddt6=0.0,      # Mean motion 2nd derivative / 6
)
```

### Numerical Propagators

For high-fidelity force-model propagation:

```python
prop_config = bh.NumericalPropagationConfig.default()
force_config = bh.ForceModelConfig.earth_gravity()

propagators = walker.as_numerical_propagators(prop_config, force_config)
```

## Visualizing Constellations

### Walker Delta Visualization

The Walker Delta pattern distributes orbital planes evenly around 360° of RAAN:

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/visualizing_walker_delta_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/visualizing_walker_delta_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="visualizing_walker_delta.py"
    --8<-- "./examples/examples/visualizing_walker_delta.py:21"
    ```

### Walker Star Visualization

The Walker Star pattern concentrates planes over 180° of RAAN for polar coverage:

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/visualizing_walker_star_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/visualizing_walker_star_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="visualizing_walker_star.py"
    --8<-- "./examples/examples/visualizing_walker_star.py:21"
    ```

---

## See Also

- [Walker Constellations API Reference](../../library_api/orbits/walker.md) - Complete API documentation
- [Keplerian Elements](properties.md) - Orbital element fundamentals
- [SGP4 Propagation](../orbit_propagation/sgp_propagation.md) - TLE-based propagation
- [Numerical Propagation](../orbit_propagation/numerical_propagation/index.md) - High-fidelity propagation
- [3D Trajectory Plotting](../plots/3d_trajectory.md) - Trajectory visualization
