# Atmospheric Drag

Atmospheric drag is one of the most significant perturbations for satellites in low Earth orbit (LEO). Even at altitudes of up to 2000 km, there are still traces of the Earth's atmosphere that create drag forces on satellites. This drag causes orbital decay, leading to a gradual decrease in altitude over time.

Drag is a non-conservative force that dissipates orbital energy. It is also highly dependent on atmospheric density, which varies with altitude, solar activity, geomagnetic conditions, and other factors. This variability makes drag one of the most challenging perturbations to model accurately and is often the largest source of uncertainty in LEO orbit prediction.

!!! note
    Currently Brahe only implements the simple Harris-Priester atmospheric density model for drag calculations. More advanced models like NRLMSISE-00, NRLMSISE 2.1, DTM-2020, etc are high-priority features for future releases.

## Physical Model

The drag acceleration on a satellite moving through the atmosphere is:

$$
\mathbf{a}_D = -\frac{1}{2} C_D \frac{A}{m} \rho v_{rel}^2 \mathbf{\hat{v}}_{rel}
$$

where:

- $\rho$ is atmospheric density (kg/m³)
- $v_{rel}$ is the satellite's speed relative to the atmosphere (m/s)
- $C_D$ is the drag coefficient (dimensionless, typically 2.0-2.5)
- $A$ is the cross-sectional area perpendicular to velocity (m²)
- $m$ is the satellite mass (kg)
- $\mathbf{\hat{v}}_{rel}$ is the unit velocity vector relative to the atmosphere

## Key Dependencies

### Atmospheric Density

Atmospheric density is the most uncertain and variable parameter in drag modeling. It depends on, altitude, solar and geomagnetic activity, time of day, geographic location, and season.

### Satellite Properties

The satellite's ballistic coefficient $B = C_D A / m$ determines drag sensitivity:

- **Large area-to-mass ratio** (high B): Lightweight satellites, large solar panels - sensitive to drag
- **Small area-to-mass ratio** (low B): Dense satellites - less affected by drag


## Density Models

There are many atmospheric density models available, ranging from simple empirical models to complex physics-based models.

### Harris-Priester Atmospheric Model

The Harris-Priester model is a simple, semi-empirical static atmospheric density model that accounts for:

- Exponential density decrease with altitude
- Day-night density variations (diurnal bulge)
- Solar activity effects through minimum/maximum density tables

The model divides the atmosphere into altitude bins and provides density values for minimum and maximum solar activity conditions. Interpolation between these values allows modeling of different solar cycle phases.

## Usage Examples

### Computing Drag Acceleration

Calculate the atmospheric drag acceleration on a satellite using the Harris-Priester density model.

=== "Python"

    ```python
    --8<-- "./examples/orbit_dynamics/atmospheric_drag.py:8"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/orbit_dynamics/atmospheric_drag.rs:4"
    ```

## See Also

- [Library API Reference: Drag](../../library_api/orbit_dynamics/drag.md)
- [Library API Reference: Atmospheric Density Models](../../library_api/orbit_dynamics/atmospheric_density_models.md)
- [Orbital Dynamics Overview](index.md)

## References

Montenbruck, O., & Gill, E. (2000). *Satellite Orbits: Models, Methods, and Applications*. Springer. Section 3.4: Atmospheric Drag.
