# Relativistic Effects

While Newtonian mechanics is sufficient for most satellite orbit calculations, general relativistic effects become measurable with modern precision orbit determination systems. These corrections are particularly important for:

- Global Navigation Satellite Systems (GPS, Galileo, GLONASS, BeiDou)
- Fundamental physics experiments in space
- Ultra-precise orbit determination (cm-level accuracy)
- Long-term orbit propagation

## Physical Basis

General relativity modifies Newton's law of gravitation by accounting for the curvature of spacetime caused by mass. Mntenbruck & Gill (2000) provide the post-Newtonian correction of the acceleration due to Earth's gravity as:

$$
\mathbf{a} = -\frac{GM}{r^2} \left( \left( 4\frac{GM}{c^2r} - \frac{v^2}{c^2} \right)\mathbf{e}_r + 4\frac{v^2}{c^2}\left(\mathbf{e}_r \cdot \mathbf{e}_v\right)\mathbf{e}_v\right)
$$

where:

- $GM$ is Earth's gravitational parameter (m³/s²)
- $c$ is the speed of light (299,792,458 m/s)
- $r$ is the satellite position magnitude (m)
- $v$ is the satellite velocity magnitude (m/s)
- $\mathbf{e}_r = \frac{\mathbf{r}}{r}$ is the radial unit vector
- $\mathbf{e}_v = \frac{\mathbf{v}}{v}$ is the velocity unit vector

## Usage Examples

### Computing Relativistic Acceleration

Calculate the general relativistic correction to a satellite's acceleration.

=== "Python"

    ```python
    --8<-- "./examples/orbit_dynamics/relativistic_acceleration.py:8"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/orbit_dynamics/relativistic_acceleration.rs:4"
    ```

## See Also

- [Library API Reference: Relativity](../../library_api/orbit_dynamics/relativity.md)
- [Orbital Dynamics Overview](index.md)

## References

Montenbruck, O., & Gill, E. (2000). *Satellite Orbits: Models, Methods, and Applications*. Springer. Section 3.7: Relativistic Effects.
