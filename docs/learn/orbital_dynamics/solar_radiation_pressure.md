# Solar Radiation Pressure

Solar radiation pressure (SRP) is the force exerted by photons emitted by the sun when they strike a satellite's surface. While small compared to gravitational forces, SRP can become a significant perturbations for satellites at higher altitude, particularly for those large solar panels or lightweight structures.

## Physical Principle

Photons carry momentum, and when they strike a surface, they transfer that momentum. The acceleration due to solar radiation pressure is:

$$
\mathbf{a}_{SRP} = -P_{\odot} C_R \frac{A}{m} \nu \frac{\mathbf{r}_{\odot}}{|\mathbf{r}_{\odot}|}
$$

where:

- $P_{\odot}$ is the solar radiation pressure at 1 AU (≈ 4.56 × 10⁻⁶ N/m²)
- $C_R$ is the radiation pressure coefficient (dimensionless, typically 1.0-1.5)
- $A$ is the effective cross-sectional area perpendicular to Sun (m²)
- $m$ is the satellite mass (kg)
- $\nu$ is the shadow function (0 = full shadow, 1 = full sunlight)
- $\mathbf{r}_{\odot}$ is the Sun position vector relative to satellite

The pressure varies as $1/r^2$ with distance from the Sun but is essentially constant for Earth-orbiting satellites due to the comparatively small variation in distance around the orbit compared to the Earth-Sun distance.

### Computing SRP Acceleration

Calculate the solar radiation pressure acceleration on a satellite, accounting for Earth's shadow.

=== "Python"

    ```python
    --8<-- "./examples/orbit_dynamics/solar_radiation_pressure.py:8"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/orbit_dynamics/solar_radiation_pressure.rs:4"
    ```


## Earth Eclipse (Earth Shadowing)

Satellites in Earth orbit periodically pass through Earth's shadow, where SRP is absent. The amount of light reaching the satellite is modeled using a shadow function $\nu$ that varies between 0 (full shadow) and 1 (full sunlight). This function accounts for:

- Earth's finite size (not a point)
- Sun's finite angular diameter (not a point source)
- Atmospheric refraction and absorption

Brahe provides two shadow models with different fidelity levels:

#### Conical (Penumbral) Model

The conical shadow model accounts for the finite size of both Earth and Sun, modeling the penumbra region. It defines:

- **Umbra** $\left(\nu = 0\right)$: Region of total shadow (Sun completely blocked)
- **Penumbra** $\left(0 < \nu < 1\right)$: Region of partial shadow (Sun partially blocked)
- **Sunlight** $\left(\nu = 1\right)$: No shadow

This model provides accurate illumination fractions and is implemented in `eclipse_conical()`.

#### Cylindrical Model

The cylindrical shadow model assumes Earth casts a cylindrical shadow parallel to the Sun-Earth line. This is computationally efficient and provides a binary output of $\nu \in \{0, 1\}$. It does not model the penumbra region. The model is efficient but less accurate for satellites near the shadow boundary.

This model is implemented in `eclipse_cylindrical()`.

For many applications, the penumbra region is small enough that the cylindrical model provides sufficient accuracy with improved computational performance.

### Eclipse Detection

Determine if a satellite is in Earth's shadow using either the conical or cylindrical model:

=== "Python"

    ```python
    --8<-- "./examples/orbit_dynamics/eclipse_detection.py:8"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/orbit_dynamics/eclipse_detection.rs:4"
    ```


## See Also

- [Library API Reference: Solar Radiation Pressure](../../library_api/orbit_dynamics/solar_radiation_pressure.md)
- [Third-Body Perturbations](third_body.md) - For Sun position calculation
- [Orbital Dynamics Overview](index.md)

## References

Montenbruck, O., & Gill, E. (2000). *Satellite Orbits: Models, Methods, and Applications*. Springer. Section 3.5: Solar Radiation Pressure.
