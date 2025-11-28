# Gravity Models

Earth's gravitational field is the dominant force acting on satellites and space debris. While a simple point-mass model provides a useful first approximation, the real Earth's non-spherical mass distribution creates additional gravitational effects that must be modeled for accurate orbit prediction.

## Point-Mass Gravity

The simplest model treats Earth (or any celestial body) as a point mass with all mass concentrated at its center. The gravitational acceleration is:

$$
\mathbf{a} = -\frac{GM}{r^3} \mathbf{r}
$$

where:

- $GM$ is the gravitational parameter (m³/s²)
- $\mathbf{r}$ is the position vector from the central body's center (m)
- $r = |\mathbf{r}|$ is the distance

This model for gravity is computationally efficient and works well for modeling the effect of third-body perturbations from other planets and moons. This is discussed further in the [Third-Body Perturbations](third_body.md) section.

## Spherical Harmonic Expansion

Newton's law is excellent since it allows us to analytically solve the two-body problem. However, for Earth-orbiting satellites, the point-mass assumption is insufficient due to the planet's non-uniform shape and mass distribution. The Earth's equatorial bulge, polar flattening, and irregular mass distribution cause the gravitational attraction to vary with location. These variations are modeled using spherical harmonics - a mathematical expansion in terms of Legendre polynomials.

### Geopotential

The gravitational potential at a point outside Earth can be expressed as:

$$
V(r, \phi, \lambda) = \frac{GM}{r} \sum_{n=0}^{\infty} \sum_{m=0}^{n} \left(\frac{R_E}{r}\right)^n \bar{P}_{nm}(\sin\phi) \left(\bar{C}_{nm}\cos(m\lambda) + \bar{S}_{nm}\sin(m\lambda)\right)
$$

where:

- $r, \phi, \lambda$ are spherical coordinates (radius, latitude, longitude)
- $R_E$ is Earth's equatorial radius
- $\bar{P}_{nm}$ are normalized associated Legendre polynomials
- $\bar{C}_{nm}, \bar{S}_{nm}$ are normalized geopotential coefficients
- $n$ is the degree, $m$ is the order

The acceleration is computed as the gradient of this potential, yielding:

$$
\mathbf{a} = -\nabla \frac{GM}{r} \sum_{n=0}^{\infty} \sum_{m=0}^{n} \left(\frac{R_E}{r}\right)^n \bar{P}_{nm}(\sin\phi) \left(\bar{C}_{nm}\cos(m\lambda) + \bar{S}_{nm}\sin(m\lambda)\right)
$$

### Dominant Terms

The most significant non-spherical terms are:

- $\mathbf{J}_2$ (the $C_{2,0}\right$ harmonic) models Earth's oblateness and is ~1000× larger than any other term. It causes orbital precession, that is regression of the ascending node and rotation of the argument of perigee, which make sun-synchronous orbits possible.

- $\mathbf{J}_{2,2}$ (the $C_{2,2}, S_{2,2}\right$ harmonics) model Earth's ellipticity (difference between equatorial radii). Creates tesseral perturbations.

- **Higher-order terms**: Become important for precise orbit determination and long-term propagation, especially for low Earth orbit satellites.

### Gravity Models

Brahe includes several standard geopotential models with different degrees and orders of expansion:

- **EGM2008**: Earth Gravitational Model 2008, high-fidelity model to degree/order 360
- **GGM05S**: GRACE Gravity Model, degree/order 180
- **JGM3**: Joint Gravity Model 3, degree/order 70

Higher degree/order models provide more accuracy but require more computation. For most applications:

- **Low Earth Orbit**: Degree/order 10-20 sufficient for short-term propagation
- **Medium/Geostationary Orbit**: Degree/order 4-8 usually adequate
- **High-precision applications**: Degree/order 50+ may be needed

Additional gravity models (`.gfc` files) can be downloaded from the [International Centre for Global Earth Models (ICGEM)](https://icgem.gfz-potsdam.de/tom_longtime) repository and used with Brahe.

## Computational Considerations

Spherical harmonic evaluation involves recursive computation of Legendre polynomials and requires rotation between Earth-fixed and inertial frames. The computational cost scales as O(n²) where n is the maximum degree.

For real-time applications or long propagations with many time steps, limiting the degree and order to only what's necessary for the required accuracy is important for performance.

## Usage Examples

### Point-Mass Gravity

The point-mass gravity model can be used for any celestial body by providing its gravitational parameter and position.

=== "Python"

    ```python
    --8<-- "./examples/orbit_dynamics/point_mass_gravity.py:8"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/orbit_dynamics/point_mass_gravity.rs:4"
    ```

### Spherical Harmonics

For high-fidelity Earth gravity modeling, use the spherical harmonic expansion with an appropriate geopotential model.

=== "Python"

    ```python
    --8<-- "./examples/orbit_dynamics/spherical_harmonics.py:8"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/orbit_dynamics/spherical_harmonics.rs:4"
    ```

## See Also

- [Library API Reference: Gravity](../../library_api/orbit_dynamics/gravity.md)
- [Orbital Dynamics Overview](index.md)
- [Constants: Physical Parameters](../constants.md#physical-constants)

## References

Montenbruck, O., & Gill, E. (2000). *Satellite Orbits: Models, Methods, and Applications*. Springer. Section 3.2: The Geopotential.
