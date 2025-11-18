# Third-Body Perturbations

Third-body perturbations are gravitational effects caused by celestial bodies other than the primary (Earth). The most significant third bodies affecting Earth satellites are the Sun and Moon, but planetary perturbations can also be important for high-precision applications or long-term orbit evolution.

## Physical Principle

The third-body perturbation is not the direct gravitational attraction of the perturbing body on the satellite, but rather the *differential* acceleration - the difference between the gravitational pull on the satellite and on Earth's center.

For a satellite at position $\mathbf{r}$ and a third body at position $\mathbf{r}_b$:

$$
\mathbf{a}_{3} = GM_{b} \left(\frac{\mathbf{r}_b - \mathbf{r}}{|\mathbf{r}_b - \mathbf{r}|^3} - \frac{\mathbf{r}_b}{|\mathbf{r}_b|^3}\right)
$$

where $GM_b$ is the gravitational parameter of the third body.

## Key Third Bodies

### Sun

The Sun is the most massive third body, but its large distance reduces its effect. Solar perturbations are particularly important for:

- Geostationary satellites (resonance effects)
- High eccentricity orbits
- Long-term orbit evolution

Typical acceleration magnitude: ~10⁻⁷ m/s² for LEO, increasing with altitude.

### Moon

Despite being less massive than the Sun, the Moon's proximity makes it a significant perturber. Lunar perturbations affect:

- Medium Earth orbit satellites (especially GPS-like orbits)
- Geostationary satellites
- Frozen orbit design

The Moon's acceleration on satellites is comparable to or larger than the Sun's at most altitudes.

### Planets

Planetary perturbations (Venus, Jupiter, Mars, etc.) are generally small but can accumulate over long time scales. They become relevant for:

- Long-term orbit propagation (years to decades)
- Precise orbit determination
- Special resonance conditions

## Modeling Approaches

Brahe provides two methods for computing third-body positions and perturbations:

### Analytical Models

Simplified analytical expressions provide approximate positions of the Sun and Moon based on time. These models are computationally efficient and suitable for many applications. They also don't require external data files.

### DE440s Ephemerides

For high-precision applications, Brahe supports using JPL's DE440s ephemerides with data provided by NASA JPL's [Naviation and Ancillary Information Facility](https://naif.jpl.nasa.gov/naif/index.html) and computations implemented using the excellent [Anise](https://github.com/nyx-space/anise) library.

The Development Ephemeris 440s (DE440s) provides high-precision positions of all major solar system bodies using numerical integration over the time span of 1849 to 2150. They provide meter-level accuracy or better for planetary positions, but require downloading and managing SPICE kernel data files. Brahe generally will download and cache these files automatically on first use.

## Usage Examples

### Sun and Moon Perturbations

Compute the combined gravitational acceleration from the Sun and Moon on a satellite.

=== "Python"

    ```python
    --8<-- "./examples/orbit_dynamics/third_body_sun_moon.py:8"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/orbit_dynamics/third_body_sun_moon.rs:4"
    ```

## See Also

- [Library API Reference: Third-Body](../../library_api/orbit_dynamics/third_body.md)
- [Datasets: NAIF](../datasets/naif.md) - DE440s ephemeris data
- [Orbital Dynamics Overview](index.md)

## References

Montenbruck, O., & Gill, E. (2000). *Satellite Orbits: Models, Methods, and Applications*. Springer. Section 3.3: Gravitational Perturbations.
