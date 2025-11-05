# Cartesian ↔ Orbital Element Transformations

The functions described here convert between Keplerian orbital elements and Cartesian state vectors. While these transformations are part of the "coordinates" module, they specifically deal with orbital mechanics - converting between two different coordinate representations of a satellite's orbit.

Understanding both representations is essential: Keplerian elements provide intuitive orbital parameters like size, shape, and orientation, while Cartesian states are necessary for numerical orbit propagation and applying perturbations.

For complete API details, see the [Cartesian Coordinates API Reference](../../library_api/coordinates/cartesian.md).

## Orbital Representations

### Keplerian Orbital Elements

Keplerian elements describe an orbit using six classical parameters:

- $a$: Semi-major axis (meters) - defines the orbit's size
- $e$: Eccentricity (dimensionless) - defines the orbit's shape (0 = circular, 0 < e < 1 = elliptical)
- $i$: Inclination (radians or degrees) - tilt of orbital plane relative to equator
- $\Omega$: Right ascension of ascending node (radians or degrees) - where orbit crosses equator going north
- $\omega$: Argument of periapsis (radians or degrees) - where orbit is closest to Earth
- $M$: Mean anomaly (radians or degrees) - position of satellite along orbit

In brahe, the combined vector has ordering `[a, e, i, Ω, ω, M]`

!!! info
    Brahe uses **mean anomaly** as the default anomaly representation for Keplerian elements. Other anomaly types (eccentric, true) can be converted using the anomaly conversion functions in the [Orbits module](../../library_api/orbits/index.md).

### Cartesian State Vectors

Cartesian states represent position and velocity in three-dimensional space:

- **Position**: $[p_x, p_y, p_z]$ in meters
- **Velocity**: $[v_x, v_y, v_z]$ in meters per second

In brahe, the state vector is combined as `[p_x, p_y, p_z, v_x, v_y, v_z]`

Cartesian states are typically expressed in an inertial reference frame like Earth-Centered Inertial (ECI), where the axes are fixed with respect to the stars rather than rotating with Earth.

!!! info
    All position and velocity components in Cartesian states are in SI base units (meters and meters per second).

    They **must** be in SI base units for inputs and are always returned in SI base units.

## Converting Orbital Elements to Cartesian

The most common workflow is to start with intuitive orbital parameters and convert them to Cartesian states for propagation.

### Using Degrees

When working with human-readable orbital parameters, degrees are more intuitive:

=== "Python"

    ``` python
    --8<-- "./examples/coordinates/elements_to_cartesian_deg.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/coordinates/elements_to_cartesian_deg.rs:4"
    ```

### Using Radians

For mathematical consistency or when working with data already in radians:

=== "Python"

    ``` python
    --8<-- "./examples/coordinates/elements_to_cartesian_rad.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/coordinates/elements_to_cartesian_rad.rs:4"
    ```

!!! info
    The `AngleFormat` parameter only affects the three angular elements (i, Ω, ω, M). Semi-major axis is always in meters, and eccentricity is always dimensionless.

## Converting Cartesian to Orbital Elements

After propagating or receiving Cartesian state data, you often want to convert back to orbital elements for interpretation and analysis.

=== "Python"

    ``` python
    --8<-- "./examples/coordinates/cartesian_to_orbital_elements.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/coordinates/cartesian_to_orbital_elements.rs:4"
    ```


---

## See Also

- [Cartesian Coordinates API Reference](../../library_api/coordinates/cartesian.md) - Complete function documentation
- [Orbital Mechanics](../../library_api/orbits/index.md) - Related orbital mechanics functions
- Anomaly Conversions - Converting between mean, eccentric, and true anomaly
