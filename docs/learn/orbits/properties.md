# Orbital Properties

The `orbits` module provides functions to compute essential properties of satellite orbits, including orbital period, mean motion, periapsis/apoapsis characteristics, and specialized orbits like sun-synchronous configurations. These properties are fundamental for mission design, orbit determination, and trajectory analysis.

For complete API documentation, see [Orbits API Reference](../../library_api/orbits/index.md).

## Orbital Period

The orbital period $T$ of a satellite is the time it takes to complete one full revolution around the central body. It is related to the semi-major axis $a$ and gravitational parameter $\mu$ by:

$$
T = 2\pi\sqrt{\frac{a^3}{\mu}}
$$

The `orbital_period` function computes the period for Earth-orbiting objects, while `orbital_period_general` accepts an explicit gravitational parameter for any celestial body.

=== "Python"

    ``` python
    --8<-- "./examples/orbits/orbital_period.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbits/orbital_period.rs:7"
    ```

The plot below shows how orbital period and velocity vary with altitude for circular Earth orbits:

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/fig_orbital_period_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/fig_orbital_period_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="fig_orbital_period.py"
    --8<-- "./plots/fig_orbital_period.py:8"
    ```

### From State Vector

When orbital elements are unknown but you have a Cartesian state vector, `orbital_period_from_state` computes the period directly from position and velocity:

=== "Python"

    ``` python
    --8<-- "./examples/orbits/orbital_period_from_state.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbits/orbital_period_from_state.rs:8"
    ```

### Semi-major Axis from Period

The inverse relationship allows computing semi-major axis when orbital period is known (useful for mission design):

$$
a = \sqrt[3]{\frac{\mu T^2}{4\pi^2}}
$$

=== "Python"

    ``` python
    --8<-- "./examples/orbits/semimajor_axis_from_period.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbits/semimajor_axis_from_period.rs:8"
    ```

## Mean Motion

A satellite's average angular rate over one orbit is its _mean motion_ $n$, calculated from the semi-major axis and gravitational parameter:

$$
n = \sqrt{\frac{\mu}{a^3}}
$$

The `mean_motion` function computes this for Earth-orbiting objects, while `mean_motion_general` works for any celestial body. Both functions support output in radians or degrees per second via the `angle_format` parameter.

=== "Python"

    ``` python
    --8<-- "./examples/orbits/mean_motion.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbits/mean_motion.rs:7"
    ```

### Semi-major Axis from Mean Motion

Since orbital data formats like TLEs specify mean motion instead of semi-major axis, the inverse computation is essential:

$$
a = \sqrt[3]{\frac{\mu}{n^2}}
$$

=== "Python"

    ``` python
    --8<-- "./examples/orbits/semimajor_axis_from_mean_motion.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbits/semimajor_axis_from_mean_motion.rs:8"
    ```

## Periapsis Properties

The periapsis is the point of closest approach to the central body, where orbital velocity is greatest.

???+ info

    The word _**periapsis**_ is formed by combination of the Greek words "_peri-_" (meaning around, about) and "_apsis_" (meaning "arch or vault"). An apsis is the farthest or nearest point in the orbit of a planetary body about its primary body.

    Therefore _periapsis_ is the point of closest approach of the orbiting body with respect to its central body. The suffix can be modified to indicate the closest approach to a specific celestial body: _perigee_ for Earth, _perihelion_ for the Sun.

Brahe provides functions to compute periapsis velocity, distance, and altitude based on orbital elements.

### Velocity

The periapsis velocity is given by:

$$
v_{p} = \sqrt{\frac{\mu}{a}}\sqrt{\frac{1+e}{1-e}}
$$

where $\mu$ is the gravitational parameter, $a$ is the semi-major axis, and $e$ is the eccentricity.

### Distance

The periapsis distance from the center of the central body is (from Vallado[^1] Equation 2-75):

$$
r_p = \frac{a(1-e^2)}{1+e} = a(1-e)
$$

### Altitude

The periapsis altitude is the height above the surface of the central body:

$$
h_p = r_p - R_{body} = a(1-e) - R_{body}
$$

where $R_{body}$ is the radius of the central body. For Earth orbits, the `perigee_altitude` function provides a convenient wrapper using $R_{\oplus}$.

### Code Example

=== "Python"

    ``` python
    --8<-- "./examples/orbits/periapsis_properties.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbits/periapsis_properties.rs:7"
    ```

## Apoapsis Properties

The apoapsis is the farthest point from the central body, where orbital velocity is lowest.

???+ info

    The word _**apoapsis**_ is formed by combination of the Greek words "_apo-_" (meaning away from, separate, or apart from) and "_apsis_".

    Therefore _apoapsis_ is the farthest point of an orbiting body with respect to its central body. The suffix can be modified to indicate the farthest point from a specific celestial body: _apogee_ for Earth, _aphelion_ for the Sun.

Brahe provides functions to compute apoapsis velocity, distance, and altitude based on orbital elements.

!!! warning

    Apoapsis position, velocity, and altitude are only defined for elliptic and circular orbits. For parabolic and hyperbolic orbits, these quantities are undefined.

### Velocity

The apoapsis velocity is given by:

$$
v_{a} = \sqrt{\frac{\mu}{a}}\sqrt{\frac{1-e}{1+e}}
$$

### Distance

The apoapsis distance from the center of the central body is:

$$
r_a = \frac{a(1-e^2)}{1-e} = a(1+e)
$$

### Altitude

The apoapsis altitude is the height above the surface of the central body:

$$
h_a = r_a - R_{body} = a(1+e) - R_{body}
$$

where $R_{body}$ is the radius of the central body. For Earth orbits, the `apogee_altitude` function provides a convenient wrapper using $R_{\oplus}$.


### Code Example

=== "Python"

    ``` python
    --8<-- "./examples/orbits/apoapsis_properties.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbits/apoapsis_properties.rs:7"
    ```

## Sun-Synchronous Inclination

A _**sun-synchronous**_ orbit maintains a constant angle relative to the Sun by matching its nodal precession rate to Earth's annual revolution. The right ascension of the ascending node ($\Omega$) advances at the same rate as the Sun's apparent motion: approximately 0.9856°/day. This configuration is highly valuable for Earth observation satellites requiring consistent illumination conditions—a sun-synchronous satellite crosses the equator at the same local time on each pass (e.g., always at 2 PM).

Earth's oblateness, characterized by the $J_2$ zonal harmonic, causes secular drift in $\Omega$:

$$
\dot{\Omega} = -\frac{3nR^2_EJ_2}{2a^2(1-e^2)^2}\cos{i}
$$

For sun-synchronicity, this must equal:

$$
\dot{\Omega}_{ss} = \frac{360°}{1 \text{ year}} = 0.9856473598°/\text{day}
$$

Solving for inclination as a function of semi-major axis and eccentricity:

$$
i = \arccos{\left(-\frac{2a^{7/2}\dot{\Omega}_{ss}(1-e^2)^2}{3R^2_EJ_2\sqrt{\mu}}\right)}
$$

The `sun_synchronous_inclination` function computes this required inclination:

=== "Python"

    ``` python
    --8<-- "./examples/orbits/sun_synchronous_inclination.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbits/sun_synchronous_inclination.rs:8"
    ```

The plot below shows how the required inclination varies with altitude for sun-synchronous orbits:

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/fig_sun_synchronous_inclination_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/fig_sun_synchronous_inclination_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="fig_sun_synchronous_inclination.py"
    --8<-- "./plots/fig_sun_synchronous_inclination.py:8"
    ```

Most sun-synchronous Earth observation missions operate at altitudes between 500-1000 km with near-zero eccentricity. The launch provider selects the precise inclination based on the above equation to achieve the desired sun-synchronous behavior.

---

---

## See Also

- [Orbits API Reference](../../library_api/orbits/index.md) - Complete Python API documentation
- [Anomaly Conversions](anomalies.md) - Converting between true, eccentric, and mean anomaly

[^1]: D. Vallado, *Fundamentals of Astrodynamics and Applications (4th Ed.)*, 2010
