# Orbital Properties

The `orbits` module also provides functions to help derive useful properties
of an orbit.

## Periapsis Properties

Two common properties of interest are the distance and velocity of an object at the periapsis of 
its orbit. The periapsis will be the point of the closest approach as well as the point of 
greatest speed throughout the orbit.

???+ info

    The word _**periapsis**_ is formed by combination of the Greek words "_peri-_", meaning around, 
    about and "_apsis_"
    meaning "arch or vault". An apsis is the farthest or nearest point in the orbit of a 
    planetary body about its primary body. 

    Thereforce _periapsis_ of an orbit is the point of closest approach of the orbiting body with 
    respect to its central body. The suffix can be modified to indicate the point of 
    closest approach to a specific celestical body. The _perigee_ is the point of cloest approach to
    an object orbiting Earth. The _perihelion_ is the point of closest approach to the Sun.

The periapsis velocity is given by `periapsis_velocity` or, for an Earth-orbiting object, the 
function `perigee_velocity` can be used. `perigee_velocity` simplified the call by supplying the 
gravitational parameter of Earth to the function call. Periapsis velocity is calculated by
$$
v_{p} = \sqrt{\frac{\mu}{a}}\sqrt{\frac{1+e}{1-e}}
$$
where $\mu$ is the gravitational constant of the central body, $a$ is the semi-major axis of the 
orbit, and $e$ is the eccentricity.

Another useful parameter of the periapsis is the distance of the object to the central body. 
Equation (2-75) from Vallado[^1]
$$
r_p = \frac{a(1-e^2)}{1+e} = a(1-e)
$$

=== "Python"

    ``` python
    --8<-- "./examples/orbits/periapsis_properties.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbits/periapsis_properties.rs:7"
    ```

## Apoapsis Properties

The distance and velocity of an object at the apoapsis of an orbit. The apoapsis is the 
furthest point from the central body. It is also the point of lowest speed throughout the orbit.

???+ info

    The word _**apoapsis**_ is formed by combination of the Greek words "_apo-_", meaning away from,
    separate, or apart from and the word "_apsis_".

    Thereforce _apoapsis_ of an orbit is the farthest point of an orbiting body with 
    respect to its central body. The suffix can be modified to indicate the farthest point
    with respect to a specific primary celestical body. The _apogee_ is furthest point away for 
    an object orbiting Earth. The _aphelion_ is the furthest away from an object orbiting the Sun.


The apoapsis velocity is given by `apoapsis_velocity` or, for an Earth-orbiting object, the
function `apoapsis_velocity` can be used. Apoapsis velocity is given by
$$
v_{a} = \sqrt{\frac{\mu}{a}}\sqrt{\frac{1-e}{1+e}}
$$
The distance of the object to the central body is given by
$$
r_a = \frac{a(1-e^2)}{1+e} = a(1+e)
$$

!!! warning

    The apoapsis position and velocity are valid for elliptic and circular orbits. For parabolic 
    and hyperbolic orbits these two quantities are undefined.

=== "Python"

    ``` python
    --8<-- "./examples/orbits/apoapsis_properties.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbits/apoapsis_properties.rs:7"
    ```

## Mean Motion and Semi-major Axis

A satellite's average angular rate over one orbit is it's _mean motion_ $n$, which can be 
calculated 
from the semi-major axis $a$ and central body's gravitational parameter $\mu$
$$
n = \sqrt{\frac{\mu}{a^3}}
$$
`mean_motion` will calculate the mean motion of an object for an Earth-orbiting object while, 
`mean_motion_general` can calculate it for any arbitrary body when provided the graviational 
parameter of that body.

Since for some orbital data exchange formats, an object's orbit is characterized in terms of 
its mean motion instead of semi-major axis, brahe provides `semimajor_axis` and 
`semimajor_axis_general` to invert above equation and recover the semi-major axis
$$
a = \sqrt[3]{\frac{\mu}{n^2}}
$$

## Orbital Period

The orbital period $T$ of a satellite is the time it takes for an orbit to progress through a full 
revolution. It is given by
$$
T = 2\pi\sqrt{\frac{a^3}{\mu}}
$$
`orbital_period` will return the orbital period for an Earth-orbiting object and 
`orbital_period_general` will do the same for any arbitrary body for which the gravitational 
constant is known.

The plot below shows how, for a circular orbit, as the semi-major axis increases, the orbital 
period increases while the speed decreases.

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/fig_orbital_period_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/fig_orbital_period_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="fig_orbital_period.py"
    --8<-- "./plots/fig_orbital_period.py:8"
    ```

## Sun-Synchronous Inclination

A _**Sun-syncrhonous**_ orbit is one whose nodal precession rate matches the average rate of the 
Earth's rotation about the Sun. That is, it is an orbit where the right ascension of the 
ascending node changes at the same rate as the Sun moves relative to the Earth. Sun-synchronous 
orbits are often highly relevant an useful for optical Earth observation 
satellites that desire to have consistent illumination conditions. A Sun-synchronous orbit is 
guaranteed to cross the equator at the same local time (e.g. 2pm) at each crossing event.

Due to Earth's oblateness known as the $J_2$ zonal harmonic, frequently referred as simply $J_2$,
there is a constant, secular drift of the right ascension of all Earth orbiting objects. Since 
the $J_2$ harmonic is the second largest after that of point-mass gravity, it is a dominant 
effect on the orbit

$$
\dot{\Omega} = -\frac{3nR^2_EJ_2}{2a^2(1-e^2)^2}
$$

when combined with the required nodal precession rate for sun synchronicity

$$
\dot{\Omega} = \frac{360 \; \text{deg}}{1 \; \text{year}}\frac{1 \; \text{year}}{365.2421897 \; 
\text{day}} = 0.98564736 \frac{\text{deg}}{\text{day}}
$$

can be rearranged to solve for the inclination as a function of semi-major axis and eccentricity

$$
i = \arccos{\Big(-\frac{2a^{\frac{7}{2}}\dot{\Omega}_{ss}(1-e^2)^2}{3R^2_EJ_2\sqrt{\mu}}\Big)}
$$

The function `sun_syncrhonous_inclination` calculates this inclination.

The figure below shows how the inclination required to maintain a sun-synchronous orbit varies 
with altitude. Rocket launches to Sun-synchronous orbits will commonly have a fixed altitude 
(around 500 to 1000 kilometers for many Earth observation missions) and zero eccentricity. The 
launch provider will select the inclination the rocket is then determined by the above equation 
to provide the desired effect.

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/fig_sun_synchronous_inclination_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/fig_sun_synchronous_inclination_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="fig_sun_synchronous_inclination.py"
    --8<-- "./plots/fig_sun_synchronous_inclination.py:8"
    ```

[^1]: D. Vallado, *Fundamentals of Astrodynamics and Applications (4th Ed.)*, 2010  
[https://celestrak.com/software/vallado-sw.php](https://celestrak.com/software/vallado-sw.php)