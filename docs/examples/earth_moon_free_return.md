# Earth-Moon Free-Return Trajectory

In this example we'll design and fly an Earth-Moon [**free-return trajectory**](https://en.wikipedia.org/wiki/Free-return_trajectory): a path that departs a low Earth parking orbit, coasts out to the Moon, swings around its far side, and lets lunar gravity bend it back onto an Earth-return leg without any dedicated return burn. This is the trajectory class that gave the early Apollo missions their abort safety margin - if the service propulsion system had failed on the way out, the spacecraft would still have looped around the Moon and returned to a survivable re-entry. Apollo 13's abort after its oxygen tank ruptured depended on exactly this property. Artemis I did *not* fly a strict free return; Artemis II, the first crewed Artemis flight, does.

We'll use the `NumericalOrbitPropagator` integrated about the Earth-Moon barycenter (the EMBI frame) with a 5x5 Earth spherical-harmonic field plus Moon and Sun third-body perturbations, target the translunar injection (TLI) delta-v with a bisection search, and apply the burn with a `TimeEvent` callback. A terminal `ValueEvent` on geodetic altitude stops the flight at the atmospheric entry interface on the way home. Finally we'll visualize the result in the Earth-Moon Rotating (EMR) frame, where the trajectory traces its characteristic figure-8.

---

## What "Free Return" Means

A free-return trajectory is a solution of the Earth-Moon-spacecraft three-body problem whose outbound leg is aimed so that the lunar flyby rotates the velocity vector back toward Earth. No propulsion is needed after the initial injection: the Moon's gravity does the work of turning the spacecraft around. The defining property is passive safety - once on the trajectory, the spacecraft returns to Earth's vicinity on its own.

The flyby must be a genuine **circumlunar** pass: the spacecraft crosses to the far side of the Moon and swings around it, so its motion about the Moon is retrograde with respect to the Moon's orbital motion about Earth. That far-side geometry is what bends the path steeply back toward home. A pass on the near side, in front of the Moon, deflects the trajectory the wrong way. The pass must also be close enough to bend the path back to Earth but not so close that the perilune drops below the surface. The window between "escapes" and "impacts the Moon" is what makes free-return design a targeting problem rather than a closed-form calculation.

## Geometry

The departure geometry fixes everything about the transfer except its energy. We depart a 400 km (ISS-like) circular parking orbit from a point near the **antipode** of the Moon's position at the expected arrival time, burning prograde in the Moon's instantaneous orbital plane so the transfer apogee reaches toward the Moon roughly half an orbit later.

A real mission targets a two-dimensional B-plane at the Moon (a miss distance and an approach angle). Here `AIM_OFFSET_DEG` is a simplified stand-in for that second dimension: rotating the departure point ahead of the pure antipode about the orbit normal sets up a flyby that swings around the far side of the Moon rather than passing in front of it. We use `spk_state` to query the Moon's Earth-relative state from the DE440s ephemeris and build an orthonormal departure frame from it. Because the mission is integrated about the Earth-Moon barycenter, the departure state is translated from ECI into the EMBI frame with `state_eci_to_emb`.

``` python
--8<-- "./examples/examples/earth_moon_free_return.py:preamble"
```

``` python
--8<-- "./examples/examples/earth_moon_free_return.py:geometry"
```

## Force Model

A free return is shaped by three bodies. We integrate about the Earth-Moon barycenter, so the central gravity term is zero - the barycenter has no mass of its own. Earth carries a 5x5 spherical-harmonic field as an attributed third body (evaluated at the spacecraft's Earth-relative position), and the Moon and Sun are point-mass perturbers from the DE440s ephemeris. This is the EMB-centered force-model pattern used for cislunar propagation: the integration state stays barycentric while Earth-fidelity gravity still acts near perigee. The lunar term is what bends the trajectory home; the Sun is a smaller but non-negligible perturbation accumulated over the multi-day flight.

``` python
--8<-- "./examples/examples/earth_moon_free_return.py:force_model"
```

## Targeting the TLI Delta-V

This example's purpose is showing the Earth-Moon rotating frame, so we take some shortcuts to generate the trajectory - a fixed departure geometry and a one-dimensional bisection on the TLI delta-v - which in practice should not be done. Real mission design uses Lambert solvers and B-plane targeting.

With the departure geometry fixed, the only free parameter is the TLI delta-v, which sets the transfer's energy. The miss distance at the Moon is a **V-shaped** function of that delta-v: too little energy and the transfer apogee never reaches lunar distance, too much and the spacecraft races out ahead of the Moon. The free-return solutions live on the **ascending branch** of the V, where the perilune radius grows with delta-v. We run a coarse scan to reveal the V and locate its minimum, then bracket the target perilune on the ascending branch and refine the delta-v with a bisection.

The same builder that scores a candidate during targeting flies the final mission, so the trajectory the search converges on is exactly the one flown.

``` python
--8<-- "./examples/examples/earth_moon_free_return.py:targeting"
```

!!! note "Why a coarse scan first?"
    A naive bisection on delta-v cannot converge here: the miss distance is not monotonic, so a bracket chosen blindly may straddle the bottom of the V, and the descending branch reaches the same perilune values without ever returning to Earth. The coarse scan finds the V's minimum so the bisection can be restricted to the ascending branch, where perilune increases monotonically with delta-v and the free-return solution for this geometry lives. The final propagation, terminated by the entry event, is what actually confirms the Earth return.

## Flying the Mission

We fly the tuned design as it would be flown. The propagator starts in the parking orbit, coasts one short arc, and applies the TLI impulsively through a `TimeEvent` callback. Because the integration state is barycentric, the burn is applied along the spacecraft's *Earth-relative* velocity - the state is translated to ECI, the delta-v is added, and it is translated back. The flight is stopped by a terminal `ValueEvent` on geodetic altitude, triggered on decreasing altitude at 120 km - the atmospheric entry interface where a real capsule would begin re-entry. A plain `AltitudeEvent` would measure altitude above the barycenter rather than the Earth, so the custom value function translates the barycentric state to ECI before computing altitude. The flight time to that point falls out of the propagation rather than being prescribed.

``` python
--8<-- "./examples/examples/earth_moon_free_return.py:final_run"
```

## Distance History

Sampling the recorded trajectory gives the distance from Earth and from the Moon over the whole flight. The Earth distance climbs to nearly lunar distance and returns to the entry interface; the Moon distance dips sharply at the flyby. The final epoch is appended to the sample times so the re-entry point itself is captured.

``` python
--8<-- "./examples/examples/earth_moon_free_return.py:distance_history"
```

``` python
--8<-- "./examples/examples/earth_moon_free_return.py:distance_plot"
```

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/earth_moon_free_return_distance_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/earth_moon_free_return_distance_dark.html"  loading="lazy"></iframe>
</div>

The Moon-distance curve reaches its minimum - the perilune - a little over three days after departure, and the Earth-distance curve turns over shortly after, marking the moment the lunar flyby has bent the trajectory back toward home. The spacecraft reaches the entry interface after about six and a half days.

## Figure-8 in the Rotating Frame

In an inertial frame the free-return path is an unremarkable elongated loop. Its structure only becomes visible in the **Earth-Moon Rotating (EMR)** frame, which co-rotates with the Earth-Moon line so the Moon sits fixed on one axis. In that frame the outbound leg, the far-side lunar swing-by, and the return leg trace the characteristic figure-8 that is the signature of a free-return trajectory - the outbound and return legs cross near Earth. We build two views: a 3D view around the textured Earth and Moon, and a top-down (X-Y) view with the bodies drawn to scale. Both carry direction-of-travel arrows along the path, and the fixed body spheres are placed at the perilune epoch so the swing-by aligns with the Moon.

``` python
--8<-- "./examples/examples/earth_moon_free_return.py:plot_emr"
```

<div class="plotly-embed medium">
  <iframe class="only-light" src="../figures/earth_moon_free_return_emr_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/earth_moon_free_return_emr_dark.html"  loading="lazy"></iframe>
</div>

The top-down view makes the figure-8 unmistakable: the outbound leg swings around the far side of the Moon and the return leg crosses it near Earth, closing the loop.

<div class="plotly-embed medium">
  <iframe class="only-light" src="../figures/earth_moon_free_return_emr_2d_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/earth_moon_free_return_emr_2d_dark.html"  loading="lazy"></iframe>
</div>

*Body textures: [Solar System Scope](https://www.solarsystemscope.com/textures/), CC BY 4.0.*

## Full Code Example

??? "Full Code"

    ```python title="earth_moon_free_return.py"
    --8<-- "./examples/examples/earth_moon_free_return.py:all"
    ```

---

## See Also

- [LRO Lunar Orbit](lro_lunar_orbit.md) - Propagating a Moon-centered orbit with the lunar force model
- [Numerical Orbit Propagation](../learn/orbit_propagation/numerical_propagation/index.md) - Propagator fundamentals
- [Event Detection](../learn/orbit_propagation/numerical_propagation/event_detection.md) - Time, altitude, and value event detectors
- [Force Models](../learn/orbit_propagation/numerical_propagation/force_models.md) - Configuring third-body perturbations
- [Synodic Frame Plots](../learn/plots/synodic_plots.md) - Visualizing trajectories in rotating frames
