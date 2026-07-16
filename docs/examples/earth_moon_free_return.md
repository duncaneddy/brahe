# Earth-Moon Free-Return Trajectory

In this example we'll design and fly an Earth-Moon **free-return trajectory**: a path that departs a low Earth parking orbit, coasts out to the Moon, and lets lunar gravity bend it back onto an Earth-return leg without any dedicated return burn. This is the trajectory class that gave the early Apollo missions their abort safety margin - if the service propulsion system had failed on the way out, the spacecraft would still have looped around the Moon and returned to a survivable re-entry. Apollo 13 depended on exactly this property after its oxygen tank ruptured, and Artemis I flew an extended "hybrid" free return uncrewed in 2022.

We'll use the `NumericalOrbitPropagator` with Earth point-mass gravity plus Moon and Sun third-body perturbations, target the translunar injection (TLI) delta-v with a bisection search, and apply the burn with a `TimeEvent` callback. A terminal `AltitudeEvent` stops the flight at the atmospheric entry interface on the way home. Finally we'll visualize the result in the Earth-Moon Rotating (EMR) frame, where the trajectory traces its characteristic figure-8.

---

## What "Free Return" Means

A free-return trajectory is a solution of the Earth-Moon-spacecraft three-body problem whose outbound leg is aimed so that the lunar flyby rotates the velocity vector back toward Earth. No propulsion is needed after the initial injection: the Moon's gravity does the work of turning the spacecraft around. The defining property is passive safety - once on the trajectory, the spacecraft returns to Earth's vicinity on its own.

The design has two competing requirements. The lunar flyby must pass close enough to the Moon to bend the path steeply back toward Earth, but not so close that the perilune drops below the surface. A pass that is too distant barely deflects the trajectory and the spacecraft escapes onto a distant loop or a heliocentric orbit instead of returning. The window between "escapes" and "impacts the Moon" is what makes free-return design a targeting problem rather than a closed-form calculation.

## Geometry

The departure geometry fixes everything about the transfer except its energy. We depart a 185 km circular parking orbit from a point near the **antipode** of the Moon's position at the expected arrival time, burning prograde in the Moon's instantaneous orbital plane so the transfer apogee reaches toward the Moon roughly half an orbit later.

A real mission targets a two-dimensional B-plane at the Moon (a miss distance and an approach angle). Here `AIM_OFFSET_DEG` is a simplified stand-in for that second dimension: rotating the departure point ahead of the pure antipode about the orbit normal (16 degrees for this design) sets up a flyby that bends the trajectory back onto an Earth-return leg rather than flinging it outward. We use `spk_state` to query the Moon's Earth-relative state from the DE440s ephemeris and build an orthonormal departure frame from it.

``` python
--8<-- "./examples/examples/earth_moon_free_return.py:preamble"
```

``` python
--8<-- "./examples/examples/earth_moon_free_return.py:geometry"
```

## Force Model

A free return is shaped by three bodies. We configure a point-mass Earth as the central body and add the Moon and Sun as third-body perturbers from the DE440s ephemeris. The lunar term is what bends the trajectory home; the Sun is a smaller but non-negligible perturbation accumulated over the eight-day flight. We deliberately omit Earth's oblateness, drag, and radiation pressure - none of them materially change the free-return geometry at these altitudes and distances.

``` python
--8<-- "./examples/examples/earth_moon_free_return.py:force_model"
```

## Targeting the TLI Delta-V

With the departure geometry fixed, the only free parameter is the TLI delta-v, which sets the transfer's energy. The miss distance at the Moon is a **V-shaped** function of that delta-v: too little energy and the transfer apogee never reaches lunar distance, too much and the spacecraft races out ahead of the Moon. The minimum of the V is the closest reachable approach for this geometry.

The free-return solutions live on the **ascending branch** of the V - the side where the perilune radius grows with delta-v. On that branch, increasing the delta-v widens the flyby from a near-impact toward a distant pass, and the return leg's closest approach to Earth sweeps up through the atmospheric entry corridor. We first run a coarse scan to reveal the V and locate its minimum, then bracket the target perilune on the ascending branch and refine the delta-v with a bisection.

This scalar search is an honest stand-in for the Lambert solvers and differential-correction targeters a real mission would use. Because the geometry above already fixes the departure point, plane, and direction, a one-dimensional search on energy is enough to place the flyby.

``` python
--8<-- "./examples/examples/earth_moon_free_return.py:targeting"
```

!!! note "Why a coarse scan first?"
    A naive bisection on delta-v cannot converge here: the miss distance is not monotonic, so a bracket chosen blindly may straddle the bottom of the V, and the descending branch reaches the same perilune values without ever returning to Earth. The coarse scan finds the V's minimum so the bisection can be restricted to the ascending branch, where perilune increases monotonically with delta-v and the free-return solution for this geometry lives. The final propagation, terminated by the entry event, is what actually confirms the Earth return.

## Flying the Mission

We fly the tuned design as it would be flown. The propagator starts in the parking orbit, coasts one short arc, and applies the TLI impulsively through a `TimeEvent` callback that adds the delta-v along the velocity vector. A terminal `AltitudeEvent` at 120 km, triggered on decreasing altitude, stops the propagation at the atmospheric entry interface on the return leg - the point where a real capsule would begin re-entry. The flight time to that point falls out of the propagation rather than being prescribed.

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

The Moon-distance curve reaches its minimum - the perilune - just under four days after departure, and the Earth-distance curve turns over shortly after, marking the moment the lunar flyby has bent the trajectory back toward home.

## Figure-8 in the Rotating Frame

In an inertial frame the free-return path is an unremarkable elongated loop. Its structure only becomes visible in the **Earth-Moon Rotating (EMR)** frame, which co-rotates with the Earth-Moon line so the Moon sits fixed on one axis. In that frame the outbound leg, the lunar swing-by, and the return leg trace the characteristic figure-8 that is the signature of a free-return trajectory. `plot_earth_moon_rotating_3d` transforms the trajectory into the EMR frame and renders it around a textured Earth and Moon.

``` python
--8<-- "./examples/examples/earth_moon_free_return.py:plot_emr"
```

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/earth_moon_free_return_emr_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/earth_moon_free_return_emr_dark.html"  loading="lazy"></iframe>
</div>

*Body textures: [Solar System Scope](https://www.solarsystemscope.com/textures/), CC BY 4.0.*

## Validation

We confirm the trajectory is a genuine free return: the perilune passes above the lunar surface but within 20,000 km of the Moon's center, and the return leg descends below 1,000 km altitude, closing the loop back at Earth. Because the propagation stops at the 120 km entry interface, the minimum Earth distance measured is that entry crossing; the true geometric perigee lies below it, inside the atmosphere.

``` python
--8<-- "./examples/examples/earth_moon_free_return.py:validation"
```

## Trajectory Summary

| Parameter | Value |
|-----------|-------|
| Parking orbit altitude | 185 km |
| Transfer time (design) | 3.1 days |
| Aim offset from antipode | 16 deg |
| Target perilune radius | $R_{Moon} + 12{,}000$ km |
| TLI delta-v | ~3.14 km/s |
| Perilune altitude | ~12,000 km |
| Return entry-interface altitude | ~120 km |
| Flight time to re-entry | ~8.1 days |

## Full Code Example

```python title="earth_moon_free_return.py"
--8<-- "./examples/examples/earth_moon_free_return.py:all"
```

---

## See Also

- [LRO Lunar Orbit](lro_lunar_orbit.md) - Propagating a Moon-centered orbit with the lunar force model
- [Numerical Orbit Propagation](../learn/orbit_propagation/numerical_propagation/index.md) - Propagator fundamentals
- [Event Detection](../learn/orbit_propagation/numerical_propagation/event_detection.md) - Time and altitude event detectors
- [Force Models](../learn/orbit_propagation/numerical_propagation/force_models.md) - Configuring third-body perturbations
- [Synodic Frame Plots](../learn/plots/synodic_plots.md) - Visualizing trajectories in rotating frames
