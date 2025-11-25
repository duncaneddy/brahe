# Trajectories

**Module**: `brahe.trajectories`

Trajectory containers for storing, managing, and interpolating time-series state data.

## Trajectory Types

### [Trajectory](trajectory.md)
**Dynamic-dimension** trajectory container where dimension is set at runtime. Flexible for storing any N-dimensional state data.

### [OrbitTrajectory](orbit_trajectory.md)
**Specialized orbital** trajectory with frame-aware storage and automatic coordinate transformations.

---

## See Also

- [InterpolationMethod](../orbits/enums.md#interpolationmethod) - Interpolation options
- [OrbitFrame](../orbits/enums.md#orbitframe) - Frame specifications
- [KeplerianPropagator](../propagators/keplerian_propagator.md) - Analytical orbit propagation
- [SGPPropagator](../propagators/sgp_propagator.md) - SGP4/SDP4 orbit propagation
