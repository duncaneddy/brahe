# Trajectories

**Module**: `brahe.trajectories`

Trajectory containers for storing, managing, and interpolating time-series state data.

## Trajectory Types

### [DTrajectory](dtrajectory.md)
**Dynamic-dimension** trajectory container where dimension is set at runtime. Flexible for storing any N-dimensional state data.

### [STrajectory6](strajectory6.md)
**Static 6-dimensional** trajectory optimized for orbital state vectors [x, y, z, vx, vy, vz]. Faster than DTrajectory for fixed-size data.

### [SOrbitTrajectory](sorbit_trajectory.md)
**Specialized orbital** trajectory with frame-aware storage and automatic coordinate transformations.

---

## See Also

- [InterpolationMethod](../orbits/enums.md#interpolationmethod) - Interpolation options
- [OrbitFrame](../orbits/enums.md#orbitframe) - Frame specifications
- [KeplerianPropagator](../propagators/keplerian_propagator.md) - Analytical orbit propagation
- [SGPPropagator](../propagators/sgp_propagator.md) - SGP4/SDP4 orbit propagation
