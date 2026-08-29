# AttitudeTrajectory

`AttitudeTrajectory` is a chronologically sorted collection of `AttitudeState` samples (a unit quaternion plus an optional body angular velocity) relating two `AttitudeFrame` endpoints, with slerp, linear, and Lagrange interpolation.

AttitudeTrajectory has the same API as [Trajectory](trajectory.md), plus quaternion-aware interpolation and the `AttitudeProvider` accessors (`quaternion`, `angular_velocity`, `euler_angle`, `euler_axis`, `rotation_matrix`) documented below.

::: brahe.AttitudeTrajectory
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.AttitudeState
    options:
      show_root_heading: true
      show_root_full_path: false

---

## See Also

- [AttitudeTrajectory Guide](../../learn/trajectories/attitude_trajectory.md) — Canonical state, interpolation methods, and frame composition
- [Trajectory](trajectory.md) — Dynamic-dimension trajectory
- [AEM API Reference](../ccsds/aem.md) — Converting AEM segments to `AttitudeTrajectory`
- [Attitude Frame](../attitude/attitude_frame.md) — Frame specifications
