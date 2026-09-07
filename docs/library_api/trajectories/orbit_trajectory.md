# OrbitTrajectory

`OrbitTrajectory` is a specialized trajectory container for orbital mechanics that stores states in a specific `ReferenceFrame` and can automatically transform between frames when querying.

OrbitTrajectory has the same API as [Trajectory](trajectory.md), plus frame awareness.

::: brahe.OrbitTrajectory
    options:
      show_root_heading: true
      show_root_full_path: false

---

## See Also

- [Trajectory](trajectory.md) - Dynamic-dimension trajectory
- [CelestialFrame](../frames/router.md#brahe.CelestialFrame) - Named celestial reference frames
- [ReferenceFrame](../frames/frame.md#brahe.ReferenceFrame) - Unified frame identity accepted by trajectories
