# SOrbitTrajectory

`SOrbitTrajectory` is a specialized trajectory container for orbital mechanics that stores states in a specific reference frame (ECI or ECEF) and can automatically transform between frames when querying.

SOrbitTrajectory has the same API as [STrajectory6](strajectory6.md) and [DTrajectory](dtrajectory.md), plus frame awareness.


::: brahe.SOrbitTrajectory
    options:
      show_root_heading: true
      show_root_full_path: false

---

## See Also

- [STrajectory6](strajectory6.md) - Non-frame-aware 6D trajectory
- [DTrajectory](dtrajectory.md) - Dynamic-dimension trajectory
- [OrbitFrame](../orbits/enums.md#orbitframe) - Frame specifications
