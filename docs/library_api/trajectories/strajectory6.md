# STrajectory6

::: brahe.STrajectory6
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 2

Static 6-dimensional trajectory optimized for orbital state storage.

## Overview

`STrajectory6` provides compile-time optimized storage for 6-dimensional Cartesian states `[x, y, z, vx, vy, vz]`. It offers the best performance for standard orbital mechanics applications.

## Features

- **Fixed dimension**: Always 6D (compile-time optimization)
- **Lower memory overhead**: More efficient than DTrajectory
- **Fastest performance**: Optimized for Cartesian orbital states
- **Full interpolation**: Supports linear, cubic, and Lagrange interpolation
- **Eviction policies**: Memory management via automatic state removal

## When to Use

Use `STrajectory6` when:

- Storing standard Cartesian orbital states
- Performance is critical
- State dimension is always 6
- Not using orbit-specific features (use OrbitTrajectory for that)

---

## See Also

- [DTrajectory](dtrajectory.md) - Variable dimension trajectory
- [OrbitTrajectory](orbit_trajectory.md) - Orbit-aware trajectory
- [Trajectories Overview](../../learn/trajectories/index.md)
