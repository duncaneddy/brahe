# STrajectory6

## Overview

`STrajectory6` is a static 6-dimensional trajectory class optimized for orbital state representation. Unlike `DTrajectory` which has dynamic dimensionality, `STrajectory6` is specifically designed for storing and interpolating 6D state vectors (position and velocity).

## Key Concepts

### Static Dimensionality

The "S" in `STrajectory6` stands for "Static" - the trajectory always stores exactly 6 dimensions. This specialization enables:

- **Memory efficiency**: Fixed-size allocations
- **Performance optimization**: Compile-time optimizations for 6D operations
- **Type safety**: Guaranteed 6D structure at compile time

### Typical Use Cases

`STrajectory6` is ideal for:

- Storing propagated orbital states
- Interpolating spacecraft trajectories
- Recording position and velocity time series
- Working with Cartesian state vectors

## State Representation

States are stored as 6-element vectors:

$$
\mathbf{x} = \begin{bmatrix} x \\ y \\ z \\ v_x \\ v_y \\ v_z \end{bmatrix}
$$

Where:
- Position components (x, y, z) are in meters
- Velocity components (vx, vy, vz) are in meters/second

## Common Operations

### Creating a Trajectory

Trajectories are typically created by:

1. Initializing an empty trajectory
2. Adding states at specific epochs during propagation
3. Setting interpolation method

### Interpolation

`STrajectory6` supports multiple interpolation methods:

- **Linear**: Fast, simple interpolation between points
- **Lagrange**: Higher-order polynomial interpolation
- **Hermite**: Cubic interpolation using derivatives

### Querying States

Once populated, you can query states at any epoch within the trajectory's time span. The trajectory will automatically interpolate between stored states.

## Comparison with Other Trajectory Types

| Feature | STrajectory6 | DTrajectory | OrbitTrajectory |
|---------|-------------|-------------|-----------------|
| Dimensionality | Fixed (6D) | Dynamic (any) | Fixed (orbital) |
| Optimization | Compile-time | Runtime | Orbital-specific |
| Use Case | Cartesian states | Generic data | Orbital elements |
| Memory | Most efficient for 6D | Flexible | Element-optimized |

## Performance Considerations

- **Best for**: Large numbers of 6D state queries
- **Memory**: Fixed overhead per state point
- **Speed**: Fastest interpolation for 6D data
- **Scalability**: Efficient up to millions of state points

## Further Reading

- See [OrbitTrajectory](orbit_trajectory.md) for orbital element representation
- See [DTrajectory](dtrajectory.md) for dynamic dimensionality
- See [Trajectories Overview](index.md) for comparison of all trajectory types

## API Reference

For detailed API documentation, see [STrajectory6 API Reference](../../library_api/trajectories/strajectory6.md).
