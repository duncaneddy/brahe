# STrajectory6

`STrajectory6` is a static, compile-time sized trajectory container optimized for 6-dimensional state vectors (position + velocity). Unlike `DTrajectory` which determines dimension at runtime, `STrajectory6` uses compile-time sizing for maximum performance and type safety.

Use `STrajectory6` when:

- Working with 6D orbital states (position + velocity or orbital elements)
- State dimension is always fixed at 6
- Performance is critical
- You want compile-time dimension checking

`STrajectory6` is part of the `STrajectory<R>` family which includes `STrajectory3` (3D, position only) and `STrajectory4` (4D, quaternions). For applications requiring frame or representation conversions, consider using [`OrbitTrajectory`](orbit_trajectory.md).

## Initialization

### Empty Trajectory

Create an empty 6D trajectory:

=== "Python"

    ```python
    --8<-- "./examples/trajectories/strajectory6/strajectory6_empty.py:8"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/trajectories/strajectory6/strajectory6_empty.rs:4"
    ```

### From Existing Data

Create a trajectory from existing epochs and 6D states:

=== "Python"

    ```python
    --8<-- "./examples/trajectories/strajectory6/strajectory6_from_data.py:8"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/trajectories/strajectory6/strajectory6_from_data.rs:4"
    ```

## Adding and Accessing States

### Adding States

Add 6D states to a trajectory:

=== "Python"

    ```python
    --8<-- "./examples/trajectories/strajectory6/strajectory6_add.py:8"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/trajectories/strajectory6/strajectory6_add.rs:4"
    ```

### Accessing by Index

Retrieve states and epochs by their index:

=== "Python"

    ```python
    --8<-- "./examples/trajectories/strajectory6/strajectory6_access_index.py:8"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/trajectories/strajectory6/strajectory6_access_index.rs:4"
    ```

### Accessing by Epoch

Get states at or near specific epochs:

=== "Python"

    ```python
    --8<-- "./examples/trajectories/strajectory6/strajectory6_access_epoch.py:8"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/trajectories/strajectory6/strajectory6_access_epoch.rs:4"
    ```

## Querying Trajectory Properties

Query the temporal extent and properties of a trajectory:

=== "Python"

    ```python
    --8<-- "./examples/trajectories/strajectory6/strajectory6_timespan.py:8"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/trajectories/strajectory6/strajectory6_timespan.rs:4"
    ```

## Interpolation

STrajectory6 supports linear interpolation to estimate states at arbitrary epochs:

=== "Python"

    ```python
    --8<-- "./examples/trajectories/strajectory6/strajectory6_interpolate.py:8"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/trajectories/strajectory6/strajectory6_interpolate.rs:4"
    ```

## Memory Management

STrajectory6 supports eviction policies for automatic memory management:

### Maximum Size Policy

Keep only the N most recent states:

=== "Python"

    ```python
    --8<-- "./examples/trajectories/strajectory6/strajectory6_evict_size.py:8"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/trajectories/strajectory6/strajectory6_evict_size.rs:4"
    ```

### Maximum Age Policy

Keep only states within a time window:

=== "Python"

    ```python
    --8<-- "./examples/trajectories/strajectory6/strajectory6_evict_age.py:8"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/trajectories/strajectory6/strajectory6_evict_age.rs:4"
    ```

## Iteration

Trajectories can be iterated to process all epoch-state pairs:

=== "Python"

    ```python
    --8<-- "./examples/trajectories/strajectory6/strajectory6_iterate.py:8"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/trajectories/strajectory6/strajectory6_iterate.rs:4"
    ```

## Matrix Export

Convert trajectory data to matrix format for analysis or export:

=== "Python"

    ```python
    --8<-- "./examples/trajectories/strajectory6/strajectory6_matrix.py:8"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/trajectories/strajectory6/strajectory6_matrix.rs:4"
    ```

## Performance Benefits

`STrajectory6` uses compile-time sized vectors (`SVector<f64, 6>` in Rust, fixed-size arrays in Python) which provide several benefits:

**Memory Layout**: Contiguous memory allocation enables better cache utilization and SIMD optimizations.

**Type Safety**: Dimension mismatches are caught at compile time in Rust, preventing runtime errors.

**Optimization**: Compilers can optimize operations on fixed-size arrays more aggressively than dynamic allocations.

**Stack Allocation**: Small fixed-size vectors can be allocated on the stack, avoiding heap allocations.

For most orbital mechanics applications with 6D states, `STrajectory6` provides the best balance of performance and ease of use.

## Other Static Trajectory Types

The `STrajectory<R>` family includes other compile-time sized variants:

- **`STrajectory3`**: 3D states (position only, or other 3D data)
- **`STrajectory4`**: 4D states (quaternions for attitude)
- **`STrajectory6`**: 6D states (position + velocity, most common)

All variants share the same API and support the same operations.

## See Also

- [Trajectories Overview](index.md) - Trait hierarchy and implementation guide
- [DTrajectory](dtrajectory.md) - Dynamic-dimension trajectory for variable sizes
- [OrbitTrajectory](orbit_trajectory.md) - Orbital trajectory with frame conversions
- [STrajectory6 API Reference](../../library_api/trajectories/strajectory6.md)
