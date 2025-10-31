# DTrajectory

`DTrajectory` is a dynamically sized trajectory container that stores time-series state data with runtime-determined dimensions. Unlike static trajectory types, `DTrajectory` allows you to specify the state vector dimension at creation time, making it ideal for applications where the dimension varies or is not known at compile time.

Use `DTrajectory` when:

- State dimension is determined at runtime
- You need flexibility to work with different dimensions in the same codebase
- State vectors are non-standard (not 3D or 6D)
- Flexibility is prioritized over maximum performance

For fixed-dimension orbital mechanics applications, consider using [`STrajectory6`](strajectory6.md) or [`OrbitTrajectory`](orbit_trajectory.md) instead for better performance.

## Initialization

### Empty Trajectory

Create an empty trajectory by specifying the state dimension. The default dimension is 6 (suitable for position + velocity states):

=== "Python"

    ``` python
    --8<-- "./examples/trajectories/dtrajectory/dtrajectory_empty.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/trajectories/dtrajectory/dtrajectory_empty.rs:4"
    ```

### From Existing Data

Create a trajectory from existing epochs and states:

=== "Python"

    ``` python
    --8<-- "./examples/trajectories/dtrajectory/dtrajectory_from_data.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/trajectories/dtrajectory/dtrajectory_from_data.rs:4"
    ```

## Adding and Accessing States

### Adding States

Add states to a trajectory one at a time:

=== "Python"

    ``` python
    --8<-- "./examples/trajectories/dtrajectory/dtrajectory_add.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/trajectories/dtrajectory/dtrajectory_add.rs:4"
    ```

### Accessing by Index

Retrieve states and epochs by their index:

=== "Python"

    ``` python
    --8<-- "./examples/trajectories/dtrajectory/dtrajectory_access_index.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/trajectories/dtrajectory/dtrajectory_access_index.rs:4"
    ```

### Accessing by Epoch

Get states at or near specific epochs:

=== "Python"

    ``` python
    --8<-- "./examples/trajectories/dtrajectory/dtrajectory_access_epoch.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/trajectories/dtrajectory/dtrajectory_access_epoch.rs:4"
    ```

## Querying Trajectory Properties

### Time Span and Bounds

Query the temporal extent of a trajectory:

=== "Python"

    ``` python
    --8<-- "./examples/trajectories/dtrajectory/dtrajectory_timespan.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/trajectories/dtrajectory/dtrajectory_timespan.rs:4"
    ```

## Interpolation

DTrajectory supports linear interpolation to estimate states at arbitrary epochs between stored data points:

=== "Python"

    ``` python
    --8<-- "./examples/trajectories/dtrajectory/dtrajectory_interpolate.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/trajectories/dtrajectory/dtrajectory_interpolate.rs:4"
    ```

## Memory Management

DTrajectory supports eviction policies to automatically manage memory in long-running applications:

### Maximum Size Policy

Keep only the N most recent states:

=== "Python"

    ``` python
    --8<-- "./examples/trajectories/dtrajectory/dtrajectory_evict_size.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/trajectories/dtrajectory/dtrajectory_evict_size.rs:4"
    ```

### Maximum Age Policy

Keep only states within a time window:

=== "Python"

    ``` python
    --8<-- "./examples/trajectories/dtrajectory/dtrajectory_evict_age.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/trajectories/dtrajectory/dtrajectory_evict_age.rs:4"
    ```

## Iteration

Trajectories can be iterated to process all epoch-state pairs:

=== "Python"

    ``` python
    --8<-- "./examples/trajectories/dtrajectory/dtrajectory_iterate.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/trajectories/dtrajectory/dtrajectory_iterate.rs:4"
    ```

## Matrix Export

Convert trajectory data to matrix format for analysis or export:

=== "Python"

    ``` python
    --8<-- "./examples/trajectories/dtrajectory/dtrajectory_matrix.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/trajectories/dtrajectory/dtrajectory_matrix.rs:4"
    ```

## See Also

- [Trajectories Overview](index.md) - Trait hierarchy and implementation guide
- [STrajectory6](strajectory6.md) - Fixed 6D trajectory for better performance
- [OrbitTrajectory](orbit_trajectory.md) - Orbital trajectory with frame conversions
- [DTrajectory API Reference](../../library_api/trajectories/dtrajectory.md)
