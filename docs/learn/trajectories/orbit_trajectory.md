# OrbitTrajectory

`OrbitTrajectory` is a specialized trajectory container for orbital mechanics that tracks reference frames (ECI/ECEF) and orbital representations (Cartesian/Keplerian). Unlike `DTrajectory` and `STrajectory6` which store frame-agnostic data, `OrbitTrajectory` understands orbital mechanics and enables automatic conversions between reference frames and representations.

Use `OrbitTrajectory` when:

- Working with orbital mechanics applications
- Need to convert between ECI and ECEF frames
- Need to convert between Cartesian and Keplerian representations
- Want frame/representation metadata tracked automatically
- Working with propagators that output orbital trajectories

`OrbitTrajectory` implements the `OrbitalTrajectory` trait in addition to `Trajectory` and `Interpolatable`, providing orbital-specific functionality on top of the standard trajectory interface.

## Initialization

### Empty Trajectory - Cartesian Representation 

For cartesian representation, the frame can be `ECI` or `ECEF`. The `AngleFormat` **must** be `None` for Cartesian representations

=== "Python"

    ``` python
    --8<-- "./examples/trajectories/orbit_trajectory/orbit_trajectory_empty_cartesian.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/trajectories/orbit_trajectory/orbit_trajectory_empty_cartesian.rs:4"
    ```

### Empty Trajectory - Keplerian Elements

To create an empty trajectory in Keplerian representation you **must** specify the frame as `ECI` and provide an `AngleFormat`.

=== "Python"

    ``` python
    --8<-- "./examples/trajectories/orbit_trajectory/orbit_trajectory_empty_keplerian.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/trajectories/orbit_trajectory/orbit_trajectory_empty_keplerian.rs:4"
    ```

### From Existing Data

You can also initialize an `OrbitTrajectory` from existing epoch and state data:

=== "Python"

    ``` python
    --8<-- "./examples/trajectories/orbit_trajectory/orbit_trajectory_from_data.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/trajectories/orbit_trajectory/orbit_trajectory_from_data.rs:4"
    ```

### From Propagator

The most common way to get an `OrbitTrajectory` from a propagator. All orbit propagators in Brahe have a `*.trajectory` attribute which is an `OrbitTrajectory`.

See the [Propagators](../orbit_propagation/index.md) section for more details on propagators.

=== "Python"

    ``` python
    --8<-- "./examples/trajectories/orbit_trajectory/orbit_trajectory_from_propagator.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/trajectories/orbit_trajectory/orbit_trajectory_from_propagator.rs:4"
    ```

## Frame Conversions

The key feature of `OrbitTrajectory` is automatic frame conversions of the trajectory data to different reference frames and representations. In particular, with a single method call you can convert between ECI and ECEF frames, and between Cartesian and Keplerian representations.

### Converting ECI to ECEF

Convert a trajectory from Earth-Centered Inertial (ECI) to Earth-Centered Earth-Fixed (ECEF):

=== "Python"

    ``` python
    --8<-- "./examples/trajectories/orbit_trajectory/orbit_trajectory_eci_to_ecef.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/trajectories/orbit_trajectory/orbit_trajectory_eci_to_ecef.rs:4"
    ```

### Converting ECEF to ECI

Convert from ECEF back to ECI:

=== "Python"

    ``` python
    --8<-- "./examples/trajectories/orbit_trajectory/orbit_trajectory_ecef_to_eci.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/trajectories/orbit_trajectory/orbit_trajectory_ecef_to_eci.rs:4"
    ```

### Round-Trip Frame Conversion

Convert from ECI to ECEF and back to verify consistency:

=== "Python"

    ``` python
    --8<-- "./examples/trajectories/orbit_trajectory/orbit_trajectory_roundtrip.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/trajectories/orbit_trajectory/orbit_trajectory_roundtrip.rs:4"
    ```

### Converting Cartesian to Keplerian

Convert from Cartesian position/velocity to Keplerian orbital elements:

=== "Python"

    ``` python
    --8<-- "./examples/trajectories/orbit_trajectory/orbit_trajectory_cart_to_kep.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/trajectories/orbit_trajectory/orbit_trajectory_cart_to_kep.rs:4"
    ```

### Converting with Different Angle Formats

Convert to Keplerian with different angle formats:

=== "Python"

    ``` python
    --8<-- "./examples/trajectories/orbit_trajectory/orbit_trajectory_angle_formats.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/trajectories/orbit_trajectory/orbit_trajectory_angle_formats.rs:4"
    ```

## Combined Frame and Representation Conversions

Every conversion method returns a new `OrbitTrajectory` instance, so you can chain conversions together if desired:

=== "Python"

    ``` python
    --8<-- "./examples/trajectories/orbit_trajectory/orbit_trajectory_combined.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/trajectories/orbit_trajectory/orbit_trajectory_combined.rs:4"
    ```

## Standard Trajectory Operations

`OrbitTrajectory` supports all standard trajectory operations since it implements the `Trajectory` and `Interpolatable` traits:

=== "Python"

    ``` python
    --8<-- "./examples/trajectories/orbit_trajectory/orbit_trajectory_operations.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/trajectories/orbit_trajectory/orbit_trajectory_operations.rs:4"
    ```

## Practical Workflow Example

A complete example showing propagation, frame conversion, and analysis:

=== "Python"

    ``` python
    --8<-- "./examples/trajectories/orbit_trajectory/orbit_trajectory_workflow.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/trajectories/orbit_trajectory/orbit_trajectory_workflow.rs:4"
    ```

## See Also

- [Trajectories Overview](index.md) - Trait hierarchy and implementation guide
- [DTrajectory](dtrajectory.md) - Dynamic-dimension trajectory
- [STrajectory6](strajectory6.md) - Static 6D trajectory
- [OrbitTrajectory API Reference](../../library_api/trajectories/orbit_trajectory.md)
