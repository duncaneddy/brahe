# RTN Transformations

The RTN (Radial-Tangential-Normal) frame is an orbital reference frame that moves
with the satellite. It is commonly used for relative motion analysis and formation
flying applications.

The RTN frame is defined as:

- **R (Radial)**: Points from the Earth's center to the satellite's position
- **T (Tangential)**: Along-track direction, perpendicular to R in the orbital plane
- **N (Normal)**: Cross-track direction, perpendicular to the orbital plane (angular momentum direction)

## Coordinate System Definition

The RTN frame is a **right-handed coordinate system** where:

- The R axis points from the center of the Earth to the satellite's position vector
- The N axis is parallel to the angular momentum vector (cross product of position and velocity)
- The T axis completes the right-handed system (it is the cross product of N and R)

This frame is useful for:

- Describing relative positions between satellites in close proximity
- Designing proximity operations and rendezvous maneuvers
- Expressing thrust directions for orbital maneuvers

## Rotation Matrices

Brahe provides functions to compute rotation matrices between the ECI (Earth-Centered
Inertial) frame and the RTN frame. These rotation matrices transform can transform
vectors between the two frames.

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/rtn_rotation_matrices.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/rtn_rotation_matrices.rs:4"
    ```

## State Transformations

For relative motion analysis between two satellites (often called "chief" and "deputy"),
Brahe provides functions to transform between absolute ECI states and relative RTN states.

### ECI to RTN (Absolute to Relative)

The `state_eci_to_rtn` function transforms the absolute states of two satellites from
the ECI frame to the relative state of the deputy with respect to the chief in the
RTN frame. This accounts for the rotating nature of the RTN frame.

The resulting relative state vector contains six components:

- Position: $[\rho_R, \rho_T, \rho_N]$ - relative position in RTN frame (m)
- Velocity: $[\dot{\rho}_R, \dot{\rho}_T, \dot{\rho}_N]$ - relative velocity in RTN frame (m/s)

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/state_eci_to_rtn.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/state_eci_to_rtn.rs:4"
    ```

### RTN to ECI (Relative to Absolute)

The `state_rtn_to_eci` function performs the inverse operation: it transforms the
relative state of a deputy satellite (in the RTN frame of the chief) back to the
absolute ECI state of the deputy. This is useful for propagating relative states
or computing deputy trajectories from relative motion plans.

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/state_rtn_to_eci.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/state_rtn_to_eci.rs:4"
    ```
