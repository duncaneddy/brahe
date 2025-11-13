# RTN Transformations

The RTN (Radial-Tangential-Normal) frame is an orbital reference frame that moves
with the satellite. It is commonly used for relative motion analysis and formation
flying applications.

The RTN frame is defined as:

- **R (Radial)**: Points from the Earth's center to the satellite's position
- **T (Tangential)**: Along-track direction, perpendicular to R in the orbital plane
- **N (Normal)**: Cross-track direction, perpendicular to the orbital plane (angular momentum direction)

### Coordinate System Definition

The RTN frame is a **right-handed coordinate system** where:

- The R axis points from the center of the Earth to the satellite's position vector
- The N axis is parallel to the angular momentum vector (cross product of position and velocity)
- The T axis completes the right-handed system (it is the cross product of N and R)

This frame is useful for:

- Describing relative positions between satellites close proximity
- Designing proximity operations and rendezvous maneuvers
- Expressing thrust directions for orbital maneuvers

### Transformations

Brahe provides functions to compute rotation matrices between the ECI (Earth-Centered
Inertial) frame and the RTN frame:

- `rotation_rtn_to_eci(x_eci)`: Computes the rotation matrix from RTN to ECI
- `rotation_eci_to_rtn(x_eci)`: Computes the rotation matrix from ECI to RTN

Both functions take the satellite's 6D state vector (position and velocity) in the
ECI frame as input.