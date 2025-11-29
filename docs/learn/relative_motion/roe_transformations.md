# Relative Orbital Elements (ROE) Transformations

Relative Orbital Elements (ROE) provide a quasi-nonsingular mean description of the relative motion between two satellites in close proximity. ROE are particularly useful for formation flying and proximity operations, where maintaining specific relative geometries is important.

Unlike instantaneous Cartesian relative states (like RTN coordinates), ROE describe the relative orbit using orbital elements, meaning that only a single element, the relative longitude $d\lambda$, is quickly changing over time. This makes ROE ideal for long-term formation design and control.

## ROE Definition

The ROE vector contains six dimensionless or angular elements that are constructed from the classical orbital elements of the chief and deputy satellites:

$$
\begin{align*}
\delta a & = \frac{a_d - a_c}{a_c} \\
\delta \lambda & = (M_d + \omega_d) - (M_c + \omega_c) + (\Omega_d - \Omega_c) \cos i_c \\
\delta e_x & = e_d \cos \omega_d - e_c \cos \omega_c \\
\delta e_y & = e_d \sin \omega_d - e_c \sin \omega_c \\
\delta i_x & = i_d - i_c \\
\delta i_y & = (\Omega_d - \Omega_c) \sin i_c
\end{align*}
$$

The elements are:
- $\delta a$ - relative semi-major axis (dimensionless)
- $\delta \lambda$ - relative mean longitude (radians)
- $\delta e_x$, $\delta e_y$ - components of the relative eccentricity vector (dimensionless)
- $\delta i_x$, $\delta i_y$ - components of the relative inclination vector (radians)

## Key Properties

**Nonsingularity**: ROE remain well-defined for circular and near-circular orbits, unlike classical orbital elements which become singular as eccentricity approaches zero.

**Periodic Orbits**: Specific ROE configurations produce periodic or quasi-periodic relative orbits:

- Setting $\delta a = 0$ prevents along-track drift
- The eccentricity vector components ($\delta e_x$, $\delta e_y$) control in-plane motion
- The inclination vector components ($\delta i_x$, $\delta i_y$) control cross-track motion

## Converting Orbital Elements to ROE

The `state_oe_to_roe` function converts the classical orbital elements of a chief and deputy satellite into ROE. This is useful when you have two satellite orbits and want to analyze their relative motion characteristics.

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/state_oe_to_roe.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/state_oe_to_roe.rs:4"
    ```

## Converting ROE to Deputy Orbital Elements

The `state_roe_to_oe` function performs the inverse operation: given the chief's orbital elements and the desired ROE, it computes the deputy's orbital elements. This is essential for:

- Initializing formation flying missions with desired relative geometries
- Retargeting maneuvers to achieve new relative configurations
- Propagating relative orbits using element-based propagators

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/state_roe_to_oe.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/state_roe_to_oe.rs:4"
    ```

## Direct ECI State to ROE Conversion

In many practical applications, satellite states are available as Cartesian ECI vectors rather than orbital elements. The `state_eci_to_roe` function provides a convenient way to compute ROE directly from the ECI states of the chief and deputy satellites, internally handling the conversion to orbital elements.

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/state_eci_to_roe.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/state_eci_to_roe.rs:4"
    ```

## Converting ROE to Deputy ECI State

The inverse operation, `state_roe_to_eci`, computes the deputy satellite's ECI state from the chief's ECI state and the ROE.

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/state_roe_to_eci.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/state_roe_to_eci.rs:4"
    ```

## References

1. [Sullivan, J. (2020). "Nonlinear Angles-Only Orbit Estimation for Autonomous Distributed Space Systems"](https://searchworks.stanford.edu/view/13680835)
