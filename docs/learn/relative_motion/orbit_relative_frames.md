# Orbit-Relative Frames

Brahe implements the local orbital frames of the SANA orbit-relative reference frame registry, which CCSDS orbit and attitude messages use for `REF_FRAME` keywords. Each frame is built from an object's position and velocity (and, for some frames, additional inputs) and exists in two variants: a rotating frame, which carries the orbital angular velocity, and an inertial snapshot, which takes the same axes at the evaluation epoch and treats them as fixed. The functions on this page take Cartesian states in an inertial frame centered on the orbited body; the frame graph evaluates the same frames for registered objects through `ReferenceFrame`.

Every function accepts batches: an `(n, 6)` array of states transforms each row, and the `axis` keyword names the component axis as described in [Vectorized Transformations](../frames/vectorized.md).

## Definitions

With $\hat{r}$ the unit position, $\hat{v}$ the unit velocity, $\hat{h}$ the unit orbital angular momentum $\mathbf{r} \times \mathbf{v}$, $\hat{n}$ the ascending-node direction, $\hat{e}$ the eccentricity-vector direction, and $\hat{s}$ the direction to the Sun. $\hat{v}$, $\hat{n}$, $\hat{e}$, and $\hat{s}$ are used by the frames documented below as they are added:

| Frame | X | Y | Z | Rate about |
|---|---|---|---|---|
| RTN (SANA RSW; also QSW, RIC) | $\hat{r}$ | $\hat{h} \times \hat{r}$ | $\hat{h}$ | $+Z$ |
| LVLH | $\hat{h} \times \hat{r}$ | $-\hat{h}$ | $-\hat{r}$ | $-Y$ |
| NTW | $\hat{v} \times \hat{h}$ | $\hat{v}$ | $\hat{h}$ | $+Z$ |
| TNW | $\hat{v}$ | $\hat{h} \times \hat{v}$ | $\hat{h}$ | $+Z$ |
| VNC | $\hat{v}$ | $\hat{h}$ | $\hat{v} \times \hat{h}$ | $+Y$ |
| PQW | $\hat{e}$ | $\hat{h} \times \hat{e}$ | $\hat{h}$ | inertial only |

Source: SANA Orbit-Relative Reference Frames registry (<https://sanaregistry.org/r/orbit_relative_reference_frames>) and CCSDS 500.0-G-4, *Navigation Data—Definitions and Conventions*, Section 4.3.7.

## Variants

Every frame exists as a rotating frame, which carries the orbital angular velocity, and as an inertial snapshot, whose rate is zero. The `state_*` relative-state functions always use the rotating transport term. The `jacobian_*` and `covariance_*` functions take an `OrbitRelativeFrameVariant` selecting which. PQW and EQW are the exception: SANA registers them only as inertial snapshots, so they have no `omega_*` functions, their `jacobian_*` and `covariance_*` functions take no variant and are always block diagonal, and their `state_*` functions apply no transport term.

## Conventions

LVLH has two incompatible definitions in the literature. Vallado and STK use the name for the RTN axes. CCSDS, SANA, and this library put Z toward nadir and Y opposite the orbit normal, so that X is along-track for a circular orbit. The two are related by $X_\mathrm{LVLH} = T$, $Y_\mathrm{LVLH} = -N$, $Z_\mathrm{LVLH} = -R$.

Frame rates are exact under two-body motion and are the rates of the osculating frame otherwise. The RTN and LVLH rates depend only on the state. The NTW, TNW, and VNC rates need the central body's gravitational parameter, so their `omega_`, `jacobian_`, `covariance_`, and `state_` functions have an Earth form and a `_for_body` form taking `gm`. The frame graph uses the declared center's value.

PQW and EQW are registered by SANA only as inertial snapshots. On a circular orbit the periapsis direction is undefined and PQW takes P along the ascending node; on an equatorial orbit the node is undefined and P (and EQW's E) is taken along the inertial x axis projected into the orbit plane. These match the zero-angle conventions for the argument of periapsis and the right ascension of the ascending node.

## LVLH

The Local-Vertical Local-Horizontal frame places Z along nadir, Y opposite the orbit normal, and X completing the right-handed set. Its angular velocity is the true-anomaly rate about $-Y$. The example builds the frame, checks it against RTN, transforms a deputy's relative state both ways, and evaluates the same frame through the frame graph.

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/lvlh_frame.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/lvlh_frame.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/relative_motion/lvlh_frame.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/relative_motion/lvlh_frame.rs.txt"
        ```

## NTW

The NTW frame places Y along the velocity, Z along the orbit normal, and X = Y × Z in the orbit plane, outward. It coincides with RTN on a circular orbit and differs by the flight-path angle otherwise. Because its axes follow the velocity direction, its rate is the two-body turn rate of the velocity, $\mu |h| / (r^3 v^2)$ about Z, which requires the central body's gravitational parameter.

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/ntw_frame.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/ntw_frame.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/relative_motion/ntw_frame.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/relative_motion/ntw_frame.rs.txt"
        ```

## TNW

The TNW frame places X along the velocity, Z along the orbit normal, and Y = Z × X in the orbit plane pointing inward, toward nadir on a circular orbit. It is the NTW frame with its in-plane axes reordered: $X_\mathrm{TNW} = Y_\mathrm{NTW}$, $Y_\mathrm{TNW} = -X_\mathrm{NTW}$. Its rate is the velocity-direction turn rate about Z.

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/tnw_frame.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/tnw_frame.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/relative_motion/tnw_frame.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/relative_motion/tnw_frame.rs.txt"
        ```

## VNC

The VNC frame places X along the velocity, Y along the orbit normal, and Z = X × Y, the co-normal, in the orbit plane pointing outward. STK calls this frame VNB. It shares its axes with NTW and TNW ($X_\mathrm{VNC} = X_\mathrm{TNW} = Y_\mathrm{NTW}$, $Y_\mathrm{VNC} = Z_\mathrm{TNW}$, $Z_\mathrm{VNC} = X_\mathrm{NTW}$), so its rate is the velocity-direction turn rate, here about Y.

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/vnc_frame.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/vnc_frame.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/relative_motion/vnc_frame.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/relative_motion/vnc_frame.rs.txt"
        ```

## PQW

The perifocal frame places P toward periapsis, W along the orbit normal, and Q = W × P, so a satellite's own position is $r[\cos f, \sin f, 0]$ with $f$ the true anomaly. The periapsis direction is the eccentricity vector, which needs the central body's gravitational parameter, so the rotation has an Earth form and a `_for_body` form. There is no rate: the frame is an inertial snapshot.

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/pqw_frame.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/pqw_frame.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/relative_motion/pqw_frame.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/relative_motion/pqw_frame.rs.txt"
        ```

### See Also

- [RTN Transformations](rtn_transformations.md)
- [Frame Graph](../frames/frame_graph.md)
- [LVLH Transformations API Reference](../../library_api/relative_motion/lvlh_transformations.md)
- [NTW Transformations API Reference](../../library_api/relative_motion/ntw_transformations.md)
- [TNW Transformations API Reference](../../library_api/relative_motion/tnw_transformations.md)
- [VNC Transformations API Reference](../../library_api/relative_motion/vnc_transformations.md)
- [PQW Transformations API Reference](../../library_api/relative_motion/pqw_transformations.md)
