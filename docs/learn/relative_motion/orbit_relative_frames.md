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

Source: SANA Orbit-Relative Reference Frames registry (<https://sanaregistry.org/r/orbit_relative_reference_frames>) and CCSDS 500.0-G-4, *Navigation Data—Definitions and Conventions*, Section 4.3.7.

## Variants

Every frame exists as a rotating frame, which carries the orbital angular velocity, and as an inertial snapshot, whose rate is zero. The `state_*` relative-state functions always use the rotating transport term. A covariance transformation selects the variant through the rate it applies, as described under [Covariance](#covariance), and the frame graph selects it through the `OrbitRelativeFrameVariant` of the `ReferenceFrame`.

## Conventions

LVLH has two incompatible definitions in the literature. Vallado and STK use the name for the RTN axes. CCSDS, SANA, and this library put Z toward nadir and Y opposite the orbit normal, so that X is along-track for a circular orbit. The two are related by $X_\mathrm{LVLH} = T$, $Y_\mathrm{LVLH} = -N$, $Z_\mathrm{LVLH} = -R$.

Frame rates are exact under two-body motion and are the rates of the osculating frame otherwise. The RTN and LVLH rates depend only on the state. The NTW and TNW rates need the central body's gravitational parameter, so their `omega_` and `state_` functions have an Earth form and a `_for_body` form taking `gm`; frames added later that share this rate take the same form. The frame graph uses the declared center's value.

## Covariance

A covariance transforms into any of these frames with the 6x6 Jacobian of the frame's state map. With $R$ the rotation from the inertial axes into the frame and $\omega$ the frame's angular velocity expressed in the frame, a state relative to the frame origin maps as $\rho = R\,\Delta\mathbf{r}$ and $\dot{\rho} = R\,\Delta\mathbf{v} - \omega \times \rho$, so

$$
J = \begin{bmatrix} R & 0 \\ -[\omega]_\times R & R \end{bmatrix}, \qquad P_\mathrm{frame} = J \, P_\mathrm{inertial} \, J^T
$$

`jacobian_inertial_to_rotating` builds $J$ from a frame's `rotation_eci_to_*` and `omega_*` outputs, `jacobian_rotating_to_inertial` builds its inverse, and `rotate_covariance` applies either one. The rotating variant uses the frame's rate. The inertial snapshot uses $\omega = 0$, which reduces $J$ to $\mathrm{blockdiag}(R, R)$. RTN also provides `covariance_eci_to_rtn` and `covariance_rtn_to_eci`, which take the variant directly. For an object registered with the frame graph, `covariance_frame_to_frame` performs the same transformation from the object's registered state and agrees with $J$ to rounding.

$J$ holds the frame origin fixed, so it applies to a covariance referenced to the origin, such as a satellite's own covariance expressed in its own local frame. A deputy's relative state also depends on the chief's state through the frame axes, by an amount that grows with separation, so a relative-state covariance that includes chief uncertainty requires the Jacobian of `state_eci_to_*` with respect to both states.

The example transforms a covariance into rotating and snapshot LVLH, back to ECI, and through the frame graph.

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/orbit_relative_covariance.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/orbit_relative_covariance.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/relative_motion/orbit_relative_covariance.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/relative_motion/orbit_relative_covariance.rs.txt"
        ```

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

### See Also

- [RTN Transformations](rtn_transformations.md)
- [Frame Graph](../frames/frame_graph.md)
- [LVLH Transformations API Reference](../../library_api/relative_motion/lvlh_transformations.md)
- [NTW Transformations API Reference](../../library_api/relative_motion/ntw_transformations.md)
- [TNW Transformations API Reference](../../library_api/relative_motion/tnw_transformations.md)
