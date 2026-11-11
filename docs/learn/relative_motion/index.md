# Relative Motion

The relative motion module transforms states and covariances between an inertial frame and frames attached to an orbiting object, and converts between Cartesian states and relative orbital elements.

Every relative-motion function accepts batches: passing `(n, 6)` arrays for the chief and deputy transforms each pair, and a single chief (or single deputy) broadcasts across the batch. The `axis` keyword (default `-1`) names the dimension holding the vector components, following the rules described in [Vectorized Transformations](../frames/vectorized.md).

## Orbit-Relative Frames

An orbit-relative frame has its origin at an object and its axes built from that object's position and velocity, so the axes turn as the object moves along its orbit. Expressing a second object's state, a thrust direction, or a covariance in such a frame separates it into components tied to the orbit geometry, such as radial, along-track, and cross-track.

Each frame provides the same functions. `rotation_eci_to_<frame>` and `rotation_<frame>_to_eci` return the rotation matrix between the inertial axes and the frame, `omega_<frame>` returns the frame's angular velocity, and `state_eci_to_<frame>` and `state_<frame>_to_eci` convert between the inertial states of a chief and a deputy and the deputy's state relative to the chief. The [frame graph](../frames/frame_graph.md) evaluates the same frames for registered objects through `ReferenceFrame`.

The table gives the axes of each frame, using $\hat{r}$ for the unit position, $\hat{v}$ for the unit velocity, $\hat{h}$ for the unit orbital angular momentum $\mathbf{r} \times \mathbf{v}$, $\hat{e}$ for the eccentricity-vector direction, and $\hat{n}$ for the ascending-node direction.

| Frame | X | Y | Z | Rate about |
|---|---|---|---|---|
| [RTN](rtn_frame.md) (also RSW, QSW, RIC) | $\hat{r}$ | $\hat{h} \times \hat{r}$ | $\hat{h}$ | $+Z$ |
| [LVLH](lvlh_frame.md) | $\hat{h} \times \hat{r}$ | $-\hat{h}$ | $-\hat{r}$ | $-Y$ |
| [NTW](ntw_frame.md) | $\hat{v} \times \hat{h}$ | $\hat{v}$ | $\hat{h}$ | $+Z$ |
| [TNW](tnw_frame.md) | $\hat{v}$ | $\hat{h} \times \hat{v}$ | $\hat{h}$ | $+Z$ |
| [VNC](vnc_frame.md) | $\hat{v}$ | $\hat{h}$ | $\hat{v} \times \hat{h}$ | $+Y$ |
| [PQW](pqw_frame.md) | $\hat{e}$ | $\hat{h} \times \hat{e}$ | $\hat{h}$ | inertial only |
| [EQW](eqw_frame.md) | $\hat{n}$ | $\hat{h} \times \hat{n}$ | $\hat{h}$ | inertial only |

!!! tip "CCSDS frame names"
    The frame definitions match the [SANA orbit-relative reference frame registry](https://sanaregistry.org/r/orbit_relative_reference_frames), which CCSDS orbit and attitude messages reference by name. RTN is the registry's RSW frame.

### Variants

Each frame has a rotating variant, which carries the frame's angular velocity, and an inertial snapshot, which takes the same axes at the evaluation epoch with zero rate. The `state_*` functions use the rotating variant. A covariance transformation selects the variant through the rate it applies (see [Covariance](#covariance)), and the frame graph selects it through the `OrbitRelativeFrameVariant` of the `ReferenceFrame`. PQW and EQW exist only as inertial snapshots, with no `omega_` function and no transport term in the `state_*` functions.

### Rates

Frame rates are exact under two-body motion and are the rates of the osculating frame otherwise. The RTN and LVLH rates depend only on the state. The NTW, TNW, and VNC rates also need the central body's gravitational parameter, so the `omega_` and `state_` functions have an Earth form and a `_for_body` form taking `gm`. The frame graph uses the value of the object's declared center.

### Covariance

A covariance transforms into any of these frames with the 6x6 Jacobian of the frame's state map. With $R$ the rotation from the inertial axes into the frame and $\omega$ the frame's angular velocity expressed in the frame, a state relative to the frame origin maps as $\rho = R\,\Delta\mathbf{r}$ and $\dot{\rho} = R\,\Delta\mathbf{v} - \omega \times \rho$, so

$$
J = \begin{bmatrix} R & 0 \\ -[\omega]_\times R & R \end{bmatrix}, \qquad P_\mathrm{frame} = J \, P_\mathrm{inertial} \, J^T
$$

`jacobian_inertial_to_rotating` builds $J$ from a frame's `rotation_eci_to_*` and `omega_*` outputs, `jacobian_rotating_to_inertial` builds its inverse, and `rotate_covariance` applies either one. RTN also provides `covariance_eci_to_rtn` and `covariance_rtn_to_eci`, which take the variant directly. For an object registered with the frame graph, `covariance_frame_to_frame` performs the same transformation from the object's registered state.

The inertial snapshot uses $\omega = 0$, which reduces $J$ to $\mathrm{blockdiag}(R, R)$. The rotating variant adds the $-[\omega]_\times R$ coupling between position and velocity.

$J$ holds the frame origin fixed, so it applies to a covariance referenced to the origin, such as a satellite's own covariance expressed in its own local frame. A deputy's relative state also depends on the chief's state through the frame axes, so a relative-state covariance that includes chief uncertainty requires the Jacobian of `state_eci_to_*` with respect to both states.

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

## See Also

- [RTN Frame](rtn_frame.md)
- [LVLH Frame](lvlh_frame.md)
- [NTW Frame](ntw_frame.md)
- [TNW Frame](tnw_frame.md)
- [VNC Frame](vnc_frame.md)
- [PQW Frame](pqw_frame.md)
- [EQW Frame](eqw_frame.md)
- [ROE Transformations](roe_transformations.md)
- [Frame Graph](../frames/frame_graph.md)
- [Relative Motion API Reference](../../library_api/relative_motion/index.md)
