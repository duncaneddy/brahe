# Orbit-Relative Frames

Brahe implements the local orbital frames of the SANA orbit-relative reference frame registry, which CCSDS orbit and attitude messages name in `REF_FRAME` keywords. The functions on this page build each frame from a Cartesian state in an inertial frame centered on the orbited body; SEZ and ENZ instead take the ECEF state of a site. The [frame graph](../frames/frame_graph.md) evaluates the same frames for registered objects through `ReferenceFrame`.

Every function accepts batches: an `(n, 6)` array of states transforms each row, and the `axis` keyword names the component axis as described in [Vectorized Transformations](../frames/vectorized.md).

## Definitions

The table uses $\hat{r}$ for the unit position, $\hat{v}$ for the unit velocity, $\hat{h}$ for the unit orbital angular momentum $\mathbf{r} \times \mathbf{v}$, $\hat{e}$ for the eccentricity-vector direction, $\hat{n}$ for the ascending-node direction, and $\hat{s}$ for the direction to the Sun.

| Frame | X | Y | Z | Rate about |
|---|---|---|---|---|
| RTN (SANA RSW; also QSW, RIC) | $\hat{r}$ | $\hat{h} \times \hat{r}$ | $\hat{h}$ | $+Z$ |
| LVLH | $\hat{h} \times \hat{r}$ | $-\hat{h}$ | $-\hat{r}$ | $-Y$ |
| NTW | $\hat{v} \times \hat{h}$ | $\hat{v}$ | $\hat{h}$ | $+Z$ |
| TNW | $\hat{v}$ | $\hat{h} \times \hat{v}$ | $\hat{h}$ | $+Z$ |
| VNC | $\hat{v}$ | $\hat{h}$ | $\hat{v} \times \hat{h}$ | $+Y$ |
| PQW | $\hat{e}$ | $\hat{h} \times \hat{e}$ | $\hat{h}$ | inertial only |
| EQW | $\hat{n}$ | $\hat{h} \times \hat{n}$ | $\hat{h}$ | inertial only |
| NSW | $-\hat{r}$ | $\hat{s}$ projected normal to X | $X \times Y$ | rate from basis derivatives |
| SEZ | south | east | geodetic up | rate relative to ECEF |
| ENZ | east | north | geodetic up | rate relative to ECEF |

Source: SANA Orbit-Relative Reference Frames registry (<https://sanaregistry.org/r/orbit_relative_reference_frames>) and CCSDS 500.0-G-4, *Navigation Data—Definitions and Conventions*, Section 4.3.7.

## Variants

Each frame has a rotating variant, which carries the frame's angular velocity, and an inertial snapshot, which takes the same axes at the evaluation epoch with zero rate. The `state_*` functions use the rotating variant. A covariance transformation selects the variant through the rate it applies (see [Covariance](#covariance)), and the frame graph selects it through the `OrbitRelativeFrameVariant` of the `ReferenceFrame`. PQW and EQW exist only as inertial snapshots, with no `omega_` function and no transport term in the `state_*` functions.

## Rates

Frame rates are exact under two-body motion and are the rates of the osculating frame otherwise. The RTN and LVLH rates depend only on the state. The NTW, TNW, and VNC rates also need the central body's gravitational parameter, so the `omega_` and `state_` functions have an Earth form and a `_for_body` form taking `gm`. The frame graph uses the value of the object's declared center.

## Covariance

A covariance transforms into any of these frames with the 6x6 Jacobian of the frame's state map. With $R$ the rotation from the inertial axes into the frame and $\omega$ the frame's angular velocity expressed in the frame, a state relative to the frame origin maps as $\rho = R\,\Delta\mathbf{r}$ and $\dot{\rho} = R\,\Delta\mathbf{v} - \omega \times \rho$, so

$$
J = \begin{bmatrix} R & 0 \\ -[\omega]_\times R & R \end{bmatrix}, \qquad P_\mathrm{frame} = J \, P_\mathrm{inertial} \, J^T
$$

`jacobian_inertial_to_rotating` builds $J$ from a frame's `rotation_eci_to_*` and `omega_*` outputs, `jacobian_rotating_to_inertial` builds its inverse, and `rotate_covariance` applies either one. For SEZ and ENZ, `rotation_ecef_to_*` and `omega_*` give the Jacobian between ECEF and the topocentric frame in the same way. RTN also provides `covariance_eci_to_rtn` and `covariance_rtn_to_eci`, which take the variant directly. For an object registered with the frame graph, `covariance_frame_to_frame` performs the same transformation from the object's registered state.

The inertial snapshot uses $\omega = 0$, which reduces $J$ to $\mathrm{blockdiag}(R, R)$, the block-diagonal covariance transformation of Vallado's AAS 03-526 and the CCSDS CDM convention for RTN covariances. The rotating variant adds the $-[\omega]_\times R$ coupling between position and velocity.

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

## LVLH

LVLH places Z along nadir, Y opposite the orbit normal, and X completing the right-handed set. Its rate is the true-anomaly rate about $-Y$. Vallado and STK use the name LVLH for the RTN axes; brahe follows CCSDS and SANA, so $X_\mathrm{LVLH} = T$, $Y_\mathrm{LVLH} = -N$, and $Z_\mathrm{LVLH} = -R$.

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

NTW places Y along the velocity, Z along the orbit normal, and $X = Y \times Z$ in the orbit plane, pointing outward. It coincides with RTN on a circular orbit and differs from it by the flight-path angle otherwise. Its rate is the two-body turn rate of the velocity direction, $\mu |h| / (r^3 v^2)$ about Z.

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

TNW is NTW with its in-plane axes reordered, $X_\mathrm{TNW} = Y_\mathrm{NTW}$ and $Y_\mathrm{TNW} = -X_\mathrm{NTW}$, so X lies along the velocity and Y points inward. Its rate is the NTW rate about Z.

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

VNC places X along the velocity, Y along the orbit normal, and the co-normal $Z = X \times Y$ in the orbit plane, pointing outward, so $Z_\mathrm{VNC} = X_\mathrm{NTW}$. STK calls this frame VNB. Its rate is the NTW rate about Y.

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

The perifocal frame places P toward periapsis, W along the orbit normal, and $Q = W \times P$, so a satellite's own position is $r[\cos f, \sin f, 0]$ with $f$ the true anomaly. The periapsis direction comes from the eccentricity vector, which needs the central body's gravitational parameter, so the rotation has an Earth form and a `_for_body` form. On a circular orbit P is taken along the ascending node, and on a circular equatorial orbit along the inertial x axis projected into the orbit plane.

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

## EQW

The equinoctial frame places E along the ascending node, W along the orbit normal, and $Q = W \times E$, so a satellite's own position is $r[\cos u, \sin u, 0]$ with $u$ the argument of latitude. On an equatorial orbit E is taken along the inertial x axis projected into the orbit plane.

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/eqw_frame.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/eqw_frame.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/relative_motion/eqw_frame.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/relative_motion/eqw_frame.rs.txt"
        ```

## NSW

NSW places X toward nadir, Y toward the Sun as nearly as possible while normal to X, and $Z = X \times Y$. Each function takes the Sun's state, relative to the same center as the spacecraft state, as the argument after the spacecraft states. The rate follows from the time derivatives of the axes and includes the Sun's velocity. When the Sun lies on the nadir line, Y falls back to the along-track direction $\hat{h} \times \hat{r}$. The frame graph takes the Sun's state from the SPICE registry.

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/nsw_frame.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/nsw_frame.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/relative_motion/nsw_frame.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/relative_motion/nsw_frame.rs.txt"
        ```

## SEZ

SEZ is the topocentric horizon frame of a site on the WGS84 ellipsoid: S points south, E east, and Z along the geodetic vertical. Its functions take the ECEF states of the site and of a target, so the frame is Earth-only. `omega_sez` is the rate relative to ECEF, $[-\dot\lambda \cos\varphi, -\dot\varphi, \dot\lambda \sin\varphi]$ from the site's longitude and geodetic latitude rates, which is zero for a fixed site. The frame graph adds Earth's rotation.

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/sez_frame.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/sez_frame.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/relative_motion/sez_frame.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/relative_motion/sez_frame.rs.txt"
        ```

## ENZ

ENZ is SEZ with its horizontal axes reordered, $E_\mathrm{ENZ} = E_\mathrm{SEZ}$ and $N_\mathrm{ENZ} = -S_\mathrm{SEZ}$. It is not in the SANA registry. Brahe's CCSDS reader and writer use the tokens `ENZ_ROTATING` and `ENZ_INERTIAL` for it, and the reader also accepts `ENU_*`; other tools may not recognize these tokens.

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/enz_frame.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/enz_frame.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/relative_motion/enz_frame.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/relative_motion/enz_frame.rs.txt"
        ```

### See Also

- [RTN Transformations](rtn_transformations.md)
- [Frame Graph](../frames/frame_graph.md)
- [LVLH Transformations API Reference](../../library_api/relative_motion/lvlh_transformations.md)
- [NTW Transformations API Reference](../../library_api/relative_motion/ntw_transformations.md)
- [TNW Transformations API Reference](../../library_api/relative_motion/tnw_transformations.md)
- [VNC Transformations API Reference](../../library_api/relative_motion/vnc_transformations.md)
- [PQW Transformations API Reference](../../library_api/relative_motion/pqw_transformations.md)
- [EQW Transformations API Reference](../../library_api/relative_motion/eqw_transformations.md)
- [NSW Transformations API Reference](../../library_api/relative_motion/nsw_transformations.md)
- [SEZ Transformations API Reference](../../library_api/relative_motion/sez_transformations.md)
- [ENZ Transformations API Reference](../../library_api/relative_motion/enz_transformations.md)
