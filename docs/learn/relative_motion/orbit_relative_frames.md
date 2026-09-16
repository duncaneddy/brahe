# Orbit-Relative Frames

Brahe implements the local orbital frames of the SANA orbit-relative reference frame registry, which CCSDS orbit and attitude messages use for `REF_FRAME` keywords. Each frame is built from an object's position and velocity (and, for some frames, additional inputs) and exists in two variants: a rotating frame, which carries the orbital angular velocity, and an inertial snapshot, which takes the same axes at the evaluation epoch and treats them as fixed. The functions on this page take Cartesian states in an inertial frame centered on the orbited body, except the SEZ functions, which take ECEF states of a site; the frame graph evaluates the same frames for registered objects through `ReferenceFrame`.

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
| EQW | $\hat{n}$ | $\hat{h} \times \hat{n}$ | $\hat{h}$ | inertial only |
| NSW | $-\hat{r}$ | $\hat{s}$ projected normal to X | $X \times Y$ | rate from basis derivatives |
| SEZ | south | east | geodetic up | rate relative to ECEF |

Source: SANA Orbit-Relative Reference Frames registry (<https://sanaregistry.org/r/orbit_relative_reference_frames>) and CCSDS 500.0-G-4, *Navigation Data—Definitions and Conventions*, Section 4.3.7.

## Variants

Every frame exists as a rotating frame, which carries the orbital angular velocity, and as an inertial snapshot, whose rate is zero. The `state_*` relative-state functions always use the rotating transport term. The `jacobian_*` and `covariance_*` functions take an `OrbitRelativeFrameVariant` selecting which. PQW and EQW are the exception: SANA registers them only as inertial snapshots, so they have no `omega_*` functions, their `jacobian_*` and `covariance_*` functions take no variant and are always block diagonal, and their `state_*` functions apply no transport term.

## Conventions

LVLH has two incompatible definitions in the literature. Vallado and STK use the name for the RTN axes. CCSDS, SANA, and this library put Z toward nadir and Y opposite the orbit normal, so that X is along-track for a circular orbit. The two are related by $X_\mathrm{LVLH} = T$, $Y_\mathrm{LVLH} = -N$, $Z_\mathrm{LVLH} = -R$.

Frame rates are exact under two-body motion and are the rates of the osculating frame otherwise. The RTN and LVLH rates depend only on the state. The NTW, TNW, and VNC rates need the central body's gravitational parameter, so their `omega_`, `jacobian_`, `covariance_`, and `state_` functions have an Earth form and a `_for_body` form taking `gm`. The frame graph uses the declared center's value. NSW is neither: its rate follows from the time derivatives of its axes and needs the Sun's velocity rather than the gravitational parameter.

PQW and EQW are registered by SANA only as inertial snapshots. On a circular orbit the periapsis direction is undefined and PQW takes P along the ascending node; on an equatorial orbit the node is undefined and P (and EQW's E) is taken along the inertial x axis projected into the orbit plane. These match the zero-angle conventions for the argument of periapsis and the right ascension of the ascending node.

Every NSW function takes the Sun's state as the argument directly after the spacecraft states: after `x_eci` for the rotation, rate, Jacobian, and covariance functions, and after the chief and deputy (or relative) states for `state_eci_to_nsw` and `state_nsw_to_eci`. Because X lies along the position vector, the frame is the same whether the Sun state is given relative to the center or relative to the spacecraft. When the Sun lies along the nadir line the projection used to build Y is degenerate and falls back to the along-track direction $\hat{h} \times \hat{r}$. In the frame graph the Sun state comes from the source selected with `set_frame_ephemeris_source`: `Auto`, the default, uses loaded SPICE kernels if any, otherwise the analytic Sun model for Earth-centered objects, otherwise the default DE kernel; `Analytic` uses the analytic Sun model and is Earth-centered only; `Kernel` always uses the SPICE registry.

SEZ is a topocentric horizon frame of a site, not an orbit-derived frame; it is in the SANA registry because ADM and TDM name it. Brahe builds it from a site's ECEF position on the WGS84 ellipsoid, so it is Earth-only. Its `omega_` functions give the rate relative to ECEF, which is zero for a fixed site; the frame graph adds Earth's rotation for the rate relative to inertial space.

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

## EQW

The equinoctial frame places E along the ascending node, W along the orbit normal, and Q = W × E, so a satellite's own position is $r[\cos u, \sin u, 0]$ with $u$ the argument of latitude. It needs no gravitational parameter. On an equatorial orbit the node is undefined and E is taken along the inertial x axis projected into the orbit plane. There is no rate: the frame is an inertial snapshot.

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

The nadir/Sun/normal frame places X toward nadir, Y as close to the Sun as possible while normal to X, and Z = X × Y. Its functions take the Sun's state relative to the same center as the spacecraft state; the direction to the Sun is measured from the spacecraft. The rate follows from the time derivatives of the axes and includes the Sun's motion when the Sun state carries a velocity. In the frame graph the Sun state comes from the source selected with `set_frame_ephemeris_source`.

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

The south/east/zenith frame is the topocentric horizon frame of a site: S points due south, E east, and Z along the geodetic vertical. Its functions take ECEF states of the site and of a target. The rate relative to ECEF is the transport rate $[-\dot\lambda \cos\varphi, -\dot\varphi, \dot\lambda \sin\varphi]$ from the site's longitude and geodetic latitude rates, zero for a fixed station. In the frame graph a station registered with an ITRF state resolves to its SEZ frame with Earth's rotation composed in.

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
