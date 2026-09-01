# Frame Graph

`ReferenceFrame` is the top-level frame identity in Brahe. It extends `CelestialFrame` &mdash; the router covered in [Reference Frame Router](frame_transformations.md) &mdash; to frames that are scoped to a specific object: a spacecraft's orbit-relative frame (`RTN`, `LVLH`, ...) or a body/sensor/actuator frame (`SC_BODY`, `CSS_1`, ...). `rotation_frame_to_frame`, `position_frame_to_frame`, and `state_frame_to_frame` (and their batch forms) accept either a `CelestialFrame` or a `ReferenceFrame` for `from`/`to`, so a call site that only ever uses `CelestialFrame` needs no changes to keep working.

A `ReferenceFrame` is one of three variants:

- **Celestial**: any `CelestialFrame` (`GCRF`, `ITRF`, `LFPA`, ...). Evaluable analytically from an epoch alone, exactly as in the router.
- **Orbit-relative**: a local orbital frame &mdash; `RTN`, `LVLH`, `NTW`, `TNW`, `PQW`, `EQW`, `SEZ`, `VNC`, or `NSW` &mdash; of one object, either rotating with the orbit or frozen as an inertial snapshot at each evaluation epoch.
- **Body**: an object-local frame with no global transformation &mdash; a spacecraft body frame, a sensor, an actuator, or an instrument.

Orbit-relative and body frames carry an object identity, a plain string (e.g. `"LRO"`, `"2024-123A"`) kept separate from NAIF or NORAD IDs. Constructing one through a family method, `ReferenceFrame.RTN("SC")` or `ReferenceFrame.CSS("SC", "1")`, binds it to that object directly.

## Bound vs. Unbound

A frame is **bound** when it can be evaluated: every `CelestialFrame` is bound by construction, and an orbit-relative or body frame is bound once it carries an object. Constructing one with no object &mdash; `ReferenceFrame.body(None, ...)`/`ReferenceFrame.orbit_relative(..., object=None)` in Python, or converting a bare `BodyFrame`/`OrbitRelativeFrame` in Rust &mdash; gives the unbound form instead: a pure label, useful for parsing a data file's frame column before an object identity is known. `is_bound()` and `object()` report which case a given `ReferenceFrame` is in.

Calling any transform on an unbound frame raises immediately, naming the frame and the constructor that binds it, rather than failing later inside a registry lookup.

## Registering Objects

An orbit-relative or body frame's origin is the registered state of the object it is bound to. `register_object(name, provider, frame)` accepts either a callable `Epoch -> state` (position and velocity, meters and m/s, expressed in `frame`) or an `OrbitTrajectory`, and stores it under `name` in a single global object registry, keyed by object identity rather than NAIF ID: kernel data only enters this registry through an explicit provider such as `register_object_from_naif`, never implicitly.

Parsing a CCSDS OEM is the common case, and `OEM.register_for(name)` is a one-liner for it: it converts the ephemeris to a trajectory and registers it under `name`, in the frame the OEM itself declares (`GCRF`, `ITRF`, or `EME2000`).

=== "Python"

    ``` python
    --8<-- "./examples/frames/register_object_oem.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/register_object_oem.rs:5"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/frames/register_object_oem.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/frames/register_object_oem.rs.txt"
        ```

## Registering Orientation Chains

A body frame's orientation is not derived from any model; it is registered explicitly with `register_frame(frame, parent, provider)`. `parent` must itself resolve to a celestial root: either it is a `CelestialFrame` directly, or it is a body frame that is already registered and whose own parent chain terminates at one. Re-registering an existing frame replaces its entry, and the replacement's parent chain is revalidated, so a change that would cycle back through the frame itself is rejected.

`provider` supplies the rotation and, optionally, the angular velocity of `frame` relative to `parent`, expressed in `frame`. Two kinds of provider are available today:

- A **constant attitude** &mdash; a `Quaternion`, `RotationMatrix`, `EulerAngle`, or `EulerAxis` &mdash; for a sensor mounted at a fixed orientation. Its angular velocity relative to its parent is zero by construction.
- A **callable**, `Epoch -> rotation matrix`, optionally paired with a second callable returning the angular velocity. This covers time-varying orientations such as a slewing sensor or an articulated appendage.

An orientation chain driven by an attitude ephemeris (AEM), analogous to `OEM.register_for`, ships in a later release; a time-varying orientation is registered as a callable in the meantime.

## Worked Example: A Sun Vector in a Sensor Frame

The example below registers a spacecraft as an object, builds a two-link orientation chain (`SC_BODY` off `GCRF`, then a coarse sun sensor `CSS_1` off `SC_BODY`), and routes the Sun's GCRF position through both links with `position_frame_to_frame`. Body frames share their object's origin exactly &mdash; there is no lever arm between an object's center and the sensor frames mounted on it &mdash; which the example confirms by routing the spacecraft's own position into `CSS_1` and getting the origin back. Querying an unregistered link (`CSS_2`, never registered) raises an error naming the missing frame and the `register_frame` call that would supply it.

=== "Python"

    ``` python
    --8<-- "./examples/frames/body_frame_chain.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/body_frame_chain.rs:5"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/frames/body_frame_chain.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/frames/body_frame_chain.rs.txt"
        ```

## The Rates Rule and `with_numerical_rates`

`position_frame_to_frame` needs no angular-velocity data: it only rotates and re-centers. `state_frame_to_frame` is different &mdash; converting a velocity requires the transport term $\omega \times r$ at every non-celestial link in the chain, so every link's provider must supply an angular velocity. A rotation-only provider (a callable given no `omega`) reports `None` for it, and a state transform through that link raises rather than silently dropping the transport term.

`with_numerical_rates` closes that gap for a provider that has no angular-velocity data of its own: it wraps the provider so a missing rate is derived by central-differencing the rotation matrix over $\pm \text{step}/2$ seconds, using $[\omega]_\times = -\dot{R} R^\mathsf{T}$. In Python, this is the `numerical_rates_step` argument to `register_frame`, and it applies only to a callable provider; a constant attitude already has an exact zero rate and does not need it.

=== "Python"

    ``` python
    --8<-- "./examples/frames/numerical_rates.py:10"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/numerical_rates.rs:6"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/frames/numerical_rates.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/frames/numerical_rates.rs.txt"
        ```

---

## See Also

- [ReferenceFrame / BodyFrame API Reference](../../library_api/frames/frame.md)
- [Reference Frame Router](frame_transformations.md) - `CelestialFrame` and the frame-to-frame router functions
- [CCSDS OEM](../ccsds/oem.md) - Parsing and writing OEM ephemeris files
