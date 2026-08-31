# AttitudeTrajectory

`AttitudeTrajectory` is a chronologically sorted collection of attitude samples relating two [`AttitudeFrame`](../attitude_representations/index.md) endpoints. Each sample is an `AttitudeState`: a unit quaternion, plus an optional body angular velocity. Use `AttitudeTrajectory` when you have or want to produce a time-ordered attitude history — converting from an AEM, storing a propagated or ground-solved attitude solution, or interpolating attitude for pointing analysis.

`AttitudeTrajectory` implements the [`Trajectory`](trajectory.md) trait, so the standard trajectory operations — `add`, `get`, `len`, `start_epoch`/`end_epoch`, eviction policies — all apply. It does not implement `InterpolatableTrajectory`: that trait's default interpolation methods require the state type to support scalar multiplication and addition, and unit quaternions are not closed under either operation. Interpolation is instead an inherent method (`interpolate`) and the [`AttitudeProvider`](#attitudeprovider) trait, both quaternion-aware.

## Canonical State and Rate Uniformity

Every quaternion stored in an `AttitudeTrajectory` represents the attitude of `frame_b` relative to `frame_a` — the same A$\to$B convention used throughout brahe's CCSDS attitude support. When a state carries an angular velocity, it is the angular velocity of frame B relative to frame A, expressed in frame B, in rad/s.

A trajectory's states must uniformly carry angular velocity or uniformly omit it. `add` rejects a state whose rate presence does not match the trajectory's existing states, so a trajectory is never a mix of rate-carrying and rate-free samples. `has_rates()` reports which case a non-empty trajectory is in.

=== "Python"
    ``` python
    import brahe as bh

    traj = bh.AttitudeTrajectory(
        bh.AttitudeFrame.reference_frame(bh.ReferenceFrame.GCRF),
        bh.AttitudeFrame.spacecraft_body_frame("SC_BODY", "1"),
    )
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    traj.add(epoch, bh.Quaternion(1.0, 0.0, 0.0, 0.0))
    print(traj.has_rates)  # False
    ```

=== "Rust"
    ``` rust
    use brahe::attitude::{AttitudeFrame, Quaternion, SpacecraftBodyFrame};
    use brahe::frames::ReferenceFrame;
    use brahe::time::{Epoch, TimeSystem};
    use brahe::traits::Trajectory;
    use brahe::trajectories::{AttitudeState, AttitudeTrajectory};

    let mut traj = AttitudeTrajectory::new(
        AttitudeFrame::ReferenceFrame(ReferenceFrame::GCRF),
        AttitudeFrame::SpacecraftBody(SpacecraftBodyFrame::SCBody(Some("1".to_string()))),
    );
    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    traj.add(epoch, AttitudeState::new(Quaternion::new(1.0, 0.0, 0.0, 0.0))).unwrap();
    assert!(!traj.has_rates());
    ```

## Interpolation

`interpolate` (Rust) or the [`AttitudeProvider`](#attitudeprovider) accessors (Rust and Python) retrieve the attitude at an arbitrary epoch within the trajectory's span, using the configured `AttitudeInterpolationMethod`:

- **`Slerp`** (default): spherical linear interpolation of the bracketing quaternions. Exact for constant-angular-rate motion, and always produces a unit quaternion. This is the same interpolation family used for attitude in most spacecraft dynamics software, and is brahe's chosen default for an AEM segment whose `INTERPOLATION_METHOD` is not set, since the standard itself does not mandate a method in that case.
- **`Linear`**: componentwise linear interpolation of the bracketing quaternions (scalar-first), renormalized afterward. Because the two bracketing quaternions can represent the same rotation with either sign — a unit quaternion and its negation are the same attitude — brahe aligns their hemisphere (negating the later quaternion's components if its dot product with the earlier one is negative) before interpolating, so the short way around is always taken and the result varies continuously across that sign boundary.
- **`Lagrange { degree }`**: Lagrange polynomial interpolation over a window of `degree + 1` samples centered on the query epoch, hemisphere-aligned sequentially and renormalized afterward.

Body angular velocity, when present, always interpolates linearly regardless of the quaternion interpolation method (except under `Lagrange`, where it uses the same polynomial degree). An epoch matching a stored node exactly returns that node's state directly; otherwise the query epoch must lie within `[start_epoch, end_epoch]`, and interpolation errors outside that range.

The following example builds a trajectory from a constant-rate rotation and compares `Slerp` against `Linear` at a query epoch away from the interpolation midpoint, where the two methods diverge:

=== "Python"
    ``` python
    --8<-- "./examples/attitude/attitude_interpolation.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/attitude/attitude_interpolation.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/attitude/attitude_interpolation.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/attitude/attitude_interpolation.rs.txt"
        ```

## AttitudeProvider

`AttitudeProvider` is the common interface for retrieving attitude at an arbitrary epoch — `AttitudeTrajectory` is its only implementer today, but the trait exists so that other epoch-parameterized attitude sources (e.g. an analytic attitude law) can be used interchangeably. It provides:

- `quaternion(epoch)` — attitude quaternion, frame A to frame B
- `angular_velocity(epoch)` — body angular velocity in rad/s; errors if the underlying trajectory does not carry rate data (brahe never silently finite-differences a quaternion history to fabricate a rate)
- `euler_angle(epoch, order)` — Euler angles in the requested sequence
- `euler_axis(epoch)` — axis-angle representation
- `rotation_matrix(epoch)` — direction cosine matrix
- `quaternions(epochs)` / `angular_velocities(epochs)` — batched forms of the above

Continuing from the `traj` built in the example above:

=== "Python"
    ``` python
    q = traj.quaternion(epoch)
    order = bh.EulerAngleOrder.ZXZ
    angles = traj.euler_angle(epoch, order)
    ```

=== "Rust"
    ``` rust
    use brahe::attitude::EulerAngleOrder;
    use brahe::traits::AttitudeProvider;

    let q = traj.quaternion(epoch).unwrap();
    let angles = traj.euler_angle(epoch, EulerAngleOrder::ZXZ).unwrap();
    ```

## frame_a / frame_b Semantics

`frame_a` and `frame_b` are [`AttitudeFrame`](../attitude_representations/index.md) values — each one either a reference frame, an orbit-relative frame, or a spacecraft frame. Every stored quaternion rotates from `frame_a` to `frame_b`, mirroring the `REF_FRAME_A`/`REF_FRAME_B` pair on an AEM segment. Which frame is inertial and which is body-fixed is not fixed by the trajectory type itself — some sources put the spacecraft frame first, others put it second — so `frame_a`/`frame_b` are the only reliable way to tell which is which for a given trajectory.

### Composing with a Different Reference Frame

`quaternion_from_frame(epoch, from)` re-expresses the trajectory's attitude relative to an arbitrary reference frame, when `frame_a` is itself a reference frame (constructed via `AttitudeFrame.reference_frame`/`AttitudeFrame::ReferenceFrame`). Internally, this composes the frame-router rotation from `from` to `frame_a`'s reference frame with the trajectory's own stored rotation from `frame_a` to `frame_b`. `frame_a` must be a reference frame for this to succeed, so the call below assumes a trajectory built as in the [rate uniformity example](#canonical-state-and-rate-uniformity) above, where `frame_a` is `GCRF`:

=== "Python"
    ``` python
    q = traj.quaternion_from_frame(epoch, bh.ReferenceFrame.EME2000)
    ```

=== "Rust"
    ``` rust
    use brahe::frames::ReferenceFrame;

    let q = traj.quaternion_from_frame(epoch, ReferenceFrame::EME2000).unwrap();
    ```

Because brahe's Hamilton product `x * y` applies `x` first, composing "from $\to$ frame_a" then "frame_a $\to$ frame_b" is written `q_from_to_a * q_a_to_b`, not the reverse; `quaternion_from_frame` handles this composition internally. It errors if `frame_a` is not a reference frame, if the frame-router transformation fails, or if the attitude at `epoch` cannot be computed.

---

## See Also

- [Trajectories Overview](index.md) — Trait hierarchy and implementation guide
- [Trajectory](trajectory.md) — Dynamic-dimension trajectory
- [AEM — Attitude Ephemeris Message](../ccsds/aem.md) — Parsing AEM data into an `AttitudeTrajectory`
- [Attitude Representations](../attitude_representations/index.md) — `AttitudeFrame`, `Quaternion`, and related types
- [AttitudeTrajectory API Reference](../../library_api/trajectories/attitude_trajectory.md)
