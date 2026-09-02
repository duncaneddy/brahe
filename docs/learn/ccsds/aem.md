# AEM — Attitude Ephemeris Message

An Attitude Ephemeris Message (AEM) carries a spacecraft's time-ordered attitude history as one or more segments, each holding a sequence of attitude data lines at strictly increasing epochs. It is the attitude-message counterpart to the OEM: the standard format for exchanging attitude ephemerides between agencies and operators. The message is defined by the [CCSDS 504.0-B-2 Attitude Data Messages standard](https://ccsds.org/Pubs/504x0b2.pdf).

## Parse and Access

Parse from file or string, then access header, metadata, and attitude data for each segment:

=== "Python"
    ``` python
    --8<-- "./examples/ccsds/aem_parse_access.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/ccsds/aem_parse_access.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/ccsds/aem_parse_access.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/ccsds/aem_parse_access.rs.txt"
        ```

## Segments and Attitude Types

An AEM message has a **header** (version, creation date, originator) and one or more **segments**. Each segment carries its own metadata — object identity, center body, reference frames, time system, and the segment's total and useable time spans — followed by a data block of attitude states at strictly increasing epochs. A message is only valid to write if every segment has at least one state.

Every segment declares `REF_FRAME_A` and `REF_FRAME_B`, and every attitude value in the segment's data block is a rotation from frame A to frame B, exactly as in APM. A segment's `ATTITUDE_TYPE` fixes which of nine data layouts its data lines use; brahe rejects a data line whose column count does not match the declared type. Data lines carry no keyword names or bracketed units — only the epoch followed by the fixed-order numeric columns below (angles and rates on the wire are degrees and deg/s; brahe converts to radians and rad/s on parse):

| `ATTITUDE_TYPE` | Columns (after epoch) | Conditional metadata |
|---|---|---|
| `QUATERNION` | `Q1 Q2 Q3 QC` | — |
| `QUATERNION/DERIVATIVE` | `Q1 Q2 Q3 QC Q1_DOT Q2_DOT Q3_DOT QC_DOT` | — |
| `QUATERNION/ANGVEL` | `Q1 Q2 Q3 QC ANGVEL_X ANGVEL_Y ANGVEL_Z` | `ANGVEL_FRAME` |
| `EULER_ANGLE` | `ANGLE_1 ANGLE_2 ANGLE_3` | `EULER_ROT_SEQ` |
| `EULER_ANGLE/DERIVATIVE` | `ANGLE_1 ANGLE_2 ANGLE_3 ANGLE_1_DOT ANGLE_2_DOT ANGLE_3_DOT` | `EULER_ROT_SEQ` |
| `EULER_ANGLE/ANGVEL` | `ANGLE_1 ANGLE_2 ANGLE_3 ANGVEL_X ANGVEL_Y ANGVEL_Z` | `EULER_ROT_SEQ`, `ANGVEL_FRAME` |
| `SPIN` | `SPIN_ALPHA SPIN_DELTA SPIN_ANGLE SPIN_ANGLE_VEL` | — |
| `SPIN/NUTATION` | `SPIN_ALPHA SPIN_DELTA SPIN_ANGLE SPIN_ANGLE_VEL NUTATION NUTATION_PER NUTATION_PHASE` | — |
| `SPIN/NUTATION_MOM` | `SPIN_ALPHA SPIN_DELTA SPIN_ANGLE SPIN_ANGLE_VEL MOMENTUM_ALPHA MOMENTUM_DELTA NUTATION_VEL` | — |

`EULER_ROT_SEQ` is required exactly when `ATTITUDE_TYPE` is one of the `EULER_ANGLE*` types, and its rotation sequence (e.g. `ZXZ`) applies to every `ANGLE_1`/`ANGLE_2`/`ANGLE_3` column in the segment. `ANGVEL_FRAME` is required exactly when `ATTITUDE_TYPE` ends in `/ANGVEL`, and must equal the segment's `REF_FRAME_A` or `REF_FRAME_B`; brahe validates both rules when parsing and when writing. `INTERPOLATION_DEGREE` is required exactly when `INTERPOLATION_METHOD` is present.

## Converting to AttitudeTrajectory

`AEM::segment_to_attitude_trajectory` (Rust) or `aem.segment_to_attitude_trajectory` (Python) converts one segment into an [`AttitudeTrajectory`](../trajectories/attitude_trajectory.md), normalizing every attitude representation to a canonical quaternion (frame A to frame B) plus optional body-frame angular velocity. `AEM::to_attitude_trajectories` (Rust) or `aem.to_attitude_trajectories()` (Python) converts every segment at once.

=== "Python"
    ``` python
    --8<-- "./examples/ccsds/aem_to_trajectory.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/ccsds/aem_to_trajectory.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/ccsds/aem_to_trajectory.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/ccsds/aem_to_trajectory.rs.txt"
        ```

**Interpolation method.** A segment's `INTERPOLATION_METHOD` maps onto [`AttitudeInterpolationMethod`](../trajectories/attitude_trajectory.md):

| AEM `INTERPOLATION_METHOD` | `AttitudeInterpolationMethod` |
|---|---|
| Unset | `Slerp` |
| `LINEAR` | `Linear` |
| `LAGRANGE` | `Lagrange { degree }` (from `INTERPOLATION_DEGREE`) |
| `HERMITE` | Conversion errors — construct the trajectory and call `set_interpolation_method` with an explicit choice instead |

**SPIN limitation.** The `SPIN`, `SPIN/NUTATION`, and `SPIN/NUTATION_MOM` attitude types describe a spin-stabilized attitude by spin-axis geometry rather than a full 3-axis orientation, and have no `AttitudeTrajectory` representation. Converting a segment with one of these types returns an error naming the offending type; the AEM itself can still be read and written normally.

**ANGVEL frame handling.** For the `QUATERNION/ANGVEL` and `EULER_ANGLE/ANGVEL` types, the wire angular velocity is expressed in whichever frame `ANGVEL_FRAME` names. `AttitudeState::angular_velocity` is always in frame B (the canonical convention used throughout brahe), so when `ANGVEL_FRAME` equals `REF_FRAME_A`, brahe re-expresses the vector as $\omega_B = R(q) \, \omega_A$, where $R(q)$ is the rotation matrix of the state's attitude quaternion. When `ANGVEL_FRAME` already equals `REF_FRAME_B`, the value is used as-is. Building an AEM from a rate-carrying `AttitudeTrajectory` always writes `ANGVEL_FRAME = REF_FRAME_B`, so no re-expression is needed on that path.

## Creating and Writing

Build an AEM programmatically by defining metadata, adding attitude states to a segment, and serializing to KVN, XML, or JSON:

=== "Python"
    ``` python
    --8<-- "./examples/ccsds/aem_create_write.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/ccsds/aem_create_write.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/ccsds/aem_create_write.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/ccsds/aem_create_write.rs.txt"
        ```

!!! info "Round-Trip Fidelity"
    Writing and re-parsing an AEM preserves all header, metadata, and attitude-state values. Numeric precision may vary slightly due to floating-point formatting, but values are preserved within the precision of the output format.

## KVN Format Example

The CCSDS 504.0-B-2 Annex G-4 example file ships with brahe as `test_assets/ccsds/aem/AEMExampleG4.txt`, and is the file the [Parse and Access](#parse-and-access) example reads. It holds a header followed by two segments, each with its own metadata block and data block:

```text
--8<-- "./test_assets/ccsds/aem/AEMExampleG4.txt"
```

The data lines carry no keyword names. `QUATERNION` fixes the column order to epoch, `Q1`, `Q2`, `Q3`, `QC`. The first segment's `INTERPOLATION_METHOD = hermite` is preserved on parse and write, but has no `AttitudeTrajectory` equivalent; see [Converting to AttitudeTrajectory](#converting-to-attitudetrajectory) above.

---

## See Also

- [API Reference — AEM](../../library_api/ccsds/aem.md)
- [CCSDS Data Formats](index.md) — Overview of all message types
- [OEM — Orbit Ephemeris Message](oem.md) — The orbit-ephemeris counterpart to AEM
- [APM — Attitude Parameter Message](apm.md) — Single-epoch attitude snapshot
- [AttitudeTrajectory](../trajectories/attitude_trajectory.md) — Native attitude trajectory storage and interpolation
- [Attitude Representations](../attitude_representations/index.md) — Quaternion, Euler angle, and rotation matrix conventions
- [CCSDS 504.0-B-2](https://ccsds.org/Pubs/504x0b2.pdf) — Attitude Data Messages, the standard AEM implements
