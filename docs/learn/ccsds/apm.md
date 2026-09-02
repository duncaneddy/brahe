# APM — Attitude Parameter Message

An Attitude Parameter Message (APM) carries a spacecraft's attitude state at a single epoch through one or more logical blocks — quaternion, Euler angle, angular velocity, spin, inertia, and maneuver. It is the attitude-message counterpart to the OPM: a compact snapshot for handing off attitude state or documenting a planned attitude maneuver. The message is defined by the [CCSDS 504.0-B-2 Attitude Data Messages standard](https://ccsds.org/Pubs/504x0b2.pdf).

## Parse and Access

Parse from file or string, then access header properties, metadata, and the attitude quaternion:

=== "Python"
    ``` python
    --8<-- "./examples/ccsds/apm_parse_access.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/ccsds/apm_parse_access.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/ccsds/apm_parse_access.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/ccsds/apm_parse_access.rs.txt"
        ```

## How APM Messages Are Organized

Every APM has a **header** (version, creation date, originator), **metadata** (object identity, center body, time system), and a single **epoch** that applies to every logical block except maneuvers. The attitude information itself lives in up to six repeatable logical blocks, each of which can appear zero or more times. A message is only valid to write or parse if at least one block is present.

**Quaternion** blocks (`QUAT_START`/`QUAT_STOP`) carry the attitude quaternion and an optional time derivative. **Euler angle** blocks (`EULER_START`/`EULER_STOP`) carry the same rotation as a three-angle sequence plus optional angle rates; the rotation sequence (e.g. `ZXZ`) is stored alongside the angles. **Angular velocity** blocks (`ANGVEL_START`/`ANGVEL_STOP`) carry an angular velocity vector along with the frame it is expressed in. **Spin** blocks (`SPIN_START`/`SPIN_STOP`) describe a spin-stabilized attitude by spin-axis right ascension, declination, phase angle, and spin rate, with an optional nutation description. **Inertia** blocks (`INERTIA_START`/`INERTIA_STOP`) carry the spacecraft's moment-of-inertia tensor. **Maneuver** blocks (`MAN_START`/`MAN_STOP`) describe a planned or executed attitude maneuver as a torque vector over a duration; unlike the other blocks, a maneuver carries its own epoch rather than using the message epoch.

Every quaternion, Euler angle, and angular velocity block declares a pair of reference frames, `REF_FRAME_A` and `REF_FRAME_B`, that together define the rotation direction: the block's values transform a vector from frame A to frame B. This A$\to$B convention is fixed by CCSDS 504.0-B-2 and does not depend on which frame is inertial or body-fixed — some fixtures put the spacecraft body frame first, others put it second, and the block's own `REF_FRAME_A`/`REF_FRAME_B` fields are the only reliable way to tell which.

CCSDS wire values use different units and component ordering than brahe's internal representation. Brahe converts at the KVN/XML/JSON parse and write boundary, so every value returned by the Python and Rust APIs is already in SI units and brahe's native quaternion convention:

| Quantity | Wire (CCSDS) | Internal (brahe) |
|---|---|---|
| Angles (Euler, spin) | degrees | radians |
| Angle rates (Euler rates, spin rate, nutation rate) | deg/s | rad/s |
| Quaternion component order | scalar-last: `Q1 Q2 Q3 QC` | scalar-first: `Quaternion::new(w, x, y, z)` |
| Quaternion derivative order | scalar-last: `Q1_DOT Q2_DOT Q3_DOT QC_DOT`, 1/s | scalar-first vector, 1/s |
| Inertia tensor components | kg$\cdot$m$^2$ | kg$\cdot$m$^2$ (unchanged) |
| Maneuver torque | N$\cdot$m | N$\cdot$m (unchanged) |

The quaternion reordering matters because most quaternion libraries, including brahe's `Quaternion`, use a scalar-first convention internally while CCSDS 504.0-B-2 fixes the wire order as scalar-last. Use `Quaternion.to_vector(scalar_first=False)` (Python) or `Quaternion::to_vector(false)` (Rust) to recover the wire-order `[Q1, Q2, Q3, QC]` components shown in a KVN file.

## Creating and Writing APMs

Build an APM programmatically by defining a header, epoch, and metadata, then adding one or more logical blocks. The resulting message can be serialized to KVN, XML, or JSON:

=== "Python"
    ``` python
    --8<-- "./examples/ccsds/apm_create_write.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/ccsds/apm_create_write.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/ccsds/apm_create_write.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/ccsds/apm_create_write.rs.txt"
        ```

!!! info "Round-Trip Fidelity"
    Writing and re-parsing an APM preserves all header, metadata, and logical-block values. Numeric precision may vary slightly due to floating-point formatting, but values are preserved within the precision of the output format.

## KVN Format Example

An excerpt from the CCSDS 504.0-B-2 Annex G-1 example file, containing a header, metadata, and a single quaternion block:

```text
CCSDS_APM_VERS = 2.0
CREATION_DATE = 2003-09-30T19:23:57
ORIGINATOR   = GSFC
MESSAGE_ID = A7015Z1

OBJECT_NAME  = TRMM
OBJECT_ID    = 1997-074A
CENTER_NAME  = EARTH
TIME_SYSTEM  = UTC

EPOCH     = 2003-09-30T14:28:15.1172

QUAT_START
REF_FRAME_A  = SC_BODY_1
REF_FRAME_B  = ITRF1997

Q1        = 0.00005
Q2        = 0.87543
Q3        = 0.40949
QC        = 0.25678
QUAT_STOP
```

Note that this quaternion block has no bracketed unit annotations — quaternion components are dimensionless. Angle-valued blocks such as Euler angle and spin blocks carry `[deg]` annotations, which brahe strips during parsing.

---

## See Also

- [API Reference — APM](../../library_api/ccsds/apm.md)
- [CCSDS Data Formats](index.md) — Overview of all message types
- [OPM — Orbit Parameter Message](opm.md) — The orbit-state counterpart to APM
- [Attitude Representations](../attitude_representations/index.md) — Quaternion, Euler angle, and rotation matrix conventions
- [CCSDS 504.0-B-2](https://ccsds.org/Pubs/504x0b2.pdf) — Attitude Data Messages, the standard APM implements
