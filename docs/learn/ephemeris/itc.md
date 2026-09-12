# Modified ITC Ephemeris Format

The Modified ITC format is the Space-Track ephemeris exchange format defined in the Spaceflight
Safety Handbook for Satellite Operators (version 1.7). A file holds a short header followed by one
record per epoch: a line giving the `YYYYDDDHHMMSS.sss` UTC epoch, position (km) and velocity
(km/s), optionally followed by the lower triangle of a 6x6 position and velocity covariance (21
values, km-based units) in the frame the header names. Brahe reads and writes the format through
`ITC`, converting positions and velocities to SI units (m, m/s) and covariance elements by $10^6$ on
input, so every quantity on an `ITC` object is already in SI.

## Reading a File

=== "Python"
    ``` python
    --8<-- "./examples/itc/itc_parse.py:12"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/itc/itc_parse.rs:7"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/itc/itc_parse.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/itc/itc_parse.rs.txt"
        ```

`ITC.from_file` infers the state frame and the object's identity from a Space-Track compliant file
name: the DataType field `MEME`, `EME2000` or `J2000` all map to EME2000, and `TEME` and `ITRF` map
to those frames; the decoded name is kept on `source_name`. `ITC.from_str` parses only the file text
and leaves the state frame at its default, EME2000, since there is no file name to read a DataType
from. The header carries the creation time, the declared ephemeris start, stop and step, a free-text
source label, and the covariance frame; the state frame comes from the file name rather than the
body. `states` holds the parsed records and
`covariances` holds one matrix per record when the file carries covariance; `has_covariance` is
`true` only when every record does, since a Modified ITC message's covariance is all-or-none.

## Frames and Covariance

The state frame is the file's `MEME J2000.0`, the mean equator and mean equinox of J2000.0, which
brahe treats as `CelestialFrame.EME2000`; it is related to GCRF by the constant frame bias. The
covariance frame `UVW` is RTN (radial, in-track, cross-track), with

$$
\hat{R} = \frac{\mathbf{r}}{|\mathbf{r}|}, \quad
\hat{N} = \frac{\mathbf{r} \times \mathbf{v}}{|\mathbf{r} \times \mathbf{v}|}, \quad
\hat{T} = \hat{N} \times \hat{R}.
$$

When a message is converted to a trajectory, each record's covariance is rotated from RTN into the
state frame with the block-diagonal Jacobian $\mathrm{blockdiag}(R, R)$ by default, following the
NASA Conjunction Assessment Risk Analysis handbook (Appendix N, eq. N-13) and the CARA `RIC2ECI`
implementation. The rotating-frame form

$$
\begin{bmatrix} R & 0 \\ R[\boldsymbol{\omega}\times] & R \end{bmatrix}
$$

(Vallado, AAS 03-526) is available by passing `OrbitRelativeFrameVariant.ROTATING` to
`to_trajectory_with_covariance_variant`. `from_trajectory` and
`from_trajectory_with_covariance_variant` invert the same rotation to build a message from a
trajectory; both reject a trajectory that is not six-dimensional Cartesian, so a Keplerian-element
trajectory must be converted first.

## Converting to a Trajectory

=== "Python"
    ``` python
    --8<-- "./examples/itc/itc_to_trajectory.py:13"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/itc/itc_to_trajectory.rs:8"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/itc/itc_to_trajectory.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/itc/itc_to_trajectory.rs.txt"
        ```

The resulting `OrbitTrajectory` interpolates position, velocity and covariance at any epoch within
its span, and `to_eci` carries the covariance through the constant frame bias between EME2000 and GCRF; converting to a frame that cannot hold covariance, such as ITRF with `to_itrf`, drops it and keeps only the converted states. Because it is an ordinary `OrbitTrajectory`, it can be used directly with
`location_accesses` to compute ground-station access windows, as in the example above, without a
separate conversion step.

## Writing a File and Naming It

=== "Python"
    ``` python
    --8<-- "./examples/itc/itc_write.py:12"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/itc/itc_write.rs:7"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/itc/itc_write.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/itc/itc_write.rs.txt"
        ```

`from_trajectory` requires a Cartesian, six-dimensional trajectory; it fills the header's ephemeris
start, stop and step from the trajectory's samples and keeps the header's state frame, covariance
frame, `created` and `ephemeris_source` fields. `ITC.file_name` then builds a Space-Track compliant
name from the message: `<DataType>_<NORAD>_<Name>_<DDDHHMM>_<Category>_<Metadata>_<Classification>.<ext>`,
where the DataType comes from the state frame, the day-time group from the ephemeris start, and the
catalog number, name, category and metadata from the arguments passed to `file_name`, and the
classification and extension from the defaults of `EphemerisFileName`. That type is the one behind
the convention: `parse` reads a compliant name, constructing an `EphemerisFileName` gives the `MEME`
data type, `UNCLASSIFIED` classification and `txt` extension, and the `with_*` methods replace
individual fields. Every field is rejected if it contains a path separator (`/` or `\`) or is `.` or `..`, so a
name built from untrusted input can never address a file outside its directory; the data type,
metadata, classification and extension fields are additionally rejected if they contain an underscore, since
that is the field delimiter -- the object name is the exception, since `parse` recovers it by
anchoring the other six fixed-position fields from either end and taking whatever remains between
them.

---

## See Also

- [Starlink Public Ephemerides](starlink.md) -- The public mirror that publishes files in this format
- [File Operations](spacetrack/file_operations.md) -- Space-Track's own SP Ephemeris downloads, in the same format
- [Modified ITC API Reference](../../library_api/ephemeris/itc.md) -- Class and function documentation
- [Space-Track Response Types](../../library_api/ephemeris/spacetrack/responses.md) -- `EphemerisFileName` and `EphemerisFileCategory`
