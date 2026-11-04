# SP3 Precise Orbit Files

The Extended Standard Product 3 (SP3) format is the exchange format for precise satellite orbits. A file holds a header followed by one block per epoch, with a position record, and optionally a velocity record, for each satellite. Positions are tabulated in an Earth-fixed frame, and the satellite clock correction and its uncertainties can accompany each state. Brahe reads versions a, b, c, and d of the format into an `SP3` object and writes version d. The format is defined by the [SP3-d specification](https://files.igs.org/pub/data/format/sp3d.pdf), the [SP3-c description](https://files.igs.org/pub/data/format/sp3c.txt), and the [SP3-a description](https://files.igs.org/pub/data/format/sp3_docu.txt).

## Reading a File

=== "Python"
    ``` python
    --8<-- "./examples/datafiles/sp3_parse.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/datafiles/sp3_parse.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/datafiles/sp3_parse.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/datafiles/sp3_parse.rs.txt"
        ```

`SP3.from_file` reads a plain-text file and `SP3.from_str` parses text already in memory. Both return the same object and raise a parse error when the file is malformed. `sp3.record(satellite, epoch_index)` returns the `SP3Record` of a satellite at an epoch, or `None` when the satellite has no usable position at that epoch. The satellite argument is an `SP3SatelliteId` or, in Python, a string such as `"G01"`. In the example, the second epoch of G01 is `None` because the file tabulates the position as the format's bad-value marker `0.000000`.

## Format and Versions

An SP3 file starts with a fixed set of header lines: the version, start epoch, epoch count, and coordinate system on the first line; the GPS week, epoch interval, and Modified Julian Day on the second line; the satellite list with per-satellite accuracy exponents; the time system and the standard deviation bases; and free-text comments. Epoch lines starting with `*` follow, each followed by a `P` position line per satellite, optional `V` velocity lines, and for versions c and d optional `EP` and `EV` correlation lines. The file ends with `EOF`. The version is the second character of the first line.

| Version | Satellites | Content |
|---|---|---|
| a | GPS only | Positions, optional velocities, clock corrections; time system fixed to GPS time |
| b | Multi-GNSS | Version a with a system letter on each satellite identifier |
| c | Up to 85 | Adds the time system label, standard deviation bases, the clock event, clock prediction, maneuver, and orbit prediction flags, and `EP` and `EV` correlation records; exactly four comment lines of up to 57 characters (columns 4-60) |
| d | Up to 999 | Version c with a three-digit satellite count field, at least five `+` lines and more above 85 satellites, an unlimited number of comment lines, and comments of up to 77 characters (columns 4-80) |

In version a the `%c` and `%f` placeholder lines carry no information, so the time system reads as GPS and a non-zero standard deviation base is an error.

`SP3Version`, `SP3FileType`, `SP3TimeSystem`, `SP3OrbitType`, and `GNSSSystem` represent the header labels. `SP3FileType` is the system letter of the first satellite line (`G`, `R`, `E`, `C`, `J`, `I`, `S`, `L`, or `M` for mixed), `SP3OrbitType` is `FIT`, `EXT`, `BCT`, or `HLM`, and any other label, such as `BHN`, `PRO`, or `UKN`, is kept as `Other` with the raw text. An `SP3SatelliteId` is a `GNSSSystem` and a number from 1 to 99 and prints as the three-character identifier, for example `G01` or `E11`. A blank system character in the file is read as GPS.

## Data Model and Units

An `SP3` stores a header, the ascending list of epochs, the satellites in header order, and a dense table of optional records indexed by epoch and then satellite. `sp3.header` returns an `SP3Header` holding the fields of the header lines as read, `sp3.epochs` and `sp3.satellite_ids` return the two axes, and `len(sp3)` and `satellite in sp3` follow the epoch and satellite axes. A table entry is `None` when the satellite is not listed at that epoch or when its position is the bad-value marker.

An `SP3Record` holds the position and, when the file carries them, the velocity, clock correction, clock rate, the standard deviations of each, the four flags (`clock_event`, `clock_predicted`, `maneuver`, `orbit_predicted`), and the `EP` and `EV` correlation records as `SP3Correlation` objects. A correlation coefficient left blank in the file is unknown and is stored as `None`, distinct from an explicit 0; the writer leaves it blank. A written coefficient must lie within $\pm 0.9999999$. Every quantity on these objects is in SI units. The parser converts from the file units when reading and the writer converts back.

| Quantity | File unit | Stored unit |
|---|---|---|
| Position | km | m |
| Velocity | dm/s | m/s |
| Clock correction | $\mu$s | s |
| Clock rate | $10^{-4}\ \mu$s/s | s/s |
| Position standard deviation | $b^n$ mm | m |
| Velocity standard deviation | $b^n \times 10^{-4}$ mm/s | m/s |
| Clock standard deviation | $b^n$ ps | s |
| Clock rate standard deviation | $b^n \times 10^{-4}$ ps/s | s/s |
| Satellite accuracy | $2^n$ mm | m |

A standard deviation is stored in the file as an exponent $n$ of a base $b$ declared on the `%f` header lines, so its value is $b^n$ in the unit of the column. The header fields `position_base` and `clock_base` hold the two bases. An exponent of 99 for a position or velocity component, or 999 for a clock, marks an unbounded value and is stored as infinity. A blank exponent is stored as `None`. A clock value of `999999.999999` in the file is a missing clock and is stored as `None`.

The header fields `start_epoch`, `number_of_epochs`, `gps_week`, `seconds_of_week`, `mjd`, and `fractional_day` describe the first epoch and are stored as read. The writer recomputes them from the epoch list, so changing the epochs of a file never leaves the written header inconsistent.

## Time Systems

The `%c` header line names the time system of the epochs. `SP3TimeSystem` keeps the label, and `to_time_system()` maps it to the Brahe time scale for the same system.

| SP3 label | Brahe `TimeSystem` |
|---|---|
| `GPS` | `GPS` |
| `GAL` | `GST` |
| `BDT` | `BDT` |
| `TAI` | `TAI` |
| `UTC` | `UTC` |
| `GLO`, `QZS`, `IRN` | none |

Parsed epochs are `Epoch` objects constructed directly in the mapped time system, so no leap-second or scale conversion is applied while reading, and the epoch of a GPS-time file prints with the `GPS` suffix. A UTC file may carry an epoch inside a leap second, written with seconds from 60 to 61 as in `23:59:60.00000000`; a second of 60 or more at any other time, or in any other time system, is rejected. A file whose label has no Brahe equivalent is rejected with an error of the form `SP3: line 13: time system 'GLO' is not supported`. The writer formats each epoch in the time system of the file and raises the same kind of error when the label has no mapping. Convert epochs to another scale with the `Epoch` methods after reading.

## Coordinate Systems

The header stores the coordinate system label as read, for example `IGS14`, `IGS20`, `WGS84`, or `ITRF14`. Positions and velocities are Earth-fixed in the realization the label names. Brahe does not interpret the label; it is kept in `SP3Header.coordinate_system` and written back unchanged when it fits the 5-character SP3-d field. A longer label, such as `ITRF14`, must be shortened in the header before the file is written.

## Parsing Details

The parser follows the line sequence the specification allows and reports the one-based line number of any error as `SP3: line N: <detail>`. The first line, the `##` line, and the epoch lines are split on whitespace, which accepts files whose columns have shifted. The satellite, record, and flag fields use the specification's fixed columns.

The parser accepts the following deviations found in files from real producers.

| Deviation | Handling |
|---|---|
| Epoch fields of minute 60, hour 24, or day 0 | Carried into the next field |
| Seconds with fewer digits than the specification, such as `0.1234` | Accepted |
| Satellite identifier padding of ` 00` in place of `  0` | Accepted |
| Coordinate system labels longer than five characters | Accepted, with later fields shifted |
| Empty agency field | Accepted |
| Inconsistent `##` line, such as GPS week 0 and MJD 0 | Ignored; the first line is authoritative |
| 136 satellites with exactly eight `+` lines | Accepted |
| Blank standard deviation exponents | Stored as `None` |
| `P` and `V` lines ending before the clock field | Clock stored as `None` |
| Header declaring more satellites than appear at an epoch | Missing satellites stored as `None` |
| Orbit type labels outside the specification | Stored as `Other` |
| More comment lines than the header declares, or comments beyond the length limit | Accepted |
| `%/*` comment lines and blank lines | Skipped |
| `EOF` in any case, with trailing whitespace; content after `EOF` | Accepted and ignored |
| Missing `EOF` when the epoch count matches the header | Accepted |

When a rollover is carried, a reading without the carry is used instead when it lies within one millisecond of the expected epoch, which covers producers that already carried into the next field.

The parser raises an error in these cases.

| Condition | Example message |
|---|---|
| Truncated record line | `SP3: line 27: record truncated in columns 47-60` |
| Epoch count differing from the header | `SP3: header declares 192 epochs but the file has 2` |
| Unknown line identifier | `SP3: line 13: unrecognized line identifier '@@'` |
| Unsupported version | `SP3: line 1: unsupported version 'z'` |
| Non-zero standard deviation base in a version a file | `SP3: line 15: version a file declares a non-zero clock base 1.025` |
| Standard deviation exponent with no declared base in a version c or d file | `SP3: line 24: clock standard deviation exponent but the %f line declares no clock base` |
| Epoch interval of zero or less | `SP3: line 2: epoch interval must be positive, got 0` |
| Epoch off the interval grid by more than one millisecond | `SP3: line 37: epoch ... is not on the 900 s grid starting at ...` |
| Time system without a Brahe equivalent | `SP3: line 13: time system 'GLO' is not supported` |

The parser does not enforce the comment count, comment length, or satellite count limits of the declared version listed in the version table. The writer always produces version d and enforces its limits: at most 999 satellites and comments of at most 77 characters.

## Per-Satellite Data

`sp3.satellite(id)` extracts one satellite as an `SP3Satellite`, and `sp3.satellites()` extracts every satellite that has at least one valid sample, in header order. A sample is valid when the satellite has a record at that epoch whose position is not the zero position. A missing or bad clock does not invalidate a sample, and neither does a maneuver flag. `satellite(id)` raises an error for a satellite with no valid sample, while `satellites()` leaves such a satellite out.

An `SP3Satellite` keeps only the valid samples, and every data accessor returns one entry per sample, aligned with `epochs()`. `positions()` and `velocities()` return the ITRF position in m and velocity in m/s, `clocks()` and `clock_rates()` return seconds and s/s, and `records()` returns the full `SP3Record` of each sample, including standard deviations, flags, and correlations. In Python the position and velocity accessors return $N \times 3$ NumPy arrays and the clock accessors return length-$N$ arrays with `NaN` where the clock is absent. In Rust the clock accessors return `Option<f64>` values. `has_file_velocities()` reports whether the velocities come from the file, which requires a `V` record at every sample.

### Orbit State Provider

An `SP3Satellite` is an orbit state provider with the same `state_*` and `states_*` methods as an `OrbitTrajectory`, and it is named after its identifier, such as `"G01"`. States are interpolated in ITRF with Lagrange polynomials of degree `SP3_DEFAULT_LAGRANGE_DEGREE` (9, which fits 10 samples) and converted on request, so `state_itrf` returns the interpolated state and `state_gcrf`, `state_eci`, and `state_in_frame` apply the Earth-orientation transforms. In Python `state_in_frame(frame, epoch)` takes a `CelestialFrame`. `set_interpolation_method` changes the interpolation method of the satellite and of the trajectories it returns afterwards. The satellite can be passed to `location_accesses` and to any other function that takes an orbit state provider.

=== "Python"
    ``` python
    --8<-- "./examples/datafiles/sp3_satellite_access.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/datafiles/sp3_satellite_access.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/datafiles/sp3_satellite_access.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/datafiles/sp3_satellite_access.rs.txt"
        ```

The example computes access windows over a ground station in Boulder for the three satellites of a CODE final orbit file. A request before the first sample, after the last sample, or inside a gap raises an error naming the satellite, so an access search that reaches a gap fails. For a satellite with gaps, search each trajectory from `to_trajectory_segments()` instead.

### Trajectories

`to_trajectory()` returns the satellite as one `OrbitTrajectory`, and `to_trajectory_segments()` returns one trajectory for each contiguous run of samples. Each trajectory is a six-element Cartesian trajectory in the `ITRF` frame, named after the satellite, with epochs in the time system of the file. Call `to_gcrf()` on a trajectory for an inertial copy. A run needs at least two samples to become a trajectory.

### Velocity Derivation

When a file has no velocity records, or only some samples have them, the velocity at each sample is derived from the positions. The derivation fits a Lagrange polynomial of degree 9 through the samples of the same contiguous run, centered on the sample as far as the run allows, and takes the analytic derivative at the sample. A run shorter than 10 samples uses the largest degree its length allows, down to a linear fit. A sample that forms a run on its own has no derived velocity and reports `NaN`; it stays in the data accessors, provider calls at its epoch raise an error, and `to_trajectory_segments()` omits it. The fit degree is fixed and does not change with `set_interpolation_method`.

The accuracy of interpolation and derived velocity depends on the position of the sample in its run. The figures below are for a 900 s GPS orbit at degree 9.

| Samples | Position error | Velocity error |
|---|---|---|
| Exact positions, interior | 0.8 mm | $2 \times 10^{-6}$ m/s |
| Positions at SP3 file precision (1 mm grid), interior | 1.4 mm | $2.2 \times 10^{-6}$ m/s |
| Positions at SP3 file precision, within 4 samples of a run edge | 1.2 cm | $1.4 \times 10^{-4}$ m/s |
| CODE final orbit subsampled from 300 s to 900 s, interior | 1.4 mm | Not measured |

The last row compares interpolated positions with the withheld 300 s samples of a real file. Within about $\mathrm{degree}/2$ samples of the start or end of a run, the fit is one-sided and both derived velocities and interpolated positions are less accurate than in the interior. The edge applies to a single file and to a chained set alike, so chaining the neighboring days around a span of interest places that span in the interior of the fit.

### Gaps

A gap is a missing sample between the first and last valid sample of a satellite. Two consecutive valid samples more than 1.5 times the larger of their epoch intervals apart belong to different runs. `segments()` returns the `(start, end)` span of each run, and `has_gaps()` is true when there is more than one. A satellite with zero positions from 10:00 to 10:45 in a 15-minute file has one gap, and chaining Monday's and Wednesday's files without Tuesday's leaves every satellite a 24-hour gap. State requests inside a gap raise an error naming the satellite and the gap, and `to_trajectory()` raises the same error and points to `to_trajectory_segments()`, which splits the satellite at each gap.

## Chaining Files

`SP3Satellite.from_files(files)` builds one `SP3Satellite` per satellite across several `SP3` objects, and returns them in ascending identifier order (GPS, GLONASS, Galileo, BeiDou, QZSS, IRNSS, SBAS, LEO, then by number). The files are sorted by start epoch, so they can be passed in any order, and each satellite in the union of the satellite lists collects its valid samples from every file that contains it. Samples are joined before velocities are derived, so the samples next to a file boundary get centered fits.

=== "Python"
    ``` python
    --8<-- "./examples/datafiles/sp3_satellites_from_files.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/datafiles/sp3_satellites_from_files.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/datafiles/sp3_satellites_from_files.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/datafiles/sp3_satellites_from_files.rs.txt"
        ```

The example chains three five-minute files. The first two files and the last two files each share a boundary epoch, so every satellite has four samples and one run. Chaining the first file with a file that starts at 00:20 leaves a 15-minute gap and two segments.

Two consecutive files may share their boundary epoch. The later file's sample is kept there when it is valid, and the earlier file's sample is kept otherwise; no agreement check is applied, and daily fits from the same center can differ by centimeters at the boundary. Files that overlap by more than a shared boundary epoch raise an error naming the agency and span of both files. A file that lacks a satellite contributes no samples, which leaves a gap in that satellite instead of dropping it. Velocities come from the files only when every sample comes from a file with velocity records; if any contributing file lacks them, velocities are derived for every sample of the satellite.

Time systems, coordinate system labels, agencies, orbit types, and epoch intervals do not need to match, because epochs are absolute instants and every label maps to Brahe's single ITRF frame. Two caveats follow from this. A change of ITRF realization, such as IGS14 to IGS20, joins the samples as they are, and the boundary can carry a jump of millimeters to centimeters. Mixing analysis centers or solution types, such as a final file followed by a rapid file, produces visible jumps at the file boundary in the same way. `sp3.satellites()` behaves as `SP3Satellite.from_files` applied to the single file.

## Writing a File

=== "Python"
    ``` python
    --8<-- "./examples/datafiles/sp3_write.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/datafiles/sp3_write.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/datafiles/sp3_write.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/datafiles/sp3_write.rs.txt"
        ```

`SP3(satellites, epoch_interval, ...)` creates an empty file with a fixed epoch grid. The keyword arguments set the agency, time system, coordinate system label, orbit type, data-used label, whether velocity records are written, and the comment lines. A label longer than its SP3-d field (agency 4, coordinate system 5, data used 5, orbit type 3 characters) is rejected when the file is created. `push_epoch` appends an epoch and returns its index, and `set_record` stores an `SP3Record` for a satellite at that index. An appended epoch must be later than the last epoch and lie on the grid anchored at the first epoch, to within one millisecond; grid points may be skipped. A satellite without a record at an epoch is written with the zero position and the missing-clock value that the format defines for absent data, and reads back as `None`.

`to_string` returns the file as SP3-d text with `\n` line endings, and `to_file` writes it to a path. The writer always produces version d, so a file read as version a, b, or c is upgraded when written. The writer reports an error when the file cannot be represented in SP3-d: a file with no epochs, more than 999 satellites, a satellite number outside 1 to 99, a time system without a Brahe equivalent, an epoch interval that rounds to zero or does not fit the 14-column `##` field at eight decimal places, epochs that, rounded to the 1e-8 s resolution of the epoch lines, are out of order or off the grid of the interval as written, a record table that does not match the epochs and satellites, a comment longer than 77 characters, a header label longer than its field (agency 4, coordinate system 5, data used 5, and orbit type 3 characters; shorten it through `header_mut` in Rust or by assigning a modified `header` in Python), a standard deviation base that is not finite and greater than 1 as written, a clock or clock rate that would read back as the missing-clock value, or a value that does not fit its field width.

Standard deviations are written only when the matching base is set in the header (`position_base` for position and velocity, `clock_base` for clock and clock rate). A standard deviation $\sigma$, expressed in the column unit, is written as the exponent

$$
n = \left\lceil \log_b \sigma \right\rceil
$$

computed with the base as written in the `%f` line. An infinite value, or an exponent that reaches 99 for a position or velocity component or 999 for a clock, is written as the unbounded marker, so the written standard deviation is never smaller than the stored one. Satellite accuracies are written as $\lceil \log_2 \sigma_\mathrm{mm} \rceil$, at least 1, with an unknown accuracy written as 0. The first line of the file written above, `#dV2024 11 10  0  0  0.00000000       3 ORBIT IGS20 EXT BRHE`, shows the version `d`, the `V` flag for a file with velocity records, the start epoch, the epoch count, and the data-used, coordinate system, orbit type, and agency labels.

## Building from Propagators and Setting Clocks

`SP3.from_propagators(propagators, start, end, step, ...)` builds a file by sampling orbit state providers on a regular grid. The `propagators` argument is a list of satellite identifier and provider pairs, where a provider is any orbit state provider: a numerical, Keplerian, or SGP propagator, an orbit trajectory, or an `SP3Satellite`, which resamples an existing file. Each provider is sampled with `state_itrf` at `start + k * step` from `start` up to the last grid point not after `end`, and velocities are stored when `include_velocity` is set. The keyword arguments are those of the `SP3` constructor, with the defaults `coordinate_system = "IGS20"`, `orbit_type = EXT`, `agency = "BRHE"`, and `data_used = "ORBIT"`. The file type follows the satellite systems: the system letter when all identifiers share one system, `L` for low Earth orbit satellites only, and `M` for mixed systems. A provider that cannot supply a state at a grid epoch, such as a propagator that has not been propagated that far, raises an error naming the satellite and the epoch.

=== "Python"
    ``` python
    --8<-- "./examples/datafiles/sp3_from_propagator.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/datafiles/sp3_from_propagator.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/datafiles/sp3_from_propagator.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/datafiles/sp3_from_propagator.rs.txt"
        ```

The example propagates two satellites for an hour before building the file, because a numerical propagator answers state requests from the trajectory it has accumulated. The first ten lines printed are the header of the written file, which lists both satellites and declares 61 epochs at a 60 s interval.

`from_propagators` takes no clock input, so clocks and clock rates are absent and are written as the missing-clock value `999999.999999`. `sp3.set_clocks(satellite, clocks)` sets the clock correction of a satellite at every epoch of the file, in seconds, and `sp3.set_clock_rates(satellite, rates)` sets the clock rate in s/s. Both take one value per epoch, `None` clears the value at that epoch, and in Python `NaN` is treated as `None`, so the `clocks()` array of an `SP3Satellite` can be passed back directly. Clock rates are stored on velocity records, so `set_clock_rates` requires a file with velocity records. A call that fails, for example because the number of values differs from the number of epochs or a value is given where the satellite has no record, leaves the file unchanged.

---

## See Also

- [SP3 Library API](../../library_api/datafiles/sp3.md) -- Reference for `SP3` and the related types
- [Epoch](../time/epoch.md) -- Time scales and epoch construction
- [Access Computation](../access_computation/computation.md) -- Access windows for orbit state providers
