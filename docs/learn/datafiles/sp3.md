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

---

## See Also

- [SP3 Library API](../../library_api/datafiles/sp3.md) -- Reference for `SP3` and the related types
- [Epoch](../time/epoch.md) -- Time scales and epoch construction
