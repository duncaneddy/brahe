# Data Files

The `brahe.datafiles` module reads and writes scientific data file formats and converts their contents to Brahe's SI-unit types. Each format has its own set of types, so file-specific units, column layouts, and sentinel values are handled at the file boundary and every quantity on a parsed object is already in meters, meters per second, or seconds.

## Available Formats

### [SP3 Precise Orbit Files](sp3.md)

The Extended Standard Product 3 format distributes precise satellite positions, velocities, and clock corrections for GNSS and other satellites, in versions a through d. Brahe parses all four versions into an `SP3` object and writes SP3-d.

---

## See Also

- [Datafiles Library API](../../library_api/datafiles/index.md) -- Reference for the datafile types
- [Ephemeris Data Sources](../ephemeris/index.md) -- Clients that download satellite ephemeris data
