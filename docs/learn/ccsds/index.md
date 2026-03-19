# CCSDS Data Formats

The Consultative Committee for Space Data Systems (CCSDS) defines international standards for exchanging orbital data between space agencies, satellite operators, and ground systems. Brahe implements three Orbit Data Message (ODM) types from the [CCSDS 502.0-B-3 standard](https://ccsds.org/Pubs/502x0b3e1.pdf) (affectionately known as the "Blue Book"), enabling interoperability with other astrodynamics tools such as STK, GMAT, Orekit as well as space traffic management, ground station, and other mission operation support systems.

## Message Types

CCSDS ODM defines several message types. Brahe supports the three most widely used:

<div class="center-table" markdown="1">

| Message | Full Name | Contents | Typical Use |
|---------|-----------|----------|-------------|
| **[OEM](oem.md)** | Orbit Ephemeris Message | Time-series of state vectors (position/velocity) with optional covariance | Ephemeris exchange, trajectory handoffs, conjunction screening |
| **[OMM](omm.md)** | Orbit Mean-elements Message | Mean Keplerian elements + TLE parameters | GP/TLE data distribution (CelesTrak, Space-Track) |
| **[OPM](opm.md)** | Orbit Parameter Message | Single state vector + optional Keplerian elements, maneuvers, covariance | Initial conditions, maneuver plans, state handoffs |

</div>

!!! info "OCM Not Yet Supported"
    The Orbit Comprehensive Message (OCM) is not yet implemented. OCM combines features of OEM, OMM, and OPM into a single flexible format and will be added in a future release.

## Encoding Formats

Each message type can be encoded in three formats. Brahe auto-detects the format when parsing:

<div class="center-table" markdown="1">

| Format | Extension | Description |
|--------|-----------|-------------|
| **KVN** | `.oem`, `.omm`, `.opm`, `.txt` | Keyword=Value Notation — the original text format. Human-readable, line-oriented. |
| **XML** | `.xml` | XML encoding using CCSDS element names. Structured, schema-validatable. |
| **JSON** | `.json` | JSON encoding mirroring the XML structure. Machine-friendly. |

</div>

Format detection examines the first non-whitespace character of the content:

- Starts with `<` or `<?xml` → XML
- Starts with `{` or `[` → JSON
- Otherwise → KVN

## Unit Conventions

CCSDS files use **km** and **km/s** for position and velocity. Brahe automatically converts these to **SI base units** (meters, m/s) on parse, and converts back when writing. This means:

- All position values returned by brahe are in **meters**
- All velocity values returned by brahe are in **m/s**
- Covariance matrices are in **m$^2$**, **m$^2$/s**, and **m$^2$/s$^2$**
- Angles in OMM and OPM Keplerian elements remain in **degrees** (matching CCSDS/TLE convention)
- GM values are converted from km$^3$/s$^2$ to **m$^3$/s$^2$**

!!! warning "Unit Conversion"
    If you compare brahe output directly to values in a CCSDS file, remember the factor of 1000 for position/velocity and 10$^9$ for GM. Covariance elements scale by 10$^6$ (km$^2$ → m$^2$).

## Quick Start

All three message types follow the same interface pattern:

```python
from brahe.ccsds import OEM, OMM, OPM

# Parse from file (format auto-detected)
oem = OEM.from_file("ephemeris.oem")
omm = OMM.from_file("mean_elements.xml")
opm = OPM.from_file("state_vector.json")

# Parse from string
oem = OEM.from_str(content_string)

# Write to KVN format
kvn_string = oem.to_string("KVN")
oem.to_file("output.oem", "KVN")

# Serialize to dictionary
d = oem.to_dict()
```

Each type provides accessor methods for the data it contains, and a `to_dict()` method that returns the full message as a nested Python dictionary for easy serialization.

---

## See Also

- [OEM — Orbit Ephemeris Message](oem.md)
- [OMM — Orbit Mean-elements Message](omm.md)
- [OPM — Orbit Parameter Message](opm.md)
- [API Reference](../../library_api/ccsds/index.md) — Python API documentation
