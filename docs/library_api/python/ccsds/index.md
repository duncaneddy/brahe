# CCSDS Orbit Data Messages

API reference for the CCSDS module. All types are available via `brahe.ccsds`.

Brahe supports three CCSDS Orbit Data Message (ODM) types defined in CCSDS 502.0-B-3, with automatic format detection for KVN, XML, and JSON encodings.

- **[OEM](oem.md)** — Orbit Ephemeris Message (time-series state vectors)
- **[OMM](omm.md)** — Orbit Mean-elements Message (SGP4/TLE data)
- **[OPM](opm.md)** — Orbit Parameter Message (single state vector)
- **[CDM](cdm.md)** — Conjunction Data Message (two-object close approach)

---

## See Also

- [CCSDS Data Formats Guide](../../learn/ccsds/index.md) — What CCSDS is, format details, and usage patterns
