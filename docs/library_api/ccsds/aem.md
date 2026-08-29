# AEM — Attitude Ephemeris Message

Parses CCSDS Attitude Ephemeris Messages containing time-ordered attitude segments across all nine 504.0-B-2 attitude data types, and converts segments to and from native `AttitudeTrajectory` objects.

---

::: brahe.ccsds.AEM
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.ccsds.AEMSegment
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.ccsds.AEMAttitudeState
    options:
      show_root_heading: true
      show_root_full_path: false

---

## See Also

- [AEM Format Guide](../../learn/ccsds/aem.md) — Segments, attitude types, and `AttitudeTrajectory` conversion
- [AttitudeTrajectory](../trajectories/attitude_trajectory.md) — Native attitude trajectory storage and interpolation
- [CCSDS Module](index.md) — Module overview
