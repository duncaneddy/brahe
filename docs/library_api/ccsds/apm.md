# APM — Attitude Parameter Message

Parses CCSDS Attitude Parameter Messages containing a single-epoch attitude state through one or more logical blocks: quaternion, Euler angle, angular velocity, spin, inertia, and maneuver.

---

::: brahe.ccsds.APM
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.ccsds.APMQuaternionState
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.ccsds.APMEulerState
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.ccsds.APMAngularVelocity
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.ccsds.APMSpin
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.ccsds.APMInertia
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.ccsds.APMManeuver
    options:
      show_root_heading: true
      show_root_full_path: false

---

## See Also

- [APM Format Guide](../../learn/ccsds/apm.md) — Structure, logical blocks, and unit conventions
- [CCSDS Module](index.md) — Module overview
