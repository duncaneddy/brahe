# Constants

Mathematical, physical, and astronomical constants used throughout the brahe library.

## Categories

### [Units](units.md)
Angle format enumerations and time system constants for specifying units and reference frames.

### [Mathematical Constants](math.md)
Conversion factors for angles and other mathematical operations.

### [Time Constants](time.md)
Julian date references and time system offset values.

### [Physical Constants](physical.md)
Physical properties of Earth, celestial bodies, and universal constants.

## Quick Reference

All constants use SI base units unless otherwise noted:

- **Distance**: meters (m)
- **Time**: seconds (s)
- **Angles**: radians (rad)
- **Gravitational Parameter**: m³/s²

Constants are accessible directly from the `brahe` module:

```python
import brahe as bh

# Mathematical constants
angle_rad = 45.0 * bh.DEG2RAD  # Convert degrees to radians

# Physical constants
mu_earth = bh.GM_EARTH  # Earth's gravitational parameter
c = bh.C_LIGHT          # Speed of light

# Time system
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
```
