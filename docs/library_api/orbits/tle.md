# Two-Line Element (TLE)

Classes and functions for working with Two-Line Element sets.

## TLE Class

The `TLE` class represents a Two-Line Element set for satellite orbit description.

```python
from brahe import TLE

# Create TLE from strings
tle = TLE(
    "ISS (ZARYA)",
    "1 25544U 98067A   21001.00000000  .00002182  00000-0  41420-4 0  9990",
    "2 25544  51.6461 339.8014 0002571  34.5857 120.4689 15.48919393265104"
)

# Access TLE properties
print(f"Satellite: {tle.name}")
print(f"NORAD ID: {tle.norad_id}")
print(f"Inclination: {tle.inclination} deg")
```

For complete API documentation, see the [Rust API documentation](https://docs.rs/brahe/latest/brahe/orbits/struct.TLE.html).

## TLE Utility Functions

::: brahe.orbits.epoch_from_tle

::: brahe.orbits.keplerian_elements_from_tle

::: brahe.orbits.keplerian_elements_to_tle

::: brahe.orbits.create_tle_lines

::: brahe.orbits.validate_tle_line

::: brahe.orbits.validate_tle_lines

::: brahe.orbits.calculate_tle_line_checksum

::: brahe.orbits.parse_norad_id

::: brahe.orbits.norad_id_numeric_to_alpha5

::: brahe.orbits.norad_id_alpha5_to_numeric