# Units

Enumerations for specifying angle formats.

## Angle Format

The `AngleFormat` enumeration specifies whether angles are in radians or degrees.

::: brahe.AngleFormat
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - RADIANS
        - DEGREES
      show_bases: false
      heading_level: 3

### Usage Example

```python
import brahe as bh
import numpy as np

# Create rotation with angle in degrees
q = bh.Quaternion.from_euler_axis(
    axis=np.array([0.0, 0.0, 1.0]),
    angle=90.0,
    angle_format=bh.AngleFormat.DEGREES
)

# Create rotation with angle in radians
q2 = bh.Quaternion.from_euler_axis(
    axis=np.array([0.0, 0.0, 1.0]),
    angle=np.pi/2,
    angle_format=bh.AngleFormat.RADIANS
)
```
