# Euler Axis

Euler axis-angle representation for spacecraft attitude.

## Class Documentation

The `EulerAxis` class provides an Euler axis-angle representation for spacecraft attitude with conversion methods to other attitude representations.

```python
from brahe import EulerAxis
import numpy as np

# Create Euler axis with axis vector and angle
axis = np.array([1.0, 0.0, 0.0])
angle = 0.5
ea = EulerAxis(axis, angle)
```

For complete API documentation, see the [Rust API documentation](https://docs.rs/brahe/latest/brahe/attitude/struct.EulerAxis.html).
