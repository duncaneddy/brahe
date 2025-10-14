# Euler Angles

Euler angle sequence representation for spacecraft attitude.

## Class Documentation

The `EulerAngle` class provides an Euler angle sequence representation for spacecraft attitude with conversion methods to other attitude representations.

```python
from brahe import EulerAngle

# Create Euler angles with a 3-1-3 sequence
euler = EulerAngle(0.1, 0.2, 0.3, "313")
```

For complete API documentation, see the [Rust API documentation](https://docs.rs/brahe/latest/brahe/attitude/struct.EulerAngle.html).
