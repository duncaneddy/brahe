# Rotation Matrix

Direction Cosine Matrix (DCM) representation for spacecraft attitude.

## Class Documentation

The `RotationMatrix` class provides a Direction Cosine Matrix (DCM) representation for spacecraft attitude with conversion methods to other attitude representations.

```python
from brahe import RotationMatrix
import numpy as np

# Create a rotation matrix
R = RotationMatrix(np.eye(3))
```

For complete API documentation, see the [Rust API documentation](https://docs.rs/brahe/latest/brahe/attitude/struct.RotationMatrix.html).
