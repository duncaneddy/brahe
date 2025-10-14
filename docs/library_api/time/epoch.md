# Epoch Class

The `Epoch` class is the foundational time representation in Brahe, providing comprehensive support for multiple time systems and high-precision time computations.

## Class Documentation

The `Epoch` class provides high-precision time representation with support for multiple time systems including UTC, TAI, TT, GPS, and UT1.

```python
from brahe import Epoch

# Create epochs in different ways
epc1 = Epoch.now()  # Current time
epc2 = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0, "UTC")  # From date/time
epc3 = Epoch.from_mjd(60000.0, "TAI")  # From Modified Julian Date

# Convert between time systems
mjd_utc = epc1.mjd("UTC")
mjd_tai = epc1.mjd("TAI")

# Time arithmetic
future_epc = epc1 + 3600  # Add 3600 seconds
time_diff = epc2 - epc1   # Difference in seconds
```

For complete API documentation, see the [Rust API documentation](https://docs.rs/brahe/latest/brahe/time/struct.Epoch.html).
