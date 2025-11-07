# /// script
# dependencies = ["brahe"]
# ///
"""
SamplingConfig examples showing different sampling modes
"""

import brahe as bh

# Single sample at window midpoint (default)
config = bh.SamplingConfig.midpoint()
print(f"Midpoint: {config}")
# Midpoint: SamplingConfig.Midpoint

# Specific relative points [0.0, 1.0] from window start to end
config = bh.SamplingConfig.relative_points([0.0, 0.25, 0.5, 0.75, 1.0])
print(f"Relative points: {config}")
# Relative points: SamplingConfig.RelativePoints([0.0, 0.25, 0.5, 0.75, 1.0])

# Fixed time interval in seconds
config = bh.SamplingConfig.fixed_interval(1.0, offset=0.0)  # 1 second
print(f"Fixed interval (1s): {config}")
# Fixed interval (1s): SamplingConfig.FixedInterval(interval=1.0, offset=0.0)

# Fixed number of evenly-spaced points
config = bh.SamplingConfig.fixed_count(50)
print(f"Fixed count (50): {config}")
# Fixed count (50): SamplingConfig.FixedCount(50)
