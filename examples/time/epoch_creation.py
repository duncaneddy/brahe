# /// script
# dependencies = ["brahe", "pytest"]
# ///
"""
Example demonstrating how to create Epoch objects from different time representations.
"""

import brahe as bh
import pytest

if __name__ == "__main__":
    # Create Epoch from datetime components
    # (year, month, day, hour, minute, second, nanosecond, time_system)
    epoch1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

    # Create Epoch from Julian Date
    jd = 2460311.0  # 2024-01-01 12:00:00 UTC
    epoch2 = bh.Epoch.from_jd(jd, bh.TimeSystem.UTC)

    # Create Epoch from Modified Julian Date
    mjd = jd - bh.MJD_ZERO
    epoch3 = bh.Epoch.from_mjd(mjd, bh.TimeSystem.UTC)

    # All three should represent the same time
    assert epoch1.jd() == pytest.approx(epoch2.jd(), abs=1e-10)
    assert epoch1.jd() == pytest.approx(epoch3.jd(), abs=1e-10)

    print("✓ All epoch creation methods validated successfully!")
