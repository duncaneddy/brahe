//! Example demonstrating how to create Epoch objects from different time representations.

use approx::assert_abs_diff_eq;
use brahe::constants::MJD_ZERO;
use brahe::time::{Epoch, TimeSystem};

fn main() {
    // Create Epoch from datetime components
    // (year, month, day, hour, minute, second, nanosecond, time_system)
    let epoch1 = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

    // Create Epoch from Julian Date
    let jd = 2460311.0; // 2024-01-01 12:00:00 UTC
    let epoch2 = Epoch::from_jd(jd, TimeSystem::UTC);

    // Create Epoch from Modified Julian Date
    let mjd = jd - MJD_ZERO;
    let epoch3 = Epoch::from_mjd(mjd, TimeSystem::UTC);

    // All three should represent the same time
    assert_abs_diff_eq!(epoch1.jd(), epoch2.jd(), epsilon = 1e-10);
    assert_abs_diff_eq!(epoch1.jd(), epoch3.jd(), epsilon = 1e-10);

    println!("✓ All epoch creation methods validated successfully!");
}
