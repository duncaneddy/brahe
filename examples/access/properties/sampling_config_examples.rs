//! ```cargo
//! [dependencies]
//! brahe = { path = "../../.." }
//! ```

use brahe::access::SamplingConfig;

fn main() {
    // Single sample at window midpoint (default)
    let config = SamplingConfig::Midpoint;
    println!("Midpoint: {:?}", config);
    // Midpoint: Midpoint

    // Specific relative points [0.0, 1.0] from window start to end
    let config = SamplingConfig::RelativePoints(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    println!("Relative points: {:?}", config);
    // Relative points: RelativePoints([0.0, 0.25, 0.5, 0.75, 1.0])

    // Fixed time interval in seconds
    let config = SamplingConfig::FixedInterval {
        interval: 1.0,  // 1 second
        offset: 0.0
    };
    println!("Fixed interval (1s): {:?}", config);
    // Fixed interval (1s): FixedInterval { interval: 1.0, offset: 0.0 }

    // Fixed number of evenly-spaced points
    let config = SamplingConfig::FixedCount(50);
    println!("Fixed count (50): {:?}", config);
    // Fixed count (50): FixedCount(50)
}
