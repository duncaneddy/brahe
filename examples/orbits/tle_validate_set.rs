//! Validate a complete TLE set.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // ISS TLE (NORAD ID 25544)
    let line1 = "1 25544U 98067A   25302.48953544  .00013618  00000-0  24977-3 0  9995";
    let line2 = "2 25544  51.6347   1.5519 0004808 353.3325   6.7599 15.49579513535999";

    // Validate the complete TLE set (both lines must have matching NORAD IDs)
    let is_valid = bh::validate_tle_lines(line1, line2);
    println!("TLE set valid: {}", is_valid);

    // Validate individual lines
    let line1_valid = bh::validate_tle_line(line1);
    let line2_valid = bh::validate_tle_line(line2);
    println!("Line 1 valid: {}", line1_valid);
    println!("Line 2 valid: {}", line2_valid);

    // Expected output:
    // TLE set valid: true
    // Line 1 valid: true
    // Line 2 valid: true
}
